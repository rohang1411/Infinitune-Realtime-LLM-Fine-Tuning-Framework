"""Validation of complete Kafka adapter snapshots before they reach a model.

Protocol v2 Features:
- Per-tensor SHA-256 verification before deserialization
- Canonical JSON manifest digests validated on commit
- Optional HMAC-SHA256 signature verification over manifest
- Monotonic version guard preventing replay or stale rollbacks
- Safe deserialization supporting safetensors payloads
"""

import hashlib
import hmac
import json
import logging
import os
from typing import Optional, Union, Dict, Any, Tuple
import torch


def canonical_json(obj: Any) -> bytes:
    """Return stable, deterministic UTF-8 encoded canonical JSON bytes.
    Uses sort_keys=True, compact separators, and ASCII-only escaping.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def manifest_digest(manifest: dict) -> str:
    """Compute SHA-256 hex digest of the canonical JSON manifest without hmac_sha256 field."""
    manifest_without_hmac = {k: v for k, v in manifest.items() if k != "hmac_sha256"}
    return hashlib.sha256(canonical_json(manifest_without_hmac)).hexdigest()


def compute_manifest_hmac(manifest: dict, key: Union[str, bytes]) -> str:
    """Compute HMAC-SHA256 hex digest over the canonical manifest JSON."""
    key_bytes = key.encode("utf-8") if isinstance(key, str) else key
    manifest_without_hmac = {k: v for k, v in manifest.items() if k != "hmac_sha256"}
    return hmac.new(key_bytes, canonical_json(manifest_without_hmac), hashlib.sha256).hexdigest()


def classify_rejection_reason(exc: Union[Exception, str]) -> str:
    """Map an exception or error string to a canonical rejection reason code."""
    msg = str(exc).lower()
    if "sha256_mismatch" in msg:
        return "sha256_mismatch"
    if "incomplete adapter step" in msg or "incomplete_snapshot" in msg:
        return "incomplete_snapshot"
    if "duplicate adapter tensor" in msg or "duplicate_tensor" in msg:
        return "duplicate_tensor"
    if "undeclared adapter tensor" in msg or "undeclared_tensor" in msg:
        return "undeclared_tensor"
    if "shape_or_dtype_mismatch" in msg or "shape_mismatch" in msg or "nbytes mismatch" in msg:
        return "shape_mismatch"
    if "unknown or expired step" in msg or "unknown_step" in msg:
        return "unknown_step"
    if "commit manifest_sha256 mismatch" in msg or "manifest_digest_mismatch" in msg:
        return "manifest_digest_mismatch"
    if "malformed manifest json" in msg or "bad_manifest" in msg or "manifest must be dict" in msg:
        return "bad_manifest"
    if "stale_version" in msg:
        return "stale_version"
    if "invalid manifest hmac signature" in msg or "hmac_sha256" in msg or "hmac" in msg:
        return "hmac_mismatch"
    if "rejected v1 manifest" in msg or "protocol_version 2 required" in msg or "protocol_version" in msg:
        return "protocol_version"
    if isinstance(exc, Exception):
        return type(exc).__name__
    return "unknown_error"


def deserialize_tensor(value_bytes: bytes):
    """Decode safetensors or JSON metadata; never unpickle broker bytes."""
    if not value_bytes:
        return None

    # 1. Safetensors format (fast, zero-copy, safe)
    try:
        from safetensors.torch import load
        loaded = load(value_bytes)
        if isinstance(loaded, dict) and "weight" in loaded:
            return loaded["weight"]
        return loaded
    except Exception:
        pass

    # 2. JSON control / manifest packet
    try:
        if value_bytes.startswith(b"{") and value_bytes.endswith(b"}"):
            return json.loads(value_bytes.decode("utf-8"))
    except Exception:
        pass

    raise ValueError("Kafka value is neither a safetensors tensor nor JSON metadata")


class SnapshotAssembler:
    """Buffers incoming adapter snapshot frames and commits only complete, verified snapshots.

    Protocol v2 validates per-tensor SHA-256 before deserialization, validates the commit's
    manifest_sha256 against the manifest digest, verifies optional HMAC signatures, and
    enforces monotonic run_id / sequence order.
    """

    def __init__(
        self,
        max_pending: int = 8,
        integrity: str = "sha256",
        hmac_key: Optional[Union[str, bytes]] = None,
        initial_active: Optional[Tuple[str, Any, float]] = None,
    ):
        self.max_pending = max_pending
        self.integrity = integrity  # "sha256" | "hmac" | "none_v1_compat"
        if self.integrity == "hmac":
            key = hmac_key or os.environ.get("INFINITUNE_ADAPTER_HMAC_KEY")
            if not key:
                raise ValueError("HMAC integrity mode requested but INFINITUNE_ADAPTER_HMAC_KEY is missing")
            self.hmac_key = key
        else:
            self.hmac_key = hmac_key or os.environ.get("INFINITUNE_ADAPTER_HMAC_KEY")

        # Monotonic version state: (run_id, seq, created_at)
        self.active: Optional[Tuple[str, Any, float]] = initial_active

        self.manifests: Dict[str, dict] = {}
        self.batches: Dict[str, Dict[str, torch.Tensor]] = {}
        self.manifest_digests: Dict[str, Optional[str]] = {}

    def _check_monotonic(self, run_id: str, seq: Any, created_at: float):
        if self.active is not None:
            act_run_id, act_seq, act_created_at = self.active
            if run_id == act_run_id:
                try:
                    s_new = float(seq)
                    s_act = float(act_seq)
                    if s_new <= s_act:
                        raise ValueError(f"stale_version: seq {seq} <= active seq {act_seq} for run {run_id}")
                except (ValueError, TypeError):
                    if seq <= act_seq:
                        raise ValueError(f"stale_version: seq {seq} <= active seq {act_seq} for run {run_id}")
            else:
                if created_at <= act_created_at:
                    raise ValueError(
                        f"stale_version: created_at {created_at} <= active created_at {act_created_at} for new run {run_id}"
                    )

    def add(self, key: str, value: Any) -> Optional[Tuple[str, Dict[str, torch.Tensor]]]:
        """Process one incoming adapter record (manifest, tensor, or commit).
        Returns (step, batch_dict) on validated commit, else None.
        Raises ValueError or TypeError on protocol violations.
        """
        if not key:
            raise ValueError("Adapter record key cannot be empty")

        # ----------------------------------------------------
        # 1. Manifest marker: __manifest__:<step>
        # ----------------------------------------------------
        if key.startswith("__manifest__:"):
            step = key.split(":", 1)[1]
            if not step:
                raise ValueError("Invalid manifest key without step")

            if isinstance(value, bytes):
                try:
                    manifest_data = json.loads(value.decode("utf-8"))
                except Exception as exc:
                    raise ValueError(f"Malformed manifest JSON: {exc}")
            elif isinstance(value, dict):
                manifest_data = value
            else:
                raise ValueError("Manifest must be dict or JSON bytes")

            proto_version = manifest_data.get("protocol_version")
            if proto_version == 2:
                # Protocol v2 verification
                run_id = manifest_data.get("run_id")
                seq = manifest_data.get("seq")
                created_at = manifest_data.get("created_at")
                names = manifest_data.get("tensor_names")
                count = manifest_data.get("num_tensors")
                tensors = manifest_data.get("tensors")
                manifest_step = str(manifest_data.get("step_id"))
                msg_integrity = manifest_data.get("integrity", "sha256")

                if not run_id or not isinstance(run_id, str):
                    raise ValueError("Protocol v2 manifest missing valid run_id")
                if seq is None:
                    raise ValueError("Protocol v2 manifest missing seq")
                if created_at is None:
                    raise ValueError("Protocol v2 manifest missing created_at")
                if manifest_step != step:
                    raise ValueError(f"Manifest step_id {manifest_step!r} != key step {step!r}")
                if not isinstance(names, list) or not names or len(set(names)) != len(names):
                    raise ValueError("Protocol v2 manifest has invalid tensor_names list")
                if count != len(names):
                    raise ValueError(f"num_tensors {count} != len(tensor_names) {len(names)}")
                if not isinstance(tensors, dict) or set(tensors.keys()) != set(names):
                    raise ValueError("Protocol v2 manifest tensors dict does not match tensor_names")

                # Validate tensor entries in manifest
                for tname, tmeta in tensors.items():
                    if not isinstance(tmeta, dict) or "sha256" not in tmeta or "shape" not in tmeta or "dtype" not in tmeta:
                        raise ValueError(f"Incomplete tensor metadata in manifest for {tname}")

                # HMAC verification
                if self.integrity == "hmac" or msg_integrity == "hmac":
                    hmac_sig = manifest_data.get("hmac_sha256")
                    if not hmac_sig:
                        raise ValueError("Protocol v2 manifest missing required hmac_sha256 signature")
                    expected_hmac = compute_manifest_hmac(manifest_data, self.hmac_key)
                    if not hmac.compare_digest(hmac_sig, expected_hmac):
                        raise ValueError("Invalid manifest HMAC signature")

                # Check monotonic version guard
                self._check_monotonic(run_id, seq, float(created_at))

                digest = manifest_digest(manifest_data)

            else:
                # Protocol v1
                if self.integrity != "none_v1_compat":
                    raise ValueError(
                        f"Rejected v1 manifest: protocol_version 2 required (got {proto_version!r}). "
                        f"Configure kafka.integrity='none_v1_compat' to allow legacy v1 manifests."
                    )
                names = manifest_data.get("tensor_names") if isinstance(manifest_data, dict) else None
                count = manifest_data.get("num_tensors") if isinstance(manifest_data, dict) else None
                if (not step or not isinstance(names, list) or not names
                        or any(not isinstance(n, str) or not n for n in names)
                        or len(set(names)) != len(names) or count != len(names)
                        or str(manifest_data.get("step_id")) != step):
                    raise ValueError(f"Invalid adapter manifest for step {step!r}")
                digest = None

            if len(self.manifests) >= self.max_pending:
                oldest = next(iter(self.manifests))
                logging.warning("Discarding stale uncommitted adapter step %s", oldest)
                self.manifests.pop(oldest, None)
                self.batches.pop(oldest, None)
                self.manifest_digests.pop(oldest, None)

            self.manifests[step] = manifest_data
            self.batches[step] = {}
            self.manifest_digests[step] = digest
            return None

        # ----------------------------------------------------
        # 2. Commit marker: __commit__:<step>
        # ----------------------------------------------------
        if key.startswith("__commit__:"):
            step = key.split(":", 1)[1]
            expected_manifest = self.manifests.pop(step, None)
            batch = self.batches.pop(step, {})
            stored_digest = self.manifest_digests.pop(step, None)

            if expected_manifest is None:
                raise ValueError(f"Rejected commit for unknown or expired step {step}")

            if isinstance(value, bytes):
                try:
                    commit_data = json.loads(value.decode("utf-8"))
                except Exception as exc:
                    raise ValueError(f"Malformed commit JSON: {exc}")
            elif isinstance(value, dict):
                commit_data = value
            else:
                raise ValueError("Commit must be dict or JSON bytes")

            if (commit_data.get("status") != "committed" or str(commit_data.get("step_id")) != step):
                raise ValueError(f"Invalid commit status or step_id for step {step}")

            expected_names = set(expected_manifest.get("tensor_names", []))
            if set(batch.keys()) != expected_names:
                raise ValueError(
                    f"Rejected incomplete adapter step {step}: "
                    f"expected={sorted(expected_names)}, received={sorted(batch.keys())}"
                )

            # Protocol v2 digest check & update monotonic active state
            if expected_manifest.get("protocol_version") == 2:
                commit_digest = commit_data.get("manifest_sha256")
                if not commit_digest or commit_digest != stored_digest:
                    raise ValueError(
                        f"Commit manifest_sha256 mismatch: expected {stored_digest}, got {commit_digest}"
                    )
                self.active = (
                    expected_manifest["run_id"],
                    expected_manifest["seq"],
                    float(expected_manifest["created_at"]),
                )

            return step, batch

        # ----------------------------------------------------
        # 3. Tensor frame: <step>:<name>
        # ----------------------------------------------------
        if ":" in key and not key.startswith("__"):
            step, name = key.split(":", 1)
            if step not in self.manifests:
                raise ValueError(f"Undeclared adapter tensor {key!r}")
            manifest = self.manifests[step]
            if name not in manifest.get("tensor_names", []):
                raise ValueError(f"Undeclared adapter tensor {key!r}")
            if name in self.batches[step]:
                raise ValueError(f"Duplicate adapter tensor {key!r}")

            if manifest.get("protocol_version") == 2:
                if not isinstance(value, bytes):
                    raise TypeError(f"Protocol v2 requires raw bytes for tensor frame {key!r}")

                tmeta = manifest["tensors"][name]
                actual_sha = hashlib.sha256(value).hexdigest()
                expected_sha = tmeta["sha256"]
                if actual_sha != expected_sha:
                    raise ValueError(
                        f"sha256_mismatch for tensor {name}: expected {expected_sha}, got {actual_sha}"
                    )

                if "nbytes" in tmeta and len(value) != tmeta["nbytes"]:
                    raise ValueError(
                        f"nbytes mismatch for tensor {name}: expected {tmeta['nbytes']}, got {len(value)}"
                    )

                tensor = deserialize_tensor(value)
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"Payload for tensor {key!r} is not a Tensor")

                actual_dtype = str(tensor.dtype).replace("torch.", "")
                expected_dtype = str(tmeta["dtype"]).replace("torch.", "")
                actual_shape = list(tensor.shape)
                expected_shape = list(tmeta["shape"])
                if actual_dtype != expected_dtype or actual_shape != expected_shape:
                    raise ValueError(
                        f"shape_or_dtype_mismatch for tensor {name}: "
                        f"expected dtype={expected_dtype}, shape={expected_shape}; "
                        f"got dtype={actual_dtype}, shape={actual_shape}"
                    )

                self.batches[step][name] = tensor
                return None
            else:
                # v1 compat
                if isinstance(value, bytes):
                    tensor = deserialize_tensor(value)
                elif isinstance(value, torch.Tensor):
                    tensor = value
                else:
                    raise TypeError(f"Adapter tensor {key!r} is not a tensor")

                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"Adapter tensor {key!r} is not a tensor")

                self.batches[step][name] = tensor
                return None

        raise ValueError(f"Unsupported adapter message key {key!r}")
