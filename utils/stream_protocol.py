"""Validation of complete Kafka adapter snapshots before they reach a model.

Kafka preserves order within a partition, but a lost record, restart, or malformed
producer can still leave a partial snapshot. A commit is therefore accepted only
when its manifest and *exact* declared tensor set have arrived.
"""

import logging
from typing import Optional

import torch


class SnapshotAssembler:
    def __init__(self, max_pending: int = 8):
        self.manifests = {}
        self.batches = {}
        self.max_pending = max_pending

    def add(self, key: str, value) -> Optional[tuple[str, dict]]:
        """Return (step, weights) only for a validated committed snapshot."""
        if key.startswith("__manifest__:"):
            step = key.split(":", 1)[1]
            names = value.get("tensor_names") if isinstance(value, dict) else None
            count = value.get("num_tensors") if isinstance(value, dict) else None
            if (not step or not isinstance(names, list) or not names
                    or any(not isinstance(n, str) or not n for n in names)
                    or len(set(names)) != len(names) or count != len(names)
                    or str(value.get("step_id")) != step):
                raise ValueError(f"Invalid adapter manifest for step {step!r}")
            if len(self.manifests) >= self.max_pending:
                oldest = next(iter(self.manifests))
                logging.warning("Discarding stale uncommitted adapter step %s", oldest)
                self.manifests.pop(oldest, None)
                self.batches.pop(oldest, None)
            self.manifests[step] = set(names)
            self.batches[step] = {}
            return None

        if key.startswith("__commit__:"):
            step = key.split(":", 1)[1]
            expected = self.manifests.pop(step, None)
            batch = self.batches.pop(step, {})
            if (expected is None or not isinstance(value, dict)
                    or value.get("status") != "committed"
                    or str(value.get("step_id")) != step
                    or set(batch) != expected):
                raise ValueError(
                    f"Rejected incomplete adapter step {step}: "
                    f"expected={sorted(expected or [])}, received={sorted(batch)}"
                )
            return step, batch

        if ":" in key and not key.startswith("__"):
            step, name = key.split(":", 1)
            if step not in self.manifests or name not in self.manifests[step]:
                raise ValueError(f"Undeclared adapter tensor {key!r}")
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Adapter tensor {key!r} is not a tensor")
            if name in self.batches[step]:
                raise ValueError(f"Duplicate adapter tensor {key!r}")
            self.batches[step][name] = value
            return None

        raise ValueError(f"Unsupported adapter message key {key!r}")
