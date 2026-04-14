"""
utils/checkpoint_manager.py
─────────────────────────────────────────────────────────────────────────────
CheckpointManager — Versioned LoRA adapter checkpoint saving and loading.

Design principles
-----------------
- Saves ONLY the LoRA adapter (~5-20 MB), not the full base model.
  The base model is always re-loaded from HuggingFace cache at eval time.
- Step directories are NEVER overwritten (skipped if they already exist).
  Only "final/" is overwritten — it always represents the latest endpoint.
- Directory naming: <output_dir>/checkpoints/<model>__<dataset>/step_XXXX/
  Zero-padded step numbers ensure natural alphabetical sorting.
- checkpoint_meta.json records provenance so evaluate.py needs only the
  checkpoint directory (not the original config path) to reconstruct the
  model.

Directory layout
----------------
<output_dir>/
  checkpoints/
    <model>__<dataset>/          ← checkpoint_root
      step_0100/
        adapter_model.safetensors
        adapter_config.json
        checkpoint_meta.json
      step_0200/
        ...
      final/
        adapter_model.safetensors
        adapter_config.json
        checkpoint_meta.json
"""

import json
import os
import re
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}][CHECKPOINT] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Directory-name helpers
# ---------------------------------------------------------------------------

_PAD_WIDTH = 6   # "step_000100" — supports up to 999,999 steps


def _step_dir_name(step) -> str:
    """Return the directory name for a given step. 'final' stays as-is."""
    if step == "final":
        return "final"
    return f"step_{int(step):0{_PAD_WIDTH}d}"


def _slugify(text: str) -> str:
    """Replace characters that are problematic in directory names with '_'."""
    # Keep alphanumeric, dash, dot; replace everything else (including '/')
    return re.sub(r"[^A-Za-z0-9.\-]", "_", text)


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Manages saving and loading of LoRA adapter checkpoints.

    Parameters
    ----------
    config : dict
        Full YAML config dict (must contain 'project', 'model', 'dataset').
    config_path : str | None
        Path to the YAML file that generated *config*.  Written into
        checkpoint_meta.json for traceability.
    """

    def __init__(self, config: dict, config_path: Optional[str] = None):
        self._config = config
        self._config_path = config_path or "(unknown)"

        project_cfg  = config.get("project", {})
        model_cfg    = config.get("model", {})
        dataset_cfg  = config.get("dataset", {})

        output_dir   = project_cfg.get("output_dir", "./output")
        model_name   = model_cfg.get("name", "model")
        dataset_name = dataset_cfg.get("name", "dataset")

        # Friendly slug: "distilgpt2__stanfordnlp_imdb"
        model_slug   = _slugify(os.path.basename(model_name))
        dataset_slug = _slugify(dataset_name)
        scope_dir    = f"{model_slug}__{dataset_slug}"

        self._model_name   = model_name
        self._dataset_name = dataset_name
        self._lora_cfg     = config.get("lora", {})

        self.checkpoint_root = os.path.join(output_dir, "checkpoints", scope_dir)
        os.makedirs(self.checkpoint_root, exist_ok=True)
        _log(f"CheckpointManager initialised. Root: {self.checkpoint_root}")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def save(
        self,
        model,
        tokenizer,
        step,
        force: bool = False,
        loss: Optional[float] = None,
    ) -> Optional[str]:
        """
        Save LoRA adapter + tokenizer + metadata to a versioned directory.

        Parameters
        ----------
        model     : PEFT model currently in training.
        tokenizer : HuggingFace tokenizer.
        step      : int (optimizer step number) or the string "final".
        force     : If True, overwrite even if the directory already exists.
                    Defaults to False for step-numbered saves; True is used
                    internally for "final".
        loss      : Optional training loss at this step, written to metadata.

        Returns
        -------
        The checkpoint directory path if saved, None if skipped.
        """
        dir_name  = _step_dir_name(step)
        save_path = os.path.join(self.checkpoint_root, dir_name)

        # "final" always overwrites; numbered steps never overwrite (unless forced).
        if step == "final":
            force = True

        if not force and os.path.isdir(save_path):
            _log(
                f"Checkpoint '{dir_name}' already exists — skipping save "
                f"(pass force=True to overwrite)."
            )
            return None

        os.makedirs(save_path, exist_ok=True)

        # --- Save LoRA adapter weights + tokenizer ---
        try:
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        except Exception as exc:
            _log(f"ERROR: Failed to save adapter to '{save_path}': {exc}")
            return None

        # --- Write checkpoint_meta.json ---
        meta = {
            "step": step,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_name": self._model_name,
            "dataset_name": self._dataset_name,
            "config_path": self._config_path,
            "lora": {k: v for k, v in self._lora_cfg.items()},
        }
        if loss is not None:
            meta["training_loss_at_save"] = round(float(loss), 6)

        meta_path = os.path.join(save_path, "checkpoint_meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as exc:
            _log(f"Warning: Could not write checkpoint_meta.json: {exc}")

        step_label = f"step {step}" if step != "final" else "final"
        _log(f"Checkpoint saved [{step_label}] → {save_path}")
        return save_path

    # -----------------------------------------------------------------------

    def list_checkpoints(self) -> list:
        """
        Return a sorted list of available checkpoints.

        Each entry is a dict:
            {
                "name":      "step_0100",
                "step":      100,          # int or "final"
                "path":      "/abs/path/step_0100",
                "timestamp": "2026-04-12T15:02:30",   # from meta, or None
            }

        Results are sorted: numbered steps first (ascending), then "final".
        """
        if not os.path.isdir(self.checkpoint_root):
            return []

        checkpoints = []
        for entry in os.scandir(self.checkpoint_root):
            if not entry.is_dir():
                continue
            name = entry.name
            path = entry.path

            # Only include directories that have an adapter config (valid ckpts)
            if not os.path.exists(os.path.join(path, "adapter_config.json")):
                continue

            meta_path = os.path.join(path, "checkpoint_meta.json")
            timestamp = None
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                    timestamp = meta.get("timestamp")
                except Exception:
                    pass

            if name == "final":
                step_val = "final"
            elif name.startswith("step_"):
                try:
                    step_val = int(name[len("step_"):])
                except ValueError:
                    continue  # malformed directory name
            else:
                continue

            checkpoints.append({
                "name":      name,
                "step":      step_val,
                "path":      path,
                "timestamp": timestamp,
            })

        # Sort: numbered steps ascending, then "final" last
        def sort_key(c):
            s = c["step"]
            return (1, 0) if s == "final" else (0, s)

        checkpoints.sort(key=sort_key)
        return checkpoints

    # -----------------------------------------------------------------------

    def get_checkpoint_path(self, step) -> str:
        """Return the expected path for a given step (does not check existence)."""
        return os.path.join(self.checkpoint_root, _step_dir_name(step))

    # -----------------------------------------------------------------------

    def load(self, checkpoint_path: str, device: str = "auto"):
        """
        Load a base model + LoRA adapter from a checkpoint directory.

        Reads checkpoint_meta.json for the model name; falls back to the
        config's model.name if metadata is absent.

        Returns
        -------
        (model, tokenizer)   — both fully loaded and ready for inference/eval.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {checkpoint_path}"
            )

        # --- Read metadata ---
        meta_path = os.path.join(checkpoint_path, "checkpoint_meta.json")
        model_name = self._model_name   # fallback
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                model_name = meta.get("model_name", model_name)
                _log(f"Checkpoint metadata: step={meta.get('step')}, "
                     f"saved={meta.get('timestamp')}, model={model_name}")
            except Exception as exc:
                _log(f"Warning: Could not read checkpoint_meta.json: {exc}")

        # --- Resolve device ---
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        _log(f"Loading base model '{model_name}' on device '{device}'...")

        # --- Load base model ---
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # --- Attach LoRA adapter ---
        _log(f"Attaching LoRA adapter from '{checkpoint_path}'...")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model.eval()

        _log("Model + adapter loaded successfully.")
        return model, tokenizer
