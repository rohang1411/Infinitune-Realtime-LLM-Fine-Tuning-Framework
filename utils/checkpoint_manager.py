"""
utils/checkpoint_manager.py
─────────────────────────────────────────────────────────────────────────────
CheckpointManager — Versioned LoRA adapter checkpoint saving and loading.

Design principles
-----------------
- Saves ONLY the LoRA adapter (~5-20 MB), not the full base model.
  The base model is always re-loaded from HuggingFace cache at eval time.
- Each training run gets its own unique subdirectory so multiple runs
  never collide. The "final" checkpoint inside a run is always overwritten
  by subsequent saves within that same run.
- Directory naming:
    <output_dir>/checkpoints/<model>__<dataset>/run_<ts>_<uid>/step_XXXX/
  Zero-padded step numbers ensure natural alphabetical sorting.
- checkpoint_meta.json records provenance so evaluate.py needs only the
  checkpoint directory (not the original config path) to reconstruct the
  model.
- list_checkpoints() discovers checkpoints from ALL past runs under the
  scope directory and includes backward-compatible discovery of legacy
  flat step_* directories from pre-versioning releases.

Directory layout
----------------
<output_dir>/
  checkpoints/
    <model>__<dataset>/          ← checkpoint_root (scope dir)
      run_20260413-153020_a3f2/  ← per-run subdir (created at init)
        step_000100/
          adapter_model.safetensors
          adapter_config.json
          checkpoint_meta.json
        step_000200/
          ...
        final/
          ...
      run_20260413-180045_b7c1/  ← second run; never collides with first
        step_000100/
        ...
"""

import json
import os
import re
import time
import uuid
from typing import Optional

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

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

        # Each CheckpointManager instance corresponds to exactly one training run.
        # Generate a unique run subdirectory so repeated runs never collide.
        _run_ts  = time.strftime("%Y%m%d-%H%M%S")
        _run_uid = uuid.uuid4().hex[:4]
        self._run_dir  = f"run_{_run_ts}_{_run_uid}"
        self._run_path = os.path.join(self.checkpoint_root, self._run_dir)
        # The run dir is created on first save, not at init (avoids empty dirs).

        _log(f"CheckpointManager initialised. Root: {self.checkpoint_root}")
        _log(f"Run directory: {self._run_dir}")

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
        save_path = os.path.join(self._run_path, dir_name)

        # "final" always overwrites within the same run; numbered steps never
        # overwrite (a step can't appear twice within a single run).
        if step == "final":
            force = True

        if not force and os.path.isdir(save_path):
            _log(
                f"Checkpoint '{dir_name}' already exists in run '{self._run_dir}' "
                f"— skipping save (pass force=True to overwrite)."
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

        # --- Write full config snapshot alongside the adapter ---
        # This ensures every checkpoint is self-contained: you can always
        # know exactly which hyperparameters produced it without needing the
        # original config file path.
        if _YAML_AVAILABLE and self._config:
            config_snapshot_path = os.path.join(save_path, "config_snapshot.yaml")
            try:
                with open(config_snapshot_path, "w", encoding="utf-8") as f:
                    _yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
            except Exception as exc:
                _log(f"Warning: Could not write config_snapshot.yaml: {exc}")

        step_label = f"step {step}" if step != "final" else "final"
        _log(f"Checkpoint saved [{step_label}] → {save_path} (run: {self._run_dir})")
        return save_path

    # -----------------------------------------------------------------------

    def list_checkpoints(self) -> list:
        """
        Return a sorted list of available checkpoints across ALL runs.

        Scans:
          1. run_* subdirectories (new hierarchical layout) — each run's
             step_* and final directories are discovered.
          2. Legacy flat step_* directories directly under checkpoint_root
             (from pre-versioning releases) for backward compatibility.

        Each entry is a dict:
            {
                "name":      "step_000100",
                "step":      100,           # int or "final"
                "path":      "/abs/path/run_xxx/step_000100",
                "run":       "run_20260413-153020_a3f2",  # None for legacy
                "timestamp": "2026-04-12T15:02:30",       # from meta, or None
            }

        Results are sorted: numbered steps ascending, then "final" last.
        """
        if not os.path.isdir(self.checkpoint_root):
            return []

        checkpoints = []

        def _load_step_dir(path: str, name: str, run_name):
            """Parse a single step directory and append to checkpoints."""
            if not os.path.exists(os.path.join(path, "adapter_config.json")):
                return  # not a valid checkpoint

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
                # Handle both "step_000200" and legacy "step_0200" widths
                numeric_part = name[len("step_"):]
                try:
                    step_val = int(numeric_part)
                except ValueError:
                    return  # malformed directory name
            else:
                return  # not a checkpoint directory

            checkpoints.append({
                "name":      name,
                "step":      step_val,
                "path":      path,
                "run":       run_name,
                "timestamp": timestamp,
            })

        for entry in os.scandir(self.checkpoint_root):
            if not entry.is_dir():
                continue
            name = entry.name

            if name.startswith("run_"):
                # New hierarchical layout: scan step dirs inside each run dir
                for step_entry in os.scandir(entry.path):
                    if step_entry.is_dir():
                        _load_step_dir(step_entry.path, step_entry.name, name)
            elif name.startswith("step_") or name == "final":
                # Legacy flat layout: step dirs directly under checkpoint_root
                _load_step_dir(entry.path, name, run_name=None)

        # Sort: numbered steps ascending, then "final" last
        def sort_key(c):
            s = c["step"]
            return (1, 0) if s == "final" else (0, s)

        checkpoints.sort(key=sort_key)
        return checkpoints

    # -----------------------------------------------------------------------

    def get_checkpoint_path(self, step) -> str:
        """Return the path for a step within the CURRENT run's directory.
        
        Use this during training. For evaluation (finding a checkpoint from
        any run), use resolve_checkpoint_path() instead.
        """
        return os.path.join(self._run_path, _step_dir_name(step))

    def resolve_checkpoint_path(self, step) -> Optional[str]:
        """Resolve a checkpoint path by scanning all runs for a given step.

        Returns the path from the most-recent run that contains the step,
        or None if no match is found. Used by evaluate.py to locate
        checkpoints from previous training runs.
        """
        target = _step_dir_name(step)
        # list_checkpoints() is already sorted ascending by step then by run;
        # we want the latest run, so collect all matches and return the last.
        matches = [
            c["path"]
            for c in self.list_checkpoints()
            if c["step"] == ("final" if step == "final" else int(step))
        ]
        return matches[-1] if matches else None

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
