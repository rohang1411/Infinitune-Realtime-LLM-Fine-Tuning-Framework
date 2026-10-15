"""
utils/continual_engine.py
─────────────────────────────────────────────────────────────────────────────
Continual Learning & Safety Governance Engine for InfiniTune.

Key Components:
1. GoldenCanaryEvaluator:
   Evaluates candidate adapter checkpoints against a fixed anchor dataset
   before promoting to live serving. Rejects checkpoints that regress past a
   configurable threshold (e.g. +10% canary loss).
2. ADWINDriftDetector:
   Adaptive windowing statistical test on rolling validation loss to detect
   concept drift and trigger targeted adaptation.
"""

import math
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn


class GoldenCanaryEvaluator:
    """
    Evaluates candidate adapters against a held-out golden canary dataset
    to gate promotion and guard against performance regressions.
    """

    def __init__(
        self,
        canary_data: List[Dict[str, Any]],
        max_regression_ratio: float = 0.10,
        device: str = "cpu"
    ):
        self.canary_data = canary_data
        self.max_regression_ratio = max_regression_ratio
        self.device = device
        self.baseline_loss: Optional[float] = None

    def set_baseline(self, baseline_loss: float) -> None:
        """Record the initial baseline canary loss."""
        self.baseline_loss = float(baseline_loss)

    def evaluate_loss(
        self,
        model: nn.Module,
        batch_collate_fn
    ) -> float:
        """Compute mean evaluation loss over the golden canary dataset."""
        if not self.canary_data:
            return 0.0

        model.eval()
        total_loss = 0.0
        count = 0

        # Evaluate in small batches of 8 to avoid OOM
        batch_size = 8
        with torch.no_grad():
            for i in range(0, len(self.canary_data), batch_size):
                chunk = self.canary_data[i : i + batch_size]
                batch = batch_collate_fn(chunk)
                outputs = model(**batch)
                loss = outputs.loss.item()
                total_loss += loss * len(chunk)
                count += len(chunk)

        model.train()
        return total_loss / max(count, 1)

    def evaluate_gate(
        self,
        candidate_loss: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Gate check: Approve promotion if candidate loss does not exceed
        baseline_loss * (1.0 + max_regression_ratio).
        """
        if self.baseline_loss is None or self.baseline_loss <= 0.0:
            # First evaluation establishes the baseline
            self.baseline_loss = candidate_loss
            return True, {
                "approved": True,
                "candidate_loss": candidate_loss,
                "baseline_loss": candidate_loss,
                "regression_ratio": 0.0,
                "reason": "Initial baseline established",
            }

        allowed_loss = self.baseline_loss * (1.0 + self.max_regression_ratio)
        regression_ratio = (candidate_loss - self.baseline_loss) / self.baseline_loss
        approved = candidate_loss <= allowed_loss

        return approved, {
            "approved": approved,
            "candidate_loss": candidate_loss,
            "baseline_loss": self.baseline_loss,
            "regression_ratio": round(regression_ratio, 4),
            "reason": "Approved within threshold" if approved else f"Regressed by {regression_ratio * 100:.1f}%",
        }


class ADWINDriftDetector:
    """
    Adaptive Windowing (ADWIN) Drift Detector.
    Maintains a variable-sized window of scalar observations. When the difference
    between two sub-windows exceeds the Hoeffding bound threshold, drift is flagged.
    """

    def __init__(self, delta: float = 0.002, min_window_size: int = 10, max_window_size: int = 200):
        self.delta = delta
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.window: List[float] = []
        self.drift_detected = False

    def update(self, value: float) -> bool:
        """Add observation and check for drift. Returns True if drift detected."""
        self.window.append(float(value))
        if len(self.window) > self.max_window_size:
            self.window.pop(0)

        self.drift_detected = False
        n = len(self.window)
        if n < self.min_window_size * 2:
            return False

        # Split window into two halves and check if difference is statistically significant
        mid = n // 2
        w0 = self.window[:mid]
        w1 = self.window[mid:]

        n0 = len(w0)
        n1 = len(w1)
        mean0 = sum(w0) / n0
        mean1 = sum(w1) / n1

        # Hoeffding epsilon bound: sqrt((1/(2*m)) * ln(4/delta))
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        eps = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / self.delta))

        if abs(mean0 - mean1) > eps:
            self.drift_detected = True
            # Shrink window upon drift
            self.window = self.window[mid:]
            return True

        return False

    def reset(self) -> None:
        """Reset the detector window."""
        self.window.clear()
        self.drift_detected = False
