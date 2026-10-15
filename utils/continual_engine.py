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
    Adaptive Windowing (ADWIN) Drift Detector (Bifet & Gavaldà, 2007).
    Maintains a variable-sized window of scalar observations. When the difference
    between two sub-windows exceeds the variance-aware Hoeffding bound threshold,
    concept drift is detected, older observations are dropped, and drift is flagged.
    """

    def __init__(
        self,
        delta: float = 0.002,
        min_window_size: int = 10,
        max_window_size: int = 200,
        clock: int = 1,
        value_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.delta = float(delta)
        self.min_window_size = int(min_window_size)
        self.max_window_size = int(max_window_size)
        self.clock = int(clock)
        self.value_range = (float(value_range[0]), float(value_range[1]))
        self.window: List[float] = []
        self.drift_detected = False
        self._updates_count = 0

    @property
    def width(self) -> int:
        """Current length of the observation window."""
        return len(self.window)

    @property
    def total(self) -> float:
        """Sum of all observations currently in the window."""
        return sum(self.window)

    @property
    def variance(self) -> float:
        """Empirical variance of observations in the window."""
        n = len(self.window)
        if n < 2:
            return 0.0
        mean = sum(self.window) / float(n)
        return max(0.0, sum((x - mean) ** 2 for x in self.window) / float(n))

    def update(self, value: float) -> bool:
        """Add observation, normalize to [0, 1], and check for drift across all cut points.

        Returns True if drift detected, False otherwise.
        """
        lo, hi = self.value_range
        val = float(value)
        if hi > lo:
            norm_val = max(0.0, min(1.0, (val - lo) / (hi - lo)))
        else:
            norm_val = max(0.0, min(1.0, val))

        self.window.append(norm_val)
        if len(self.window) > self.max_window_size:
            self.window.pop(0)

        self._updates_count += 1
        self.drift_detected = False

        if self._updates_count % self.clock != 0:
            return False

        n = len(self.window)
        if n < self.min_window_size * 2:
            return False

        # Build prefix sums for O(n) scan across all valid split points
        prefix_sum = [0.0] * (n + 1)
        prefix_sq_sum = [0.0] * (n + 1)
        for idx, x in enumerate(self.window):
            prefix_sum[idx + 1] = prefix_sum[idx] + x
            prefix_sq_sum[idx + 1] = prefix_sq_sum[idx] + (x * x)

        tot_sum = prefix_sum[n]
        tot_sq = prefix_sq_sum[n]
        mean_w = tot_sum / float(n)
        var_w = max(0.0, (tot_sq / float(n)) - (mean_w * mean_w))
        delta_prime = max(self.delta / float(n), 1e-12)
        ln_term = math.log(2.0 / delta_prime)

        # Check every split point i from min_window_size to n - min_window_size
        for i in range(self.min_window_size, n - self.min_window_size + 1):
            n0 = i
            n1 = n - i
            m = (n0 * n1) / float(n)  # harmonic mean factor: 1 / (1/n0 + 1/n1)
            eps = math.sqrt((2.0 / m) * var_w * ln_term) + (2.0 / (3.0 * m)) * ln_term

            mean0 = prefix_sum[i] / float(n0)
            mean1 = (tot_sum - prefix_sum[i]) / float(n1)

            if abs(mean0 - mean1) >= eps:
                self.drift_detected = True
                # Drop older subwindow W0, preserve newer W1
                self.window = self.window[i:]
                return True

        return False

    def reset(self) -> None:
        """Reset the detector window and internal state."""
        self.window.clear()
        self.drift_detected = False
        self._updates_count = 0
