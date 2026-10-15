"""
utils/replay_buffer.py
─────────────────────────────────────────────────────────────────────────────
Reservoir Sampling Replay Buffer for Streaming Continual Learning in InfiniTune.

Key Continual Learning Capabilities:
1. Algorithm R (Vitter 1985) Reservoir Sampling:
   Guarantees that at any point in an unbounded stream of N samples, every
   historical sample has an identical probability (M / N) of residing in
   the buffer of capacity M.
2. Experience Rehearsal Batch Collation:
   Merges incoming streaming mini-batches with a configurable fraction
   gamma in [0.01, 0.10] of historical exemplars to prevent catastrophic
   forgetting on earlier domains.
"""

import random
from typing import List, Dict, Any, Optional


class ReservoirReplayBuffer:
    """
    Fixed-capacity memory buffer maintaining a uniform random sample of an
    unbounded stream using reservoir sampling.
    """

    def __init__(self, capacity: int = 500, seed: Optional[int] = 42):
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []
        self.total_seen = 0
        self.rng = random.Random(seed)

    def add(self, item: Dict[str, Any]) -> None:
        """Add a single streaming record to the reservoir."""
        self.total_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            # Randomly replace an existing sample with probability capacity / total_seen
            j = self.rng.randint(0, self.total_seen - 1)
            if j < self.capacity:
                self.buffer[j] = item

    def add_batch(self, items: List[Dict[str, Any]]) -> None:
        """Add a batch of streaming records."""
        for item in items:
            self.add(item)

    def sample(self, k: int) -> List[Dict[str, Any]]:
        """Sample k uniform random exemplars without replacement."""
        if not self.buffer:
            return []
        sample_size = min(k, len(self.buffer))
        return self.rng.sample(self.buffer, sample_size)

    def collate_with_stream(
        self,
        stream_batch: List[Dict[str, Any]],
        replay_ratio: float = 0.10
    ) -> List[Dict[str, Any]]:
        """
        Mix a streaming batch with replay samples based on replay_ratio.
        E.g. for batch_size=10 and replay_ratio=0.2, replaces 2 samples with replay exemplars.
        """
        if not self.buffer or replay_ratio <= 0.0:
            return list(stream_batch)

        batch_size = len(stream_batch)
        num_replay = max(1, int(round(batch_size * replay_ratio)))
        num_replay = min(num_replay, len(self.buffer), batch_size - 1)

        if num_replay <= 0:
            return list(stream_batch)

        replay_samples = self.sample(num_replay)
        # Keep (batch_size - num_replay) stream samples and append replay samples
        combined = list(stream_batch[: batch_size - num_replay]) + replay_samples
        self.rng.shuffle(combined)
        return combined

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """Reset the buffer."""
        self.buffer.clear()
        self.total_seen = 0
