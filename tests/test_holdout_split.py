import pytest
from utils.holdout import is_holdout_example


def test_holdout_split_determinism():
    example = {"instruction": "Summarize the following meeting.", "input": "Attendees: Alice, Bob."}
    cfg = {"method": "hash", "fraction": 0.05, "seed": 42}

    res1 = is_holdout_example(example, cfg)
    res2 = is_holdout_example(example, cfg)
    assert res1 == res2


def test_holdout_split_seed_sensitivity():
    # Over a set of diverse examples, changing the seed produces different partitions
    examples = [{"instruction": f"Task number {i}", "input": f"Context {i}"} for i in range(100)]
    cfg_seed42 = {"method": "hash", "fraction": 0.10, "seed": 42}
    cfg_seed99 = {"method": "hash", "fraction": 0.10, "seed": 99}

    partition_42 = {i for i, ex in enumerate(examples) if is_holdout_example(ex, cfg_seed42)}
    partition_99 = {i for i, ex in enumerate(examples) if is_holdout_example(ex, cfg_seed99)}

    assert partition_42 != partition_99


def test_holdout_split_disjointness_and_fraction():
    # On 1,000 synthetic examples with fraction 0.10, train and eval are disjoint
    # and holdout fraction is roughly 10% (within statistical variance: 75 to 125)
    examples = [{"instruction": f"Instruction prompt {i}", "input": f"Input data {i}"} for i in range(1000)]
    cfg = {"method": "hash", "fraction": 0.10, "seed": 42}

    holdout_indices = set()
    train_indices = set()

    for idx, ex in enumerate(examples):
        if is_holdout_example(ex, cfg):
            holdout_indices.add(idx)
        else:
            train_indices.add(idx)

    # Disjointness check
    assert len(holdout_indices & train_indices) == 0
    assert len(holdout_indices | train_indices) == 1000

    # Fraction check: expected ~100
    assert 75 <= len(holdout_indices) <= 125


def test_holdout_split_empty_or_non_hash():
    example = {"instruction": "Hello", "input": "World"}
    assert is_holdout_example(example, None) is False
    assert is_holdout_example(example, {}) is False
    assert is_holdout_example(example, {"method": "unknown_method"}) is False
