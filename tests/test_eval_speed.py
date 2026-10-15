"""
Unit tests for Qualitative Evaluation Speed Benchmark (WP-17 / SDE-4).

Tests generation_mode routing, sequential generation loop, greedy equivalence,
and speed metrics calculation in-memory without network or broker dependencies.
"""

import pytest
import torch
from unittest.mock import patch

from utils.eval_qualitative import QualitativeEvaluator
from benchmarks.eval_speed import (
    evaluate_greedy_equivalence,
    summarize_eval_speed_results,
)


class DummyBatchEncoding(dict):
    def to(self, device):
        return DummyBatchEncoding({k: v.to(device) if hasattr(v, "to") else v for k, v in self.items()})


class DummyTokenizer:
    def __init__(self):
        self.eos_token_id = 50256
        self.pad_token_id = 50256
        self.padding_side = "right"

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            text = [text]
        batch_size = len(text)
        return DummyBatchEncoding({
            "input_ids": torch.tensor([[10, 20, 30, 40]] * batch_size, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1]] * batch_size, dtype=torch.long),
        })

    def decode(self, token_ids, **kwargs):
        return "The Punter is a pub."

    def batch_decode(self, token_ids_batch, **kwargs):
        return ["The Punter is a pub." for _ in range(len(token_ids_batch))]


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.training = False

    def generate(self, input_ids=None, generation_config=None, **kwargs):
        batch_size = input_ids.shape[0]
        num_return_sequences = getattr(generation_config, "num_return_sequences", 1) or 1
        total_seqs = batch_size * num_return_sequences
        return torch.tensor([[10, 20, 30, 40, 50, 60, 70]] * total_seqs, dtype=torch.long)


@pytest.fixture
def eval_setup():
    tokenizer = DummyTokenizer()
    model = DummyModel()
    device = torch.device("cpu")
    cfg = {
        "model": {"name": "dummy", "max_seq_length": 64},
        "testing_strategy": {
            "enabled": True,
            "method": "structured_slot_coverage",
            "eval_interval": 1,
            "eval_samples": 2,
            "max_new_tokens": 10,
            "consistency_runs": 1,
            "generation_mode": "batched",
        },
        "preprocessing": {
            "prompt_template": "MR: {{ input }}\nDescription:",
        },
    }
    return cfg, tokenizer, model, device


def test_generation_mode_config_validation(eval_setup):
    cfg, tokenizer, model, device = eval_setup
    # Batched valid
    cfg["testing_strategy"]["generation_mode"] = "batched"
    ev_batched = QualitativeEvaluator(cfg, tokenizer, device)
    assert ev_batched._generation_mode == "batched"

    # Sequential valid
    cfg["testing_strategy"]["generation_mode"] = "sequential"
    ev_seq = QualitativeEvaluator(cfg, tokenizer, device)
    assert ev_seq._generation_mode == "sequential"

    # Invalid raises ValueError
    cfg["testing_strategy"]["generation_mode"] = "turbo_invalid"
    with pytest.raises(ValueError, match="Unknown testing_strategy.generation_mode"):
        QualitativeEvaluator(cfg, tokenizer, device)


def test_sequential_vs_batched_generation_shape_and_output(eval_setup):
    cfg, tokenizer, model, device = eval_setup
    evaluator = QualitativeEvaluator(cfg, tokenizer, device)
    window = [
        {"input": "name[The Punter], food[English]", "reference": "The Punter is an English pub."},
        {"input": "name[Loch Fyne], food[Seafood]", "reference": "Loch Fyne serves seafood."},
    ]

    # Test single sequence (num_return_sequences=1)
    seq_preds, seq_refs = evaluator._generate_responses_sequential(
        model, window, do_sample=False, num_return_sequences=1
    )
    batch_preds, batch_refs = evaluator._generate_responses(
        model, window, do_sample=False, num_return_sequences=1
    )

    assert len(seq_preds) == 2
    assert len(batch_preds) == 2
    assert seq_preds == batch_preds
    assert seq_refs == batch_refs

    # Test multiple sequences (num_return_sequences=3)
    seq_preds_3, seq_refs_3 = evaluator._generate_responses_sequential(
        model, window, do_sample=True, num_return_sequences=3
    )
    batch_preds_3, batch_refs_3 = evaluator._generate_responses(
        model, window, do_sample=True, num_return_sequences=3
    )

    assert len(seq_preds_3) == 6
    assert len(batch_preds_3) == 6
    assert len(seq_refs_3) == 2
    assert len(batch_refs_3) == 2
    assert seq_refs_3 == batch_refs_3


def test_evaluate_greedy_equivalence(eval_setup):
    cfg, tokenizer, model, device = eval_setup
    evaluator = QualitativeEvaluator(cfg, tokenizer, device)
    window = [
        {"input": "name[The Punter]", "reference": "The Punter."},
        {"input": "name[Loch Fyne]", "reference": "Loch Fyne."},
    ]
    frac = evaluate_greedy_equivalence(evaluator, model, window, n_check=2)
    assert frac == 1.0


def test_run_routing_by_generation_mode(eval_setup):
    cfg, tokenizer, model, device = eval_setup
    window = [{"input": "name[The Punter]", "reference": "The Punter."}]

    # Case 1: batched mode routes to _generate_responses
    cfg["testing_strategy"]["generation_mode"] = "batched"
    ev_batched = QualitativeEvaluator(cfg, tokenizer, device)
    ev_batched.enabled = True
    ev_batched.eval_data = window
    with patch.object(ev_batched, "_generate_responses", wraps=ev_batched._generate_responses) as mock_b, \
         patch.object(ev_batched, "_generate_responses_sequential", wraps=ev_batched._generate_responses_sequential) as mock_s:
        ev_batched.run(model, step=1)
        assert mock_b.called
        assert not mock_s.called

    # Case 2: sequential mode routes to _generate_responses_sequential
    cfg["testing_strategy"]["generation_mode"] = "sequential"
    ev_seq = QualitativeEvaluator(cfg, tokenizer, device)
    ev_seq.enabled = True
    ev_seq.eval_data = window
    with patch.object(ev_seq, "_generate_responses", wraps=ev_seq._generate_responses) as mock_b, \
         patch.object(ev_seq, "_generate_responses_sequential", wraps=ev_seq._generate_responses_sequential) as mock_s:
        ev_seq.run(model, step=1)
        assert mock_s.called
        assert not mock_b.called


def test_summarize_eval_speed_results():
    trials = [
        {
            "repeat": 1,
            "sequential_s": 10.0,
            "batched_s": 1.0,
            "reduction_pct": 90.0,
            "speedup_ratio": 10.0,
            "total_generations": 100,
        },
        {
            "repeat": 2,
            "sequential_s": 12.0,
            "batched_s": 1.2,
            "reduction_pct": 90.0,
            "speedup_ratio": 10.0,
            "total_generations": 100,
        },
    ]
    summary = summarize_eval_speed_results(trials, identical_fraction=1.0, n_mrs=10, consistency_runs=10)
    assert summary["suite"] == "eval_speed"
    assert summary["n_mrs"] == 10
    assert summary["consistency_runs"] == 10
    assert summary["total_generations_per_run"] == 100
    assert summary["sequential_s"]["mean"] == 11.0
    assert summary["batched_s"]["mean"] == 1.1
    assert summary["reduction_pct_mean"] == 90.0
    assert summary["speedup_ratio_mean"] == 10.0
    assert summary["greedy_identical_fraction"] == 1.0
    assert "boundary" in summary
