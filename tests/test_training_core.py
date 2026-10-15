import torch
import torch.nn as nn
from typing import List, Optional
import pytest

from trainer import (
    tokenize_with_label_masking,
    finish_optimizer_step,
    OptimizerStepResult,
)
from benchmarks.ablation import (
    detect_gradient_spikes,
    compute_prompt_regurgitation_rate,
)


class TinyMockTokenizer:
    """Hermetic mock tokenizer for fast unit testing with no external downloads."""
    def __init__(self, eos_token_id: int = 50256, bos_token_id: Optional[int] = 50256):
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = 50256
        self._vocab = {}
        self._next_id = 100

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = text.strip().split()
        res = []
        for t in tokens:
            if t not in self._vocab:
                self._vocab[t] = self._next_id
                self._next_id += 1
            res.append(self._vocab[t])
        return res


def test_prompt_tokens_are_masked_minus_100():
    tokenizer = TinyMockTokenizer()
    prompt = "Review: The movie was truly breathtaking. Sentiment:"
    response = " positive"

    # Default: label_masking=True
    tok_masked = tokenize_with_label_masking(
        tokenizer, prompt, response, max_seq_length=64, label_masking=True
    )
    p_len = tok_masked["prompt_len"]
    assert p_len > 0
    # Every position in prompt is -100
    assert tok_masked["labels"][:p_len] == [-100] * p_len
    # Every response position matches input_ids
    assert tok_masked["labels"][p_len:] == tok_masked["input_ids"][p_len:]
    # Ends with eos_token_id
    assert tok_masked["input_ids"][-1] == tokenizer.eos_token_id

    # Ablation: label_masking=False
    tok_unmasked = tokenize_with_label_masking(
        tokenizer, prompt, response, max_seq_length=64, label_masking=False
    )
    assert tok_unmasked["labels"] == tok_unmasked["input_ids"]


def test_response_preserved_under_truncation():
    tokenizer = TinyMockTokenizer()
    # Create long prompt (15 tokens) and moderate response (5 tokens)
    prompt = " ".join([f"word{i}" for i in range(15)])
    response = "answer_a answer_b answer_c"

    # Set max_seq_length=10.
    # Response has 3 tokens + 1 EOS = 4 tokens.
    # max_prompt_len = 10 - 4 = 6 tokens.
    # With left_prompt truncation, the prompt keeps the LAST 6 tokens.
    tok_left = tokenize_with_label_masking(
        tokenizer, prompt, response, max_seq_length=10, truncation_side="left_prompt"
    )
    assert len(tok_left["input_ids"]) == 10
    assert tok_left["prompt_len"] == 6
    # Response is fully preserved at the end including EOS
    expected_resp_ids = tokenizer.encode(response) + [tokenizer.eos_token_id]
    assert tok_left["input_ids"][6:] == expected_resp_ids

    # Ablation: truncation_side="right"
    # Concatenates prompt + response, then truncates from right.
    tok_right = tokenize_with_label_masking(
        tokenizer, prompt, response, max_seq_length=10, truncation_side="right"
    )
    assert len(tok_right["input_ids"]) == 10
    # Prompt had 15 tokens, so the first 10 tokens are all prompt tokens; response was completely cut off!
    assert tok_right["prompt_len"] == 10
    expected_prompt_ids = tokenizer.encode(prompt)[:10]
    assert tok_right["input_ids"] == expected_prompt_ids


def test_all_masked_sample_is_rejected_or_skipped():
    tokenizer = TinyMockTokenizer(eos_token_id=None) # no EOS
    prompt = "Some instructions here"
    empty_response = ""

    # When response is completely empty and no EOS is appended, all labels become -100
    tok = tokenize_with_label_masking(
        tokenizer, prompt, empty_response, max_seq_length=32, label_masking=True
    )
    assert all(lbl == -100 for lbl in tok["labels"])

    # Corner case: response exceeds max_seq_length
    tokenizer_with_eos = TinyMockTokenizer(eos_token_id=50256)
    long_response = " ".join([f"resp{i}" for i in range(50)])
    tok_long_resp = tokenize_with_label_masking(
        tokenizer_with_eos, "Short prompt", long_response, max_seq_length=16, truncation_side="left_prompt"
    )
    # When response alone exceeds length, prompt is dropped to 0 tokens
    assert tok_long_resp["prompt_len"] == 0
    assert len(tok_long_resp["input_ids"]) == 16
    assert tok_long_resp["input_ids"][-1] == tokenizer_with_eos.eos_token_id


def test_gradient_clipping_caps_global_norm():
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10, bias=False)

    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Set huge gradients manually: norm should be sqrt(100 * 20^2) = 200.0
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 20.0

    # Execute finish_optimizer_step with max_grad_norm=1.0
    res = finish_optimizer_step(
        model, optimizer, scheduler=None, precision_mgr=None, max_grad_norm=1.0
    )

    # Verify OptimizerStepResult behaves as a 2-tuple (did_step, pre_clip_norm)
    did_step, pre_clip_norm = res
    assert did_step is True
    assert pre_clip_norm > 100.0  # Reported pre-clip norm is unclipped
    assert res.post_clip_norm <= 1.0 + 1e-6  # Clipped norm capped at max_norm

    # Now verify unclipped behavior when max_grad_norm is None
    model_unclipped = ToyModel()
    opt_unclipped = torch.optim.SGD(model_unclipped.parameters(), lr=0.01)
    for p in model_unclipped.parameters():
        p.grad = torch.ones_like(p) * 20.0

    res_unclipped = finish_optimizer_step(
        model_unclipped, opt_unclipped, scheduler=None, precision_mgr=None, max_grad_norm=None
    )
    assert res_unclipped.pre_clip_norm > 100.0
    assert abs(res_unclipped.pre_clip_norm - res_unclipped.post_clip_norm) < 1e-5


def test_detect_gradient_spikes():
    # 50 stationary norms around 1.0, then a spike of 15.0 at step 51
    norms = [1.0 + (i % 3) * 0.1 for i in range(50)] + [15.0] + [1.1 for _ in range(10)]
    count, indices = detect_gradient_spikes(norms, multiplier=5.0, window=50)
    assert count == 1
    assert indices == [50]


def test_compute_prompt_regurgitation_rate():
    # 20 identical consecutive tokens
    repeated_phrase = " ".join([f"token_{i}" for i in range(25)])
    prompts = [
        f"Please explain {repeated_phrase} clearly.",
        "A completely different prompt about cats and dogs.",
    ]
    # First generation copies the 25 tokens verbatim; second does not
    generations = [
        f"Here is the answer: {repeated_phrase} which means something.",
        "The cat sat on the mat and purred quietly.",
    ]

    rate = compute_prompt_regurgitation_rate(prompts, generations, min_tokens=20)
    assert rate == 0.5
