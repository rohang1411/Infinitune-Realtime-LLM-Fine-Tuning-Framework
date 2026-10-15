import yaml
import pytest
from utils.stream_filter import (
    StreamFilter,
    FilterStats,
    STAGE_LENGTH,
    STAGE_CONTENT,
    STAGE_REPETITION,
    STAGE_CHAT,
    STAGE_REGEX,
    STAGE_ERROR,
)


def test_stage_prefix_drops():
    # 1. Length stage
    cfg_len = {"data": {"filtering": {"universal": {"min_chars": 15}}}}
    sf_len = StreamFilter(cfg_len)
    ok, reason = sf_len.validate({}, "too short")
    assert not ok
    assert reason.startswith(f"{STAGE_LENGTH}:")

    # 2. Content stage (alphanumeric ratio)
    cfg_content_alnum = {"data": {"filtering": {"universal": {"min_alphanumeric_ratio": 0.8}}}}
    sf_content_alnum = StreamFilter(cfg_content_alnum)
    ok, reason = sf_content_alnum.validate({}, "????....!!!!$$$$")
    assert not ok
    assert reason.startswith(f"{STAGE_CONTENT}:")

    # 2b. Content stage (numeric requirement)
    cfg_content_num = {"data": {"filtering": {"domain_specific": {"require_numeric_content": True}}}}
    sf_content_num = StreamFilter(cfg_content_num)
    ok, reason = sf_content_num.validate({}, "No digits in this sentence")
    assert not ok
    assert reason.startswith(f"{STAGE_CONTENT}:")

    # 3. Repetition stage (zlib)
    cfg_rep = {"data": {"filtering": {"universal": {"max_repetition_ratio": 0.2}}}}
    sf_rep = StreamFilter(cfg_rep)
    ok, reason = sf_rep.validate({}, "repetitive " * 500)
    assert not ok
    assert reason.startswith(f"{STAGE_REPETITION}:")

    # 4. Chat stage
    cfg_chat = {"data": {"filtering": {"domain_specific": {"chat_structure": {"min_turns": 3}}}}}
    sf_chat = StreamFilter(cfg_chat)
    ok, reason = sf_chat.validate([{"role": "user", "content": "Hi"}], "Hi")
    assert not ok
    assert reason.startswith(f"{STAGE_CHAT}:")

    # 5. Regex stage (must_match)
    cfg_reg_match = {"data": {"filtering": {"domain_specific": {"custom_regex_must_match": [r"^REQUIRED:"]}}}}
    sf_reg_match = StreamFilter(cfg_reg_match)
    ok, reason = sf_reg_match.validate({}, "does not start with required")
    assert not ok
    assert reason.startswith(f"{STAGE_REGEX}:")

    # 5b. Regex stage (must_not_match)
    cfg_reg_nomatch = {"data": {"filtering": {"domain_specific": {"custom_regex_must_not_match": [r"BANNED"]}}}}
    sf_reg_nomatch = StreamFilter(cfg_reg_nomatch)
    ok, reason = sf_reg_nomatch.validate({}, "contains BANNED word")
    assert not ok
    assert reason.startswith(f"{STAGE_REGEX}:")


def test_on_error_drop_default(monkeypatch):
    cfg = {"data": {"filtering": {"universal": {"min_chars": 5}}}}
    sf = StreamFilter(cfg)
    assert sf.on_error == "drop"

    def _boom(*args, **kwargs):
        raise RuntimeError("Simulated build error")

    monkeypatch.setattr(sf, "_build_text_for_metrics", _boom)
    ok, reason = sf.validate({}, "valid text")
    assert ok is False
    assert reason.startswith(f"{STAGE_ERROR}:")
    assert "RuntimeError" in reason


def test_on_error_pass_legacy(monkeypatch):
    cfg = {"data": {"filtering": {"on_error": "pass", "universal": {"min_chars": 5}}}}
    sf = StreamFilter(cfg)
    assert sf.on_error == "pass"

    def _boom(*args, **kwargs):
        raise RuntimeError("Simulated build error")

    monkeypatch.setattr(sf, "_build_text_for_metrics", _boom)
    ok, reason = sf.validate({}, "valid text")
    assert ok is True
    assert reason is None


def test_stats_counts_add_up():
    cfg = {
        "data": {
            "filtering": {
                "universal": {"min_chars": 10, "max_repetition_ratio": 0.2},
                "domain_specific": {"require_numeric_content": True},
            }
        }
    }
    sf = StreamFilter(cfg)
    # 1. Valid record (passes all)
    sf.validate({}, "This is a valid text with 123 numbers and sufficient entropy.")
    # 2. Dropped by length
    sf.validate({}, "short 1")
    # 3. Dropped by content
    sf.validate({}, "This text has plenty of characters but no digits at all.")
    # 4. Dropped by repetition
    sf.validate({}, "abc 123 " * 200)

    stats = sf.stats
    assert stats.seen == 4
    assert stats.passed == 1
    assert stats.seen == stats.passed + sum(stats.dropped_by_stage.values())
    assert stats.dropped_by_stage[STAGE_LENGTH] == 1
    assert stats.dropped_by_stage[STAGE_CONTENT] == 1
    assert stats.dropped_by_stage[STAGE_REPETITION] == 1
    assert stats.elapsed_s > 0

    d = stats.to_dict()
    assert d["seen"] == 4
    assert d["passed"] == 1
    assert len(d["examples_by_stage"][STAGE_LENGTH]) == 1
    assert len(d["examples_by_stage"][STAGE_CONTENT]) == 1
    assert len(d["examples_by_stage"][STAGE_REPETITION]) == 1


def test_active_stages_alpaca_config():
    with open("configs/alpaca_qualitative.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sf = StreamFilter(cfg)
    assert sf.active_stages() == ["length", "content", "repetition"]
