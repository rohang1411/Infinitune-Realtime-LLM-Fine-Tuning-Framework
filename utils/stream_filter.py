import re
import time
import zlib
from typing import Optional


STAGE_LENGTH = "length"
STAGE_CONTENT = "content"
STAGE_REPETITION = "repetition"
STAGE_CHAT = "chat"
STAGE_REGEX = "regex"
STAGE_ERROR = "error"


class FilterStats:
    """Tracks filtering statistics, execution timing, and examples per stage."""

    def __init__(self):
        self.seen: int = 0
        self.passed: int = 0
        self.dropped_by_stage: dict[str, int] = {
            STAGE_LENGTH: 0,
            STAGE_CONTENT: 0,
            STAGE_REPETITION: 0,
            STAGE_CHAT: 0,
            STAGE_REGEX: 0,
            STAGE_ERROR: 0,
        }
        self.elapsed_s: float = 0.0
        self.examples_by_stage: dict[str, list[str]] = {
            STAGE_LENGTH: [],
            STAGE_CONTENT: [],
            STAGE_REPETITION: [],
            STAGE_CHAT: [],
            STAGE_REGEX: [],
            STAGE_ERROR: [],
        }

    def record(self, is_valid: bool, reason: Optional[str], text: str, elapsed: float):
        self.seen += 1
        self.elapsed_s += elapsed
        if is_valid:
            self.passed += 1
        else:
            stage = reason.split(":", 1)[0] if reason and ":" in reason else STAGE_ERROR
            self.dropped_by_stage[stage] = self.dropped_by_stage.get(stage, 0) + 1
            examples = self.examples_by_stage.setdefault(stage, [])
            if len(examples) < 5:
                truncated = (text or "")[:200]
                examples.append(truncated)

    def to_dict(self) -> dict:
        return {
            "seen": self.seen,
            "passed": self.passed,
            "dropped_by_stage": dict(self.dropped_by_stage),
            "elapsed_s": round(self.elapsed_s, 6),
            "examples_by_stage": {k: list(v) for k, v in self.examples_by_stage.items()},
        }


class StreamFilter:
    """
    High-performance, multi-stage filtering for Producer streams.

    Complexity ordering (fast short-circuiting):
    - O(1): len() bounds (STAGE_LENGTH)
    - O(N): character scans (isalnum / isdigit) (STAGE_CONTENT)
    - O(N): zlib compression ratio (STAGE_REPETITION)
    - O(turns): chat structure checks (STAGE_CHAT)
    - O(N * R): regex checks last (R = number of patterns) (STAGE_REGEX)
    """

    def __init__(self, config, log_fn=None):
        self._log = log_fn or (lambda msg: print(msg, flush=True))

        data_cfg = (config.get("data") or {})
        filtering_cfg = (data_cfg.get("filtering") or {})

        self.on_error = filtering_cfg.get("on_error", "drop")
        if self.on_error not in ("drop", "pass"):
            self.on_error = "drop"

        # Universal filters (simple text heuristics)
        universal = filtering_cfg.get("universal") or {}
        self.min_chars = universal.get("min_chars")
        self.max_chars = universal.get("max_chars")
        self.min_alphanumeric_ratio = universal.get("min_alphanumeric_ratio")
        self.max_repetition_ratio = universal.get("max_repetition_ratio")

        # Domain-specific filters
        domain = filtering_cfg.get("domain_specific") or {}
        self.require_numeric_content = domain.get("require_numeric_content")

        chat_cfg = domain.get("chat_structure") or {}
        self.chat_min_turns = chat_cfg.get("min_turns")
        self.chat_require_assistant_final = chat_cfg.get("require_assistant_final")

        # Pre-compile regex patterns for speed; invalid patterns are skipped safely.
        self.custom_regex_must_match = []
        self.custom_regex_must_not_match = []

        must_match_patterns = domain.get("custom_regex_must_match") or []
        must_not_match_patterns = domain.get("custom_regex_must_not_match") or []

        for p in must_match_patterns:
            try:
                self.custom_regex_must_match.append(re.compile(p))
            except re.error as e:
                self._log(f"[{self._ts()}][FILTER] WARNING: bad must_match regex '{p}': {e}")

        for p in must_not_match_patterns:
            try:
                self.custom_regex_must_not_match.append(re.compile(p))
            except re.error as e:
                self._log(f"[{self._ts()}][FILTER] WARNING: bad must_not_match regex '{p}': {e}")

        self.enabled = any(
            [
                self.min_chars is not None,
                self.max_chars is not None,
                self.min_alphanumeric_ratio is not None,
                self.max_repetition_ratio is not None,
                self.require_numeric_content is not None,
                self.chat_min_turns is not None,
                self.chat_require_assistant_final is not None,
                bool(self.custom_regex_must_match),
                bool(self.custom_regex_must_not_match),
            ]
        )
        self.stats = FilterStats()

    @staticmethod
    def _ts():
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def active_stages(self) -> list[str]:
        """Return the list of stage names that have any rule configured."""
        stages = []
        if self.min_chars is not None or self.max_chars is not None:
            stages.append(STAGE_LENGTH)
        if self.min_alphanumeric_ratio is not None or self.require_numeric_content is not None:
            stages.append(STAGE_CONTENT)
        if self.max_repetition_ratio is not None:
            stages.append(STAGE_REPETITION)
        if self.chat_min_turns is not None or self.chat_require_assistant_final is not None:
            stages.append(STAGE_CHAT)
        if bool(self.custom_regex_must_match) or bool(self.custom_regex_must_not_match):
            stages.append(STAGE_REGEX)
        return stages

    def _build_text_for_metrics(self, raw_record, extracted_text):
        """
        If raw_record looks like chat messages (list[dict]), concatenate message
        contents so JSON/list syntax doesn't skew alphanumeric/compression metrics.
        """
        if isinstance(raw_record, (list, tuple)) and raw_record:
            try:
                if all(isinstance(m, dict) for m in raw_record):
                    return " ".join(str(m.get("content", "")) for m in raw_record)
            except Exception:
                pass
        return "" if extracted_text is None else str(extracted_text)

    def _passes_chat_structure(self, raw_record):
        if not (self.chat_min_turns is not None or self.chat_require_assistant_final is not None):
            return True

        if not isinstance(raw_record, (list, tuple)):
            return True

        try:
            turns = len(raw_record)
            if self.chat_min_turns is not None and turns < self.chat_min_turns:
                return False

            if self.chat_require_assistant_final:
                if turns == 0:
                    return False
                last = raw_record[-1]
                if isinstance(last, dict) and last.get("role") != "assistant":
                    return False
        except Exception:
            # Safety: don't crash the producer due to unexpected structure.
            return True

        return True

    def _check_repetition(self, text: str) -> Optional[str]:
        if self.max_repetition_ratio is None:
            return None
        raw_bytes = text.encode("utf-8", errors="ignore")
        if len(raw_bytes) > 0:
            try:
                comp = zlib.compress(raw_bytes)
                ratio = len(comp) / float(len(raw_bytes))  # safe: len(raw_bytes)>0
                if ratio <= self.max_repetition_ratio:
                    return f"{STAGE_REPETITION}:max_repetition_ratio({ratio:.3f}<={self.max_repetition_ratio})"
            except Exception as e:
                self._log(f"[{self._ts()}][FILTER] WARNING: zlib failed: {e}")
        return None

    def is_valid(self, raw_record, extracted_text):
        ok, _reason = self.validate(raw_record, extracted_text)
        return ok

    def validate(self, raw_record, extracted_text):
        """
        Returns (is_valid: bool, reason: str|None).
        Reason is intended for debugging/telemetry and starts with '<stage>:'.
        """
        t0 = time.perf_counter()
        text = ""
        if not self.enabled:
            if self.stats is not None:
                self.stats.record(True, None, "", time.perf_counter() - t0)
            return True, None

        try:
            text = self._build_text_for_metrics(raw_record, extracted_text)
            n = len(text)

            # -------------------------
            # O(1) checks first: STAGE_LENGTH
            # -------------------------
            if self.min_chars is not None and n < self.min_chars:
                ok, reason = False, f"{STAGE_LENGTH}:min_chars({n}<{self.min_chars})"
            elif self.max_chars is not None and n > self.max_chars:
                ok, reason = False, f"{STAGE_LENGTH}:max_chars({n}>{self.max_chars})"

            # -------------------------
            # O(N) scans second: STAGE_CONTENT
            # -------------------------
            elif self.require_numeric_content is True and not any(ch.isdigit() for ch in text):
                ok, reason = False, f"{STAGE_CONTENT}:require_numeric_content(no_digit_found)"
            elif self.min_alphanumeric_ratio is not None and n == 0:
                ok, reason = False, f"{STAGE_CONTENT}:min_alphanumeric_ratio(empty_text)"
            elif self.min_alphanumeric_ratio is not None and (sum(1 for ch in text if ch.isalnum()) / float(n)) < self.min_alphanumeric_ratio:
                ratio = sum(1 for ch in text if ch.isalnum()) / float(n)
                ok, reason = False, f"{STAGE_CONTENT}:min_alphanumeric_ratio({ratio:.3f}<{self.min_alphanumeric_ratio})"

            # -------------------------
            # repetition (zlib) third: STAGE_REPETITION
            # -------------------------
            elif self.max_repetition_ratio is not None and self._check_repetition(text) is not None:
                ok, reason = False, self._check_repetition(text)

            # -------------------------
            # chat-structure fourth: STAGE_CHAT
            # -------------------------
            elif not self._passes_chat_structure(raw_record):
                ok, reason = False, f"{STAGE_CHAT}:chat_structure(failed)"

            # -------------------------
            # regex last: STAGE_REGEX
            # -------------------------
            else:
                regex_failed = False
                for pat in self.custom_regex_must_match:
                    try:
                        if not pat.search(text):
                            ok, reason = False, f"{STAGE_REGEX}:custom_regex_must_match(missed:{pat.pattern})"
                            regex_failed = True
                            break
                    except Exception as e:
                        self._log(f"[{self._ts()}][FILTER] WARNING: must_match exec failed: {e}")

                if not regex_failed:
                    for pat in self.custom_regex_must_not_match:
                        try:
                            if pat.search(text):
                                ok, reason = False, f"{STAGE_REGEX}:custom_regex_must_not_match(hit:{pat.pattern})"
                                regex_failed = True
                                break
                        except Exception as e:
                            self._log(f"[{self._ts()}][FILTER] WARNING: must_not_match exec failed: {e}")

                if not regex_failed:
                    ok, reason = True, None

        except Exception as e:
            self._log(f"[{self._ts()}][FILTER] ERROR: validate crashed: {e}")
            if self.on_error == "pass":
                ok, reason = True, None
            else:
                ok, reason = False, f"{STAGE_ERROR}:{type(e).__name__}: {e}"

        elapsed = time.perf_counter() - t0
        if self.stats is not None:
            self.stats.record(ok, reason, text, elapsed)
        return ok, reason
