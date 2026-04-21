import re
import time
import zlib


class StreamFilter:
    """
    High-performance, two-tier filtering for Producer streams.

    Complexity ordering (fast short-circuiting):
    - O(1): len() bounds, basic flags
    - O(N): character scans (isalnum / isdigit)
    - O(N): zlib compression ratio (bytes processed once)
    - O(N * R): regex checks last (R = number of patterns)
    """

    def __init__(self, config, log_fn=None):
        self._log = log_fn or (lambda msg: print(msg, flush=True))

        data_cfg = (config.get("data") or {})
        filtering_cfg = (data_cfg.get("filtering") or {})

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

    @staticmethod
    def _ts():
        return time.strftime("%Y-%m-%d %H:%M:%S")

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

    def is_valid(self, raw_record, extracted_text):
        ok, _reason = self.validate(raw_record, extracted_text)
        return ok

    def validate(self, raw_record, extracted_text):
        """
        Returns (is_valid: bool, reason: str|None).
        Reason is intended for debugging/telemetry and is kept lightweight.
        """
        if not self.enabled:
            return True, None

        try:
            text = self._build_text_for_metrics(raw_record, extracted_text)
            n = len(text)

            # -------------------------
            # O(1) checks first
            # -------------------------
            if self.min_chars is not None and n < self.min_chars:
                return False, f"min_chars({n}<{self.min_chars})"
            if self.max_chars is not None and n > self.max_chars:
                return False, f"max_chars({n}>{self.max_chars})"

            # -------------------------
            # O(N) scans second
            # -------------------------
            if self.require_numeric_content is True:
                if not any(ch.isdigit() for ch in text):
                    return False, "require_numeric_content(no_digit_found)"

            if self.min_alphanumeric_ratio is not None:
                if n == 0:
                    return False, "min_alphanumeric_ratio(empty_text)"
                alnum = sum(1 for ch in text if ch.isalnum())
                ratio = alnum / float(n)
                if ratio < self.min_alphanumeric_ratio:
                    return False, f"min_alphanumeric_ratio({ratio:.3f}<{self.min_alphanumeric_ratio})"

            # -------------------------
            # repetition (zlib) third
            # -------------------------
            if self.max_repetition_ratio is not None:
                raw_bytes = text.encode("utf-8", errors="ignore")
                if len(raw_bytes) > 0:
                    try:
                        comp = zlib.compress(raw_bytes)
                        ratio = len(comp) / float(len(raw_bytes))  # safe: len(raw_bytes)>0
                        if ratio <= self.max_repetition_ratio:
                            return False, f"max_repetition_ratio({ratio:.3f}<={self.max_repetition_ratio})"
                    except Exception as e:
                        self._log(f"[{self._ts()}][FILTER] WARNING: zlib failed: {e}")

            # -------------------------
            # chat-structure fourth
            # -------------------------
            if not self._passes_chat_structure(raw_record):
                return False, "chat_structure(failed)"

            # -------------------------
            # regex last
            # -------------------------
            for pat in self.custom_regex_must_match:
                try:
                    if not pat.search(text):
                        return False, f"custom_regex_must_match(missed:{pat.pattern})"
                except Exception as e:
                    self._log(f"[{self._ts()}][FILTER] WARNING: must_match exec failed: {e}")

            for pat in self.custom_regex_must_not_match:
                try:
                    if pat.search(text):
                        return False, f"custom_regex_must_not_match(hit:{pat.pattern})"
                except Exception as e:
                    self._log(f"[{self._ts()}][FILTER] WARNING: must_not_match exec failed: {e}")

            return True, None
        except Exception as e:
            self._log(f"[{self._ts()}][FILTER] ERROR: validate crashed: {e}")
            return True, None

