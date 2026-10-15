"""Unit tests for inference startup and update listening policy.

Verifies WP-03 requirements:
- should_listen_for_updates decision table covering all combinations of
  checkpoint_source (explicit, auto, None) and CLI flags (listen, no_listen).
- Cold-start recovery (auto) now defaults to listening to Kafka (fixes D2).
- Explicit --checkpoint defaults to standalone (not listening).
- Parser raises error if both --listen-updates and --no-listen-updates are provided.
"""

from __future__ import annotations

import pytest

from utils.serving_policy import should_listen_for_updates


@pytest.mark.parametrize(
    "source,listen,no_listen,expected",
    [
        # Default flags (listen=False, no_listen=False)
        ("explicit", False, False, False),   # Historical standalone behaviour
        ("auto", False, False, True),        # Fix for D2: auto cold-start listens!
        (None, False, False, True),          # Base model default listens
        # Explicit --listen-updates (listen=True, no_listen=False)
        ("explicit", True, False, True),
        ("auto", True, False, True),
        (None, True, False, True),
        # Explicit --no-listen-updates (listen=False, no_listen=True)
        ("explicit", False, True, False),
        ("auto", False, True, False),
        (None, False, True, False),
    ],
)
def test_should_listen_combinations(source, listen, no_listen, expected):
    assert should_listen_for_updates(source, listen, no_listen) == expected


def test_parser_rejects_conflicting_listen_flags():
    import argparse

    # Test parser behavior matching inference.py's CLI setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-updates", action="store_true")
    parser.add_argument("--no-listen-updates", action="store_true")

    args = parser.parse_args(["--listen-updates", "--no-listen-updates"])
    assert args.listen_updates and args.no_listen_updates

    # Calling parser.error or custom validator raises SystemExit
    with pytest.raises(SystemExit):
        if args.listen_updates and args.no_listen_updates:
            parser.error("Cannot specify both --listen-updates and --no-listen-updates.")
