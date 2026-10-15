"""Serving policy logic for InfiniTune inference runtime."""

from __future__ import annotations

from typing import Optional


def should_listen_for_updates(
    checkpoint_source: Optional[str],
    listen_flag: bool,
    no_listen_flag: bool,
) -> bool:
    """Decide whether the inference server should start Kafka update threads.

    Parameters
    ----------
    checkpoint_source : {"explicit", "auto"} | None
        How the active checkpoint was selected (explicit CLI argument, auto cold-start, or None).
    listen_flag : bool
        Whether --listen-updates was explicitly supplied.
    no_listen_flag : bool
        Whether --no-listen-updates was explicitly supplied.

    Returns
    -------
    bool
        True if the server should subscribe to Kafka updates, False if standalone.
    """
    if no_listen_flag:
        return False
    if listen_flag:
        return True
    # Explicit --checkpoint keeps the historical standalone behaviour.
    return checkpoint_source != "explicit"
