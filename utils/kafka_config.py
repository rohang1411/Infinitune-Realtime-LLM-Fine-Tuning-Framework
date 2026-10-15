"""
utils/kafka_config.py
-----------------------------------------------------------------------------
Kafka client configuration builder, secret isolation, and client parameter passthrough.

Ensures credentials never leak into YAML configuration, provides environment-based
credential injection (INFINITUNE_KAFKA_USERNAME, INFINITUNE_KAFKA_PASSWORD), and
validates producer/consumer payload sizing invariants across nodes.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

VALID_ROLES = {"producer", "consumer", "admin"}

SHARED_ALLOWLIST = {
    "security_protocol",
    "sasl_mechanism",
    "ssl_cafile",
    "ssl_check_hostname",
    "request_timeout_ms",
    "api_version_auto_timeout_ms",
}

FORBIDDEN_SECRET_KEYS = {
    "username",
    "password",
    "sasl_plain_username",
    "sasl_plain_password",
    "sasl_username",
    "sasl_password",
}


def redact_client_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of kwargs with sensitive credentials redacted."""
    redacted = dict(kwargs)
    for key in ("sasl_plain_password", "password"):
        if key in redacted and redacted[key] is not None:
            redacted[key] = "********"
    return redacted


def build_client_kwargs(kafka_cfg: Dict[str, Any] | None, role: str) -> Dict[str, Any]:
    """
    Build kwargs dictionary for kafka-python clients (KafkaProducer, KafkaConsumer, KafkaAdminClient).

    Parameters
    ----------
    kafka_cfg : dict
        Kafka configuration block from YAML (or containing 'client' sub-dictionary).
    role : str
        One of {'producer', 'consumer', 'admin'}.

    Returns
    -------
    dict
        Validated keyword arguments for kafka-python client constructors.
    """
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role {role!r}. Must be one of {sorted(VALID_ROLES)}")

    if kafka_cfg is None:
        kafka_cfg = {}

    client_cfg = kafka_cfg.get("client")
    if not isinstance(client_cfg, dict):
        client_cfg = {}

    # Reject credentials/passwords embedded in YAML
    for key in FORBIDDEN_SECRET_KEYS:
        if key in kafka_cfg or key in client_cfg:
            raise ValueError(
                f"Kafka secret/credential key '{key}' is forbidden in YAML configuration. "
                "Provide credentials via INFINITUNE_KAFKA_USERNAME and INFINITUNE_KAFKA_PASSWORD environment variables."
            )

    kwargs: Dict[str, Any] = {}

    # Bootstrap servers: environment override wins
    env_bootstrap = os.environ.get("INFINITUNE_KAFKA_BOOTSTRAP")
    bootstrap = env_bootstrap if env_bootstrap else kafka_cfg.get("bootstrap_servers", "localhost:9092")
    kwargs["bootstrap_servers"] = bootstrap

    # Allowlist inspection for client sub-config
    known_keys = SHARED_ALLOWLIST | {"max_request_size", "compression_type", "max_partition_fetch_bytes"}
    for k, v in client_cfg.items():
        if k in SHARED_ALLOWLIST:
            kwargs[k] = v
        elif k not in known_keys:
            logger.warning("Unknown kafka.client key ignored: %s", k)

    # Check top-level kafka_cfg for shared allowlist keys if not set by client_cfg
    for k in SHARED_ALLOWLIST:
        if k not in kwargs and k in kafka_cfg:
            kwargs[k] = kafka_cfg[k]

    # SASL credentials from environment
    sasl_mechanism = kwargs.get("sasl_mechanism") or client_cfg.get("sasl_mechanism") or kafka_cfg.get("sasl_mechanism")
    if sasl_mechanism:
        kwargs["sasl_mechanism"] = sasl_mechanism
        username = os.environ.get("INFINITUNE_KAFKA_USERNAME")
        password = os.environ.get("INFINITUNE_KAFKA_PASSWORD")
        if not username or not password:
            raise ValueError(
                "SASL mechanism configured but INFINITUNE_KAFKA_USERNAME or "
                "INFINITUNE_KAFKA_PASSWORD environment variable is missing"
            )
        kwargs["sasl_plain_username"] = username
        kwargs["sasl_plain_password"] = password

    # Sizing and compression
    raw_mrs = client_cfg.get("max_request_size") or kafka_cfg.get("max_request_size")
    mrs = int(raw_mrs) if raw_mrs is not None else 1048576

    if role == "producer":
        kwargs["max_request_size"] = mrs
        raw_comp = client_cfg.get("compression_type") if "compression_type" in client_cfg else kafka_cfg.get("compression_type")
        if raw_comp is not None:
            if raw_comp != "gzip":
                raise ValueError(
                    f"Unsupported compression_type '{raw_comp}'. kafka-python supports only None or 'gzip' without extra C libraries."
                )
            kwargs["compression_type"] = raw_comp
        else:
            kwargs["compression_type"] = None

    elif role == "consumer":
        raw_mpfb = client_cfg.get("max_partition_fetch_bytes") if "max_partition_fetch_bytes" in client_cfg else kafka_cfg.get("max_partition_fetch_bytes")
        if raw_mpfb is not None:
            mpfb = int(raw_mpfb)
            if mpfb < mrs:
                raise ValueError(
                    f"max_partition_fetch_bytes ({mpfb}) must be at least max_request_size ({mrs})"
                )
            kwargs["max_partition_fetch_bytes"] = mpfb
        else:
            kwargs["max_partition_fetch_bytes"] = max(1048576, mrs)

    return kwargs
