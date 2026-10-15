"""
tests/test_kafka_config.py
-----------------------------------------------------------------------------
Unit tests for Kafka client configuration passthrough and secret isolation (WP-12).
"""

import logging
import pytest

from utils.kafka_config import (
    build_client_kwargs,
    redact_client_kwargs,
    FORBIDDEN_SECRET_KEYS,
)


def test_plaintext_default():
    cfg = {"bootstrap_servers": "localhost:9092"}
    producer_kwargs = build_client_kwargs(cfg, "producer")
    assert producer_kwargs["bootstrap_servers"] == "localhost:9092"
    assert producer_kwargs["max_request_size"] == 1048576
    assert producer_kwargs["compression_type"] is None
    assert "sasl_plain_password" not in producer_kwargs

    consumer_kwargs = build_client_kwargs(cfg, "consumer")
    assert consumer_kwargs["bootstrap_servers"] == "localhost:9092"
    assert consumer_kwargs["max_partition_fetch_bytes"] >= 1048576

    admin_kwargs = build_client_kwargs(cfg, "admin")
    assert admin_kwargs["bootstrap_servers"] == "localhost:9092"
    assert "max_request_size" not in admin_kwargs


def test_env_bootstrap_override_wins(monkeypatch):
    monkeypatch.setenv("INFINITUNE_KAFKA_BOOTSTRAP", "cluster.internal:9094")
    cfg = {"bootstrap_servers": "localhost:9092"}
    kwargs = build_client_kwargs(cfg, "producer")
    assert kwargs["bootstrap_servers"] == "cluster.internal:9094"


def test_sasl_fails_when_env_vars_missing(monkeypatch):
    monkeypatch.delenv("INFINITUNE_KAFKA_USERNAME", raising=False)
    monkeypatch.delenv("INFINITUNE_KAFKA_PASSWORD", raising=False)
    cfg = {
        "bootstrap_servers": "localhost:9092",
        "client": {
            "security_protocol": "SASL_SSL",
            "sasl_mechanism": "PLAIN",
        },
    }
    with pytest.raises(ValueError, match="INFINITUNE_KAFKA_USERNAME or INFINITUNE_KAFKA_PASSWORD"):
        build_client_kwargs(cfg, "consumer")


def test_sasl_succeeds_with_env_vars(monkeypatch):
    monkeypatch.setenv("INFINITUNE_KAFKA_USERNAME", "alice")
    monkeypatch.setenv("INFINITUNE_KAFKA_PASSWORD", "supersecret123")
    cfg = {
        "bootstrap_servers": "remote.kafka:9092",
        "client": {
            "security_protocol": "SASL_SSL",
            "sasl_mechanism": "PLAIN",
            "ssl_check_hostname": True,
        },
    }
    kwargs = build_client_kwargs(cfg, "producer")
    assert kwargs["security_protocol"] == "SASL_SSL"
    assert kwargs["sasl_mechanism"] == "PLAIN"
    assert kwargs["sasl_plain_username"] == "alice"
    assert kwargs["sasl_plain_password"] == "supersecret123"
    assert kwargs["ssl_check_hostname"] is True

    # Redaction test
    redacted = redact_client_kwargs(kwargs)
    assert redacted["sasl_plain_password"] == "********"
    assert redacted["sasl_plain_username"] == "alice"


@pytest.mark.parametrize("secret_key", sorted(FORBIDDEN_SECRET_KEYS))
def test_sasl_secrets_in_yaml_rejected(secret_key):
    # Top-level rejection
    cfg_top = {"bootstrap_servers": "localhost:9092", secret_key: "leaked_secret"}
    with pytest.raises(ValueError, match="forbidden in YAML configuration"):
        build_client_kwargs(cfg_top, "producer")

    # Client sub-config rejection
    cfg_nested = {"bootstrap_servers": "localhost:9092", "client": {secret_key: "leaked_secret"}}
    with pytest.raises(ValueError, match="forbidden in YAML configuration"):
        build_client_kwargs(cfg_nested, "consumer")


def test_max_partition_fetch_bytes_smaller_than_max_request_size_raises():
    cfg = {
        "bootstrap_servers": "localhost:9092",
        "client": {
            "max_request_size": 2000000,
            "max_partition_fetch_bytes": 1000000,
        },
    }
    with pytest.raises(ValueError, match="must be at least max_request_size"):
        build_client_kwargs(cfg, "consumer")


def test_unknown_keys_ignored_with_warning(caplog):
    cfg = {
        "bootstrap_servers": "localhost:9092",
        "client": {
            "bogus_unsupported_setting": 42,
            "another_random_option": "value",
        },
    }
    with caplog.at_level(logging.WARNING):
        kwargs = build_client_kwargs(cfg, "producer")

    assert "bogus_unsupported_setting" not in kwargs
    assert "another_random_option" not in kwargs
    warning_msgs = [record.message for record in caplog.records if record.levelno == logging.WARNING]
    assert any("bogus_unsupported_setting" in msg for msg in warning_msgs)


def test_compression_type_validation():
    # gzip is supported
    cfg_gzip = {"client": {"compression_type": "gzip"}}
    kwargs = build_client_kwargs(cfg_gzip, "producer")
    assert kwargs["compression_type"] == "gzip"

    # snappy/lz4/zstd unsupported without extra C libs
    cfg_snappy = {"client": {"compression_type": "snappy"}}
    with pytest.raises(ValueError, match="Unsupported compression_type 'snappy'"):
        build_client_kwargs(cfg_snappy, "producer")


def test_invalid_role_raises():
    with pytest.raises(ValueError, match="Invalid role 'invalid'"):
        build_client_kwargs({}, "invalid")
