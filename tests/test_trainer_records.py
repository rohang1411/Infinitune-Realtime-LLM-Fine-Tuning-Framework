import json
import pytest
from trainer import decode_training_record, MetricsLogger


def test_decode_training_record_empty():
    assert decode_training_record(None) == (None, "empty")
    assert decode_training_record(b"") == (None, "empty")
    assert decode_training_record(b"   \r\n\t  ") == (None, "empty")


def test_decode_training_record_not_utf8():
    assert decode_training_record(b"\xff\xfe\x00\x12") == (None, "not_utf8")
    assert decode_training_record(b"\x80invalid_utf8") == (None, "not_utf8")


def test_decode_training_record_not_json():
    assert decode_training_record(b"hello world, this is raw text") == (None, "not_json")
    assert decode_training_record(b"{unclosed_json: true") == (None, "not_json")
    assert decode_training_record(b'{"key": "missing_quote}') == (None, "not_json")


def test_decode_training_record_not_object():
    assert decode_training_record(b"42") == (None, "not_object")
    assert decode_training_record(b'"just a json string"') == (None, "not_object")
    assert decode_training_record(b'[1, 2, 3]') == (None, "not_object")
    assert decode_training_record(b"true") == (None, "not_object")


def test_decode_training_record_valid_record():
    payload = {"input": "Great film!", "target": "positive"}
    raw = json.dumps(payload).encode("utf-8")
    data, reason = decode_training_record(raw)
    assert reason is None
    assert data == payload


def test_decode_training_record_eof_and_verify_markers():
    # Shapes matching producer.py exactly
    verify_payload = {"_verify": True}
    raw_verify = json.dumps(verify_payload).encode("utf-8")
    data_v, reason_v = decode_training_record(raw_verify)
    assert reason_v is None
    assert data_v == verify_payload
    assert "_verify" in data_v

    eof_payload = {"_eof": True}
    raw_eof = json.dumps(eof_payload).encode("utf-8")
    data_e, reason_e = decode_training_record(raw_eof)
    assert reason_e is None
    assert data_e == eof_payload
    assert "_eof" in data_e
    assert data_e.get("_eof") is True


def test_metrics_logger_write_params_merges_malformed_records(tmp_path):
    logger = MetricsLogger(output_dir=str(tmp_path), run_name="test_merge")
    logger.write_params({"project": {"name": "test"}, "training": {"max_steps": 10}})
    
    # Merge malformed_records at end of training
    malformed = {"empty": 2, "not_utf8": 1, "not_json": 0, "not_object": 3}
    logger.write_params({"malformed_records": malformed})
    
    with open(logger.params_path, "r", encoding="utf-8") as f:
        saved_params = json.load(f)
        
    assert "project" in saved_params
    assert saved_params["project"]["name"] == "test"
    assert "training" in saved_params
    assert saved_params["training"]["max_steps"] == 10
    assert "malformed_records" in saved_params
    assert saved_params["malformed_records"] == malformed
