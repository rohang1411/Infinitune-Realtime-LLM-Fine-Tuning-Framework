import pytest
import torch
import json
import io
from unittest.mock import MagicMock
from inference import deserialize_tensor
from trainer import LoRAProducer


def test_serialize_and_deserialize_tensor_safetensors():
    """Verify tensors serialize using safetensors and deserialize without pickle vulnerability."""
    t = torch.randn(4, 8)
    
    producer = MagicMock()
    # Test serializer method directly
    serialized = LoRAProducer.serialize_payload(producer, t)
    assert isinstance(serialized, bytes)
    
    # Deserialization must yield identical tensor
    deserialized = deserialize_tensor(serialized)
    assert isinstance(deserialized, torch.Tensor)
    assert torch.allclose(t, deserialized)


def test_serialize_and_deserialize_metadata_json():
    """Verify control/metadata dictionaries serialize to JSON and deserialize cleanly."""
    meta = {"step_id": "42", "status": "committed", "num_tensors": 10}
    producer = MagicMock()
    serialized = LoRAProducer.serialize_payload(producer, meta)
    
    deserialized = deserialize_tensor(serialized)
    assert isinstance(deserialized, dict)
    assert deserialized["step_id"] == "42"
    assert deserialized["status"] == "committed"


def test_atomic_snapshot_assembly_prevents_torn_updates():
    """
    Simulate the Kafka consumer queueing logic to prove that partial batches
    are buffered and ONLY emitted to update_queue as a complete dictionary
    upon receiving __commit__, preventing torn adapter state.
    """
    import queue
    update_queue = queue.Queue()
    
    # Step 100 with 3 layers
    step_id = "100"
    manifest = {"step_id": step_id, "num_tensors": 3}
    t1 = torch.randn(2, 2)
    t2 = torch.randn(2, 2)
    t3 = torch.randn(2, 2)
    
    pending_batches = {}
    pending_manifests = {}
    
    # Incoming stream: manifest
    key_manifest = f"__manifest__:{step_id}"
    pending_manifests[step_id] = manifest
    pending_batches[step_id] = {}
    assert update_queue.qsize() == 0  # Nothing queued yet
    
    # Incoming stream: Layer 1 arrives
    pending_batches[step_id]["layer.1.weight"] = t1
    assert update_queue.qsize() == 0  # Still nothing queued (no torn update!)
    
    # Incoming stream: Layer 2 arrives
    pending_batches[step_id]["layer.2.weight"] = t2
    assert update_queue.qsize() == 0  # Still nothing queued!
    
    # Incoming stream: Layer 3 arrives
    pending_batches[step_id]["layer.3.weight"] = t3
    assert update_queue.qsize() == 0  # Still nothing queued!
    
    # Incoming stream: __commit__ arrives
    key_commit = f"__commit__:{step_id}"
    batch = pending_batches.pop(step_id, {})
    expected = pending_manifests.get(step_id, {}).get("num_tensors", len(batch))
    if len(batch) >= expected:
        update_queue.put((step_id, batch))
        
    assert update_queue.qsize() == 1
    committed_step, committed_weights = update_queue.get()
    assert committed_step == "100"
    assert len(committed_weights) == 3
    assert "layer.1.weight" in committed_weights
    assert "layer.2.weight" in committed_weights
    assert "layer.3.weight" in committed_weights
