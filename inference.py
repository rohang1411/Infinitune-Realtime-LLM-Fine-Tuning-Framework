from typing import Optional
import time
import json
import argparse
import torch
import threading
import queue
import os
from collections import deque
import traceback
import yaml
from kafka import KafkaConsumer
from utils.checkpoint_manager import CheckpointManager
from utils.serving_router import DoubleBufferedAdapterRouter
from utils.stream_protocol import SnapshotAssembler, deserialize_tensor
from utils.serving_policy import should_listen_for_updates
from utils.kafka_config import build_client_kwargs, redact_client_kwargs
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model, PeftModel
from flask import Flask, request, jsonify

def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _log(msg):
    print(f"[{_ts()}][INFERENCE] {msg}", flush=True)

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def deserialize_tensor(value_bytes):
    """Decode safetensors or JSON metadata; never unpickle broker bytes."""
    if not value_bytes:
        return None

    # 1. Safetensors format (fast, zero-copy, safe)
    try:
        from safetensors.torch import load
        loaded = load(value_bytes)
        if isinstance(loaded, dict) and "weight" in loaded:
            return loaded["weight"]
        return loaded
    except Exception:
        pass

    # 2. JSON control / manifest packet
    try:
        if value_bytes.startswith(b"{") and value_bytes.endswith(b"}"):
            return json.loads(value_bytes.decode("utf-8"))
    except Exception:
        pass

    raise ValueError("Kafka value is neither a safetensors tensor nor JSON metadata")

# --------------------------------------------------
# Kafka Consumer Thread
# --------------------------------------------------
_DONE_SENTINEL = ("__done__", None)
stream_state_global = {
    "enabled": False,
    "consumer_ready": False,
    "consumer_error": None,
    "applier_error": None,
    "counters": {
        "rejected_frames": 0,
        "rejected_by_reason": {},
    },
}
initial_active_global = None
adapter_history_global = deque(maxlen=200)


def process_adapter_record(assembler, key, raw_value, counters=None) -> Optional[tuple]:
    """Return (step, batch) on a validated commit, else None. Never raises."""
    try:
        if not key or raw_value is None:
            raise ValueError(f"Missing key or raw_value (key={key!r}, value_is_none={raw_value is None})")
        return assembler.add(key, raw_value)
    except Exception as exc:
        if counters is not None:
            counters["rejected_frames"] = counters.get("rejected_frames", 0) + 1
            by_reason = counters.setdefault("rejected_by_reason", {})
            reason_name = type(exc).__name__
            by_reason[reason_name] = by_reason.get(reason_name, 0) + 1
        msg = f"REJECTED adapter frame key={key!r} reason={type(exc).__name__}: {exc}"
        if len(msg) > 300:
            msg = msg[:300]
        _log(msg)
        return None


def kafka_consumer_thread(update_queue: queue.Queue, config: dict):
    """
    Listens to the Kafka topic for LoRA weight updates and puts them in a queue.
    Stops gracefully when it receives a '__done__' signal from the trainer.
    """
    kafka_cfg = config['kafka']
    topic = kafka_cfg['lora_updates_topic']
    _log(f"Consumer thread started, listening to topic '{topic}'...")
    consumer = None
    try:
        consumer_timeout_ms = int(kafka_cfg.get('consumer_timeout_ms', 1000))
        poll_timeout_ms = int(kafka_cfg.get('poll_timeout_ms', 1000))
        consumer_kwargs = build_client_kwargs(kafka_cfg, "consumer")
        _log(f"Connecting KafkaConsumer: kwargs={redact_client_kwargs(consumer_kwargs)}, group_id='{kafka_cfg.get('consumer_group_inference', 'inference-api-group')}', consumer_timeout_ms={consumer_timeout_ms}, poll_timeout_ms={poll_timeout_ms}")
        consumer = KafkaConsumer(
            topic,
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            group_id=kafka_cfg.get('consumer_group_inference', 'inference-api-group'),
            auto_offset_reset="latest",
            consumer_timeout_ms=consumer_timeout_ms,
            **consumer_kwargs,
        )
        # A producer can publish only after this marker: `latest` offset is
        # established once the consumer has joined and received assignment.
        ready_deadline = time.monotonic() + 30
        while not consumer.assignment() and time.monotonic() < ready_deadline:
            consumer.poll(timeout_ms=250)
        if not consumer.assignment():
            raise TimeoutError("Kafka listener did not receive a partition assignment")
        stream_state_global["consumer_ready"] = True
        _log(f"Kafka listener ready: assignment={consumer.assignment()}")
        
        received_count = 0
        last_heartbeat_time = time.time()
        heartbeat_every_s = 5.0
        assembler = SnapshotAssembler(
            integrity=kafka_cfg.get("integrity", "sha256"),
            initial_active=initial_active_global,
        )

        while True:
            messages = consumer.poll(timeout_ms=poll_timeout_ms)
            if not messages:
                 now = time.time()
                 if now - last_heartbeat_time >= heartbeat_every_s:
                     _log(f"Waiting for LoRA updates... received_count={received_count}, queue_size={update_queue.qsize()}")
                     last_heartbeat_time = now
                 time.sleep(0.1)
                 continue

            done = False
            for tp, records in messages.items():
                for message in records:
                    key = message.key
                    val = message.value
                    if key == "__done__":
                        _log("Received training-done signal from trainer. Stopping LoRA listener.")
                        done = True
                        break

                    complete = process_adapter_record(
                        assembler,
                        key,
                        val,
                        counters=stream_state_global.get("counters"),
                    )
                    if complete is not None:
                        t_assembled = time.time()
                        step_id, batch = complete
                        received_count += 1
                        update_queue.put((step_id, batch, getattr(assembler, "active", None), t_assembled))
                        _log(f"Committed complete adapter step {step_id}: tensors={len(batch)}")
                if done:
                    break
            if done:
                # Put sentinel so the weight-application thread also stops.
                update_queue.put(_DONE_SENTINEL)
                break

    except Exception as e:
        stream_state_global["consumer_error"] = f"{type(e).__name__}: {e}"
        _log(f"FATAL Kafka consumer thread: {stream_state_global['consumer_error']}\n{traceback.format_exc()}")
    finally:
        stream_state_global["consumer_ready"] = False
        if consumer:
            consumer.close()
        _log("Consumer thread finished.")


# --------------------------------------------------
# Weight Application Thread
# --------------------------------------------------
def weight_application_thread(model: PeftModel, update_queue: queue.Queue,
                              model_lock: threading.Lock, device: str):
    """
    Applies LoRA weight updates from the queue to the model.
    Stops when it receives the _DONE_SENTINEL from the consumer thread.
    """
    _log("Weight application thread started...")
    applied_batches = 0
    while True:
        try:
            # Wait for the first update (blocking)
            item = update_queue.get(block=True, timeout=None)

            # Check for the done sentinel pushed by the consumer thread
            if item == _DONE_SENTINEL:
                _log("Weight application thread received done signal. Stopping.")
                break

            active_run_id = None
            active_seq = None
            t_publish_start = None
            t_assembled = None
            if len(item) == 4:
                step_or_layer, tensor_or_dict, active_meta, t_assembled = item
                if active_meta:
                    active_run_id, active_seq = active_meta[0], active_meta[1]
                    if len(active_meta) >= 3:
                        t_publish_start = float(active_meta[2])
            elif len(item) == 3:
                step_or_layer, tensor_or_dict, active_meta = item
                if active_meta:
                    active_run_id, active_seq = active_meta[0], active_meta[1]
                    if len(active_meta) >= 3:
                        t_publish_start = float(active_meta[2])
            else:
                step_or_layer, tensor_or_dict = item

            if isinstance(tensor_or_dict, dict):
                # Complete atomic snapshot delivered via commit marker (prevents torn updates)
                updates_to_apply = tensor_or_dict
                version_tag = f"step_{step_or_layer}"
                while True:
                    try:
                        next_item = update_queue.get(block=False)
                        if next_item == _DONE_SENTINEL:
                            item = next_item
                            break
                        if len(next_item) == 4:
                            n_step, n_val, n_meta, n_tassembled = next_item
                        elif len(next_item) == 3:
                            n_step, n_val, n_meta = next_item
                            n_tassembled = None
                        else:
                            n_step, n_val = next_item
                            n_meta = None
                            n_tassembled = None
                        if isinstance(n_val, dict):
                            updates_to_apply = n_val
                            version_tag = f"step_{n_step}"
                            if n_meta:
                                active_run_id, active_seq = n_meta[0], n_meta[1]
                                if len(n_meta) >= 3:
                                    t_publish_start = float(n_meta[2])
                            if n_tassembled:
                                t_assembled = n_tassembled
                        else:
                            updates_to_apply[n_step] = n_val
                    except queue.Empty:
                        break
            else:
                layer_name = step_or_layer
                updates_to_apply = {layer_name: tensor_or_dict}
                version_tag = f"batch_{applied_batches + 1}"
                while True:
                    try:
                        next_item = update_queue.get(block=False)
                        if next_item == _DONE_SENTINEL:
                            item = next_item
                            break
                        if len(next_item) == 4:
                            name, t, _, _ = next_item
                        elif len(next_item) == 3:
                            name, t, _ = next_item
                        else:
                            name, t = next_item
                        if isinstance(t, dict):
                            updates_to_apply = t
                            version_tag = f"step_{name}"
                        else:
                            updates_to_apply[name] = t
                    except queue.Empty:
                        break

            if updates_to_apply:
                _log(f"Applying weight updates: tensors={len(updates_to_apply)}, queue_size_before_apply={update_queue.qsize()}")
                global active_version_global
                if serving_mode_global == "router" and router_global is not None:
                    swap_ms = router_global.swap_weights(
                        updates_to_apply,
                        version_tag=version_tag,
                        run_id=active_run_id,
                        seq=active_seq,
                    )
                    active_version_global = version_tag
                    applied_batches += 1
                    _log(f"Weight updates applied via DoubleBufferedAdapterRouter in {swap_ms:.2f}ms. version='{version_tag}', applied_batches={applied_batches}")
                else:
                    swap_start = time.perf_counter()
                    with model_lock:
                        updates_to_apply_on_device = {
                            k: v.to(device) for k, v in updates_to_apply.items()
                        }
                        model.load_state_dict(updates_to_apply_on_device, strict=False)
                        active_version_global = version_tag
                    swap_ms = (time.perf_counter() - swap_start) * 1000.0
                    applied_batches += 1
                    _log(f"Weight updates applied via legacy model_lock in {swap_ms:.2f}ms. version='{version_tag}', applied_batches={applied_batches}")

                t_applied = time.time()
                adapter_history_global.append({
                    "version": version_tag,
                    "run_id": active_run_id,
                    "seq": active_seq,
                    "t_publish_start": t_publish_start,
                    "t_assembled": t_assembled,
                    "t_applied": t_applied,
                })

            # If we hit the sentinel inside the drain loop, stop after this apply
            if item == _DONE_SENTINEL:
                break
                
        except queue.Empty:
             continue
        except Exception as e:
            stream_state_global["applier_error"] = f"{type(e).__name__}: {e}"
            _log(f"FATAL adapter apply: {stream_state_global['applier_error']}\n{traceback.format_exc()}")
            break

# --------------------------------------------------
# Inference Function
# --------------------------------------------------
def generate_text(prompt: str, model: PeftModel, tokenizer: AutoTokenizer,
                  model_lock: threading.Lock, device: str, inference_cfg: dict):
    """
    Generates text using the current state of the LoRA-adapted model.
    Returns only the completion (new tokens), not the echoed prompt.
    """
    with model_lock:
        start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_token_len = inputs['input_ids'].shape[1]
        
        generation_config = GenerationConfig(
            max_new_tokens=inference_cfg.get('max_new_tokens', 100),
            do_sample=bool(inference_cfg.get('do_sample', True)),
            temperature=inference_cfg.get('temperature', 0.7),
            top_p=inference_cfg.get('top_p', 0.9),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        with torch.no_grad(): 
            outputs = model.generate(**inputs, generation_config=generation_config)

        # Slice off the prompt tokens and decode only the generated completion
        generated_ids = outputs[0, prompt_token_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        _log(f"generate_text completed in {time.time() - start:.2f}s (prompt_len={len(prompt)}, gen_tokens={len(generated_ids)})")
        
    return generated_text

# --------------------------------------------------
# Flask App Setup
# --------------------------------------------------
app = Flask(__name__)

# Global variables to hold the model, tokenizer, lock, and config
# These will be initialized in the main block
model_global = None
tokenizer_global = None
model_lock_global = None
device_global = None
config_global = None
router_global: Optional[DoubleBufferedAdapterRouter] = None
serving_mode_global: str = "router"
active_version_global: str = "v0_base" 

@app.route('/generate', methods=['POST'])
def handle_generate():
    """API endpoint to handle generation requests."""
    start_time = time.time()
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({"error": "Missing 'prompt' in JSON body"}), 400
        
    # Check if model is initialized (should be unless server started incorrectly)
    if model_global is None or tokenizer_global is None or model_lock_global is None:
         print("Error: Model not initialized when request received.")
         return jsonify({"error": "Model not initialized yet. Please wait."}), 503 # Service Unavailable

    print(f"Received generation request for prompt: '{prompt[:80]}...' [mode={serving_mode_global}]")
    try:
        if serving_mode_global == "router" and router_global is not None:
            res = router_global.generate(prompt, config_global['inference'])
            end_time = time.time()
            resp = jsonify({
                "generated_text": res["response"],
                "adapter_version": res["version"],
                "latency_ms": res["latency_ms"],
                "serving_mode": "router",
            })
            resp.headers["X-Adapter-Version"] = res["version"]
            return resp, 200
        else:
            generated_text = generate_text(
                prompt, model_global, tokenizer_global, model_lock_global,
                device_global, config_global['inference']
            )
            end_time = time.time()
            print(f"Generation finished in {end_time - start_time:.2f} seconds.")
            v = active_version_global
            resp = jsonify({
                "generated_text": generated_text,
                "adapter_version": v,
                "latency_ms": round((end_time - start_time) * 1000.0, 2),
                "serving_mode": "legacy_lock",
            })
            resp.headers["X-Adapter-Version"] = v
            return resp, 200
    except Exception as e:
        print(f"Error during generation endpoint: {e}")
        return jsonify({"error": "Internal server error during generation"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Flag background-stream failures instead of reporting false health."""
    if stream_state_global["enabled"]:
        if stream_state_global["consumer_error"] or stream_state_global["applier_error"]:
            return jsonify({"status": "degraded", "stream": stream_state_global}), 503
    return jsonify({"status": "ok"}), 200

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Returns real-time serving SLAs, latency percentiles, and hot-swap telemetry."""
    if router_global is not None:
        return jsonify({**router_global.get_metrics(), "stream": stream_state_global}), 200
    return jsonify({"status": "router_not_initialized", "stream": stream_state_global}), 200

@app.route('/adapter', methods=['GET'])
def get_adapter():
    """Returns cryptographic state digest of the active adapter and verification status."""
    if router_global is not None:
        live_digest = router_global.adapter_state_digest()
        expected = router_global.expected_state_sha256
        match = bool(live_digest and expected and live_digest == expected)
        return jsonify({
            "active_version": router_global.active_version,
            "run_id": router_global.active_run_id,
            "seq": router_global.active_seq,
            "adapter_state_sha256": live_digest,
            "expected_state_sha256": expected,
            "match": match,
        }), 200

    run_id = initial_active_global[0] if initial_active_global else None
    seq = initial_active_global[1] if initial_active_global else None
    return jsonify({
        "active_version": active_version_global,
        "run_id": run_id,
        "seq": seq,
        "adapter_state_sha256": None,
        "expected_state_sha256": None,
        "match": False,
    }), 200

@app.route('/adapter/history', methods=['GET'])
def get_adapter_history():
    """Returns bounded dictionary/list of applied adapter versions and latency timestamps."""
    hist_list = list(adapter_history_global)
    hist_dict = {entry["version"]: entry for entry in hist_list}
    return jsonify({
        "history": hist_list,
        "versions": hist_dict,
        **hist_dict,
    }), 200


# --------------------------------------------------
# Main Execution Block
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InfiniTune Inference Server")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration YAML file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a saved LoRA adapter checkpoint, a specific step (e.g., '600' or 'step_000600'), or 'latest' to automatically find and load the newest checkpoint. (Bypasses Kafka inference)")
    parser.add_argument("--listen-updates", action="store_true",
                        help="Continue consuming Kafka adapter snapshots after loading a checkpoint")
    parser.add_argument("--no-listen-updates", action="store_true",
                        help="Disable consuming Kafka adapter snapshots even if auto-loading checkpoints")
    parser.add_argument("--serving-mode", type=str, choices=["router", "legacy_lock"], default="router",
                        help="Serving mode: 'router' uses DoubleBufferedAdapterRouter with ReadWriteLock; 'legacy_lock' uses global mutex")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind the Flask HTTP server to (default: 8000)")
    args = parser.parse_args()
    if args.listen_updates and args.no_listen_updates:
        parser.error("Cannot specify both --listen-updates and --no-listen-updates.")

    config = load_config(args.config)
    config_global = config
    serving_mode_global = args.serving_mode
    _log(f"Loaded config: {args.config}")
    _log(f"Serving mode configured: '{serving_mode_global}'")

    # Resolve checkpoint path if provided
    resolved_checkpoint_path = None
    checkpoint_source = None
    ckpt_mgr = CheckpointManager(config)
    if args.checkpoint:
        checkpoint_source = "explicit"
        if os.path.isdir(args.checkpoint):
            resolved_checkpoint_path = args.checkpoint
        else:
            if args.checkpoint.lower() == "latest":
                latest_ckpt = ckpt_mgr.latest_checkpoint()
                if not latest_ckpt:
                    _log("Warning: '--checkpoint latest' requested but no checkpoints found. Falling back to base model with empty LoRA adapter.")
                    resolved_checkpoint_path = None
                    checkpoint_source = None
                else:
                    resolved_checkpoint_path = latest_ckpt["path"]
                    _log(f"Auto-discovered latest checkpoint: {resolved_checkpoint_path}")
            else:
                candidate_path = ckpt_mgr.resolve_checkpoint_path(args.checkpoint)
                if candidate_path and os.path.exists(candidate_path):
                    resolved_checkpoint_path = candidate_path
                    _log(f"Resolved step '{args.checkpoint}' to: {resolved_checkpoint_path}")
                else:
                    _log(f"FATAL: Checkpoint '{args.checkpoint}' not found locally at '{candidate_path}' and is not a valid directory.")
                    exit(1)
    elif config.get('inference', {}).get('auto_load_latest_checkpoint', True):
        # Cold-start resilience: Auto-load latest saved checkpoint to avoid serving stale base weights
        try:
            latest_ckpt = ckpt_mgr.latest_checkpoint()
            if latest_ckpt:
                resolved_checkpoint_path = latest_ckpt["path"]
                checkpoint_source = "auto"
                _log(f"Cold-start recovery: Discovered existing latest checkpoint on disk: {resolved_checkpoint_path}")
        except Exception as e:
            _log(f"Cold-start checkpoint check skipped: {e}")

    model_cfg = config['model']
    lora_cfg = config['lora']
    inference_cfg = config['inference']
    kafka_cfg = config['kafka']

    # Build LoRA config from YAML (must match trainer)
    LORA_CONFIG = LoraConfig(
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['alpha'],
        target_modules=lora_cfg['target_modules'],
        lora_dropout=lora_cfg.get('dropout', 0.05),
        bias=lora_cfg.get('bias', 'none'),
        task_type=model_cfg.get('task_type', 'CAUSAL_LM'),
    )

    # Determine device
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    device_global = DEVICE

    _log("Initializing inference server with API endpoint...")
    _log(f"Using device: {DEVICE}")
    _log(f"Kafka bootstrap_servers={kafka_cfg.get('bootstrap_servers')}, lora_updates_topic='{kafka_cfg.get('lora_updates_topic')}', consumer_group='{kafka_cfg.get('consumer_group_inference', 'inference-api-group')}'")

    BASE_MODEL_NAME = model_cfg['name']
    prec = model_cfg.get('precision', 'fp32')
    dtype = torch.float16 if prec == 'fp16' else (torch.bfloat16 if prec == 'bf16' else torch.float32)
    
    model_revision = model_cfg.get('revision') or (
        os.environ.get('INFINITUNE_MODEL_REVISION')
        if os.environ.get('INFINITUNE_MODEL_NAME') == BASE_MODEL_NAME else None)
    _log(f"Loading base model: {BASE_MODEL_NAME} revision={model_revision or 'default'} with dtype {dtype}")
    # Use a try-except block for robustness during model loading
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=dtype,
            device_map={"": device_global}, # Simpler contiguous mapping (prevents arbitrary offload faults)
            revision=model_revision,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, revision=model_revision)
        tokenizer.pad_token = tokenizer.eos_token 
    except Exception as e:
        _log(f"FATAL: Failed to load base model or tokenizer: {e}")
        exit(1) # Exit if model loading fails

    # 2. Apply the initial LoRA configuration to the base model (or load from checkpoint)
    _log("Applying initial LoRA configuration...")
    try:
        if resolved_checkpoint_path:
            _log(f"Loading standalone checkpoint from disk: {resolved_checkpoint_path}")
            model = PeftModel.from_pretrained(base_model, resolved_checkpoint_path)
            _log("Model checkpoint loaded successfully.")
        else:
            model = get_peft_model(base_model, LORA_CONFIG)
            model.print_trainable_parameters() 
        model.eval() # Set the model to evaluation mode
    except Exception as e:
        _log(f"FATAL: Failed to apply LoRA config / checkpoint: {e}")
        exit(1)

    # Assign to global variables for access by Flask routes
    model_global = model
    tokenizer_global = tokenizer
    
    # 3. Create shared resources: update queue and model lock
    update_queue = queue.Queue()
    model_lock = threading.Lock()
    model_lock_global = model_lock
    router_global = DoubleBufferedAdapterRouter(model, tokenizer, device=device_global)
    if resolved_checkpoint_path:
        router_global.active_version = os.path.basename(resolved_checkpoint_path)
        c_run = os.path.basename(os.path.dirname(resolved_checkpoint_path))
        c_seq = None
        base_name = os.path.basename(resolved_checkpoint_path)
        if base_name.startswith("step_"):
            try:
                c_seq = int(base_name.split("_", 1)[1])
            except ValueError:
                c_seq = None
        c_created = None
        meta_file = os.path.join(resolved_checkpoint_path, "checkpoint_meta.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, encoding="utf-8") as f:
                    c_meta = json.load(f)
                if c_meta.get("run_id"):
                    c_run = c_meta.get("run_id")
                if c_meta.get("seq") is not None:
                    c_seq = c_meta.get("seq")
                elif c_meta.get("step") is not None:
                    c_seq = c_meta.get("step")
                c_created = c_meta.get("created_at")
            except Exception as e:
                _log(f"Cold-start active metadata read skipped: {e}")
        if c_run and c_seq is not None:
            c_created = c_created or time.time()
            initial_active_global = (str(c_run), int(c_seq), float(c_created))
            router_global.active_run_id = str(c_run)
            router_global.active_seq = int(c_seq)
            _log(f"Initialized active version from cold start: {initial_active_global}")
        try:
            router_global.expected_state_sha256 = router_global.adapter_state_digest()
        except Exception as e:
            _log(f"Cold-start adapter digest computation skipped: {e}")
    _log("Adapter router initialized with single-model read/write locking.")

    # 4. Start background Kafka threads for receiving LoRA weight updates.
    listen_updates = should_listen_for_updates(
        checkpoint_source=checkpoint_source,
        listen_flag=args.listen_updates,
        no_listen_flag=args.no_listen_updates,
    )
    if not listen_updates:
        _log(f"Kafka update listener disabled (checkpoint_source='{checkpoint_source}'). Skipping Kafka consumer threads.")
    else:
        _log(f"Kafka update listener enabled (checkpoint_source='{checkpoint_source}'). Starting Kafka consumer and applier threads.")
        stream_state_global["enabled"] = True
        consumer_thread = threading.Thread(
            target=kafka_consumer_thread,
            args=(update_queue, config),
            daemon=True
        )
        applier_thread = threading.Thread(
            target=weight_application_thread,
            args=(model_global, update_queue, model_lock_global, DEVICE),
            daemon=True
        )
        consumer_thread.start()
        applier_thread.start()

    FLASK_HOST = inference_cfg.get('host', '0.0.0.0')
    FLASK_PORT = args.port or inference_cfg.get('port', 8000)

    _log(f"Starting Flask server on http://{FLASK_HOST}:{FLASK_PORT}")
    _log("Send POST requests to /generate with JSON body: {'prompt': 'your prompt here'}")
    _log("GET /health for health check.")
    
    # 5. Start Flask server (blocking call)
    # Use 'threaded=True' explicitly if needed, though it's often default.
    # 'debug=False' is recommended for stability unless actively debugging Flask itself.
    try:
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True) 
    except Exception as e:
         _log(f"FATAL: Failed to start Flask server: {e}")
         exit(1)

    # Code after app.run() executes only when the server stops (e.g., Ctrl+C)
    _log("Inference server stopped.")
