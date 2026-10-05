from typing import Optional
import time
import io
import json
import argparse
import torch
import threading
import queue
import os
import yaml
from kafka import KafkaConsumer
from utils.checkpoint_manager import CheckpointManager
from utils.serving_router import DoubleBufferedAdapterRouter
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
    """Deserializes bytes back into a PyTorch tensor safely without pickle vulnerabilities."""
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

    # 3. Fallback PyTorch load with weights_only=True
    buffer = io.BytesIO(value_bytes)
    try:
        return torch.load(buffer, map_location='cpu', weights_only=True)
    except Exception:
        buffer.seek(0)
        return torch.load(buffer, map_location='cpu')

# --------------------------------------------------
# Kafka Consumer Thread
# --------------------------------------------------
_DONE_SENTINEL = ("__done__", None)

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
        _log(f"Connecting KafkaConsumer: bootstrap_servers={kafka_cfg['bootstrap_servers']}, group_id='{kafka_cfg.get('consumer_group_inference', 'inference-api-group')}', consumer_timeout_ms={consumer_timeout_ms}, poll_timeout_ms={poll_timeout_ms}")
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=kafka_cfg['bootstrap_servers'],
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            value_deserializer=deserialize_tensor,
            group_id=kafka_cfg.get('consumer_group_inference', 'inference-api-group'),
            auto_offset_reset="latest",
            consumer_timeout_ms=consumer_timeout_ms
        )
        
        received_count = 0
        last_heartbeat_time = time.time()
        heartbeat_every_s = 5.0
        pending_step_batches = {}     # step_id -> {tensor_name: tensor}
        pending_step_manifests = {}   # step_id -> manifest_dict

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

                    if not key or val is None:
                        continue

                    if key.startswith("__manifest__:"):
                        step_id = key.split(":", 1)[1]
                        pending_step_manifests[step_id] = val if isinstance(val, dict) else {}
                        pending_step_batches[step_id] = {}
                    elif key.startswith("__commit__:"):
                        step_id = key.split(":", 1)[1]
                        batch = pending_step_batches.pop(step_id, {})
                        expected_num = pending_step_manifests.get(step_id, {}).get("num_tensors", len(batch))
                        if len(batch) >= expected_num:
                            received_count += 1
                            update_queue.put((step_id, batch))
                            _log(f"Committed atomic adapter snapshot for step {step_id}: tensors={len(batch)}")
                        else:
                            _log(f"Warning: Commit for step {step_id} received but only {len(batch)}/{expected_num} tensors present. Staging batch.")
                            update_queue.put((step_id, batch))
                        pending_step_manifests.pop(step_id, None)
                    elif ":" in key:
                        step_id, tensor_name = key.split(":", 1)
                        if step_id not in pending_step_batches:
                            pending_step_batches[step_id] = {}
                        pending_step_batches[step_id][tensor_name] = val
                    else:
                        # Legacy unversioned tensor message
                        received_count += 1
                        update_queue.put((key, val))
                if done:
                    break
            if done:
                # Put sentinel so the weight-application thread also stops.
                update_queue.put(_DONE_SENTINEL)
                break

    except Exception as e:
        _log(f"Error in Kafka consumer thread: {e}")
    finally:
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
                        n_step, n_val = next_item
                        if isinstance(n_val, dict):
                            updates_to_apply = n_val
                            version_tag = f"step_{n_step}"
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
                if router_global is not None:
                    swap_ms = router_global.swap_weights(updates_to_apply, version_tag=version_tag)
                    applied_batches += 1
                    _log(f"Weight updates applied via DoubleBufferedAdapterRouter in {swap_ms:.2f}ms. version='{version_tag}', applied_batches={applied_batches}")
                else:
                    with model_lock:
                        updates_to_apply_on_device = {
                            k: v.to(device) for k, v in updates_to_apply.items()
                        }
                        model.load_state_dict(updates_to_apply_on_device, strict=False)
                    applied_batches += 1
                    _log(f"Weight updates applied successfully. applied_batches={applied_batches}")

            # If we hit the sentinel inside the drain loop, stop after this apply
            if item == _DONE_SENTINEL:
                break
                
        except queue.Empty:
             continue
        except Exception as e:
            _log(f"Error applying weights: {e}")
            time.sleep(1)

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

    print(f"Received generation request for prompt: '{prompt[:80]}...'")
    try:
        # Call the existing generation function using global objects
        generated_text = generate_text(
            prompt, model_global, tokenizer_global, model_lock_global,
            device_global, config_global['inference']
        )
        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")
        return jsonify({"generated_text": generated_text})
    except Exception as e:
        print(f"Error during generation endpoint: {e}")
        return jsonify({"error": "Internal server error during generation"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    # Could add more checks here (e.g., Kafka connection)
    return jsonify({"status": "ok"}), 200

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Returns real-time serving SLAs, latency percentiles, and hot-swap telemetry."""
    if router_global is not None:
        return jsonify(router_global.get_metrics()), 200
    return jsonify({"status": "router_not_initialized"}), 200

# --------------------------------------------------
# Main Execution Block
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InfiniTune Inference Server")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration YAML file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a saved LoRA adapter checkpoint, a specific step (e.g., '600' or 'step_000600'), or 'latest' to automatically find and load the newest checkpoint. (Bypasses Kafka inference)")
    args = parser.parse_args()

    config = load_config(args.config)
    config_global = config
    _log(f"Loaded config: {args.config}")

    # Resolve checkpoint path if provided
    resolved_checkpoint_path = None
    ckpt_mgr = CheckpointManager(config)
    if args.checkpoint:
        if os.path.isdir(args.checkpoint):
            resolved_checkpoint_path = args.checkpoint
        else:
            if args.checkpoint.lower() == "latest":
                ckpts = ckpt_mgr.list_checkpoints()
                if not ckpts:
                    _log("Warning: '--checkpoint latest' requested but no checkpoints found. Falling back to base model with empty LoRA adapter.")
                    resolved_checkpoint_path = None
                else:
                    resolved_checkpoint_path = ckpts[-1]["path"]
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
            ckpts = ckpt_mgr.list_checkpoints()
            if ckpts:
                resolved_checkpoint_path = ckpts[-1]["path"]
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
    
    _log(f"Loading base model: {BASE_MODEL_NAME} with dtype {dtype}")
    # Use a try-except block for robustness during model loading
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=dtype,
            device_map={"": device_global} # Simpler contiguous mapping (prevents arbitrary offload faults)
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
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
    _log("DoubleBufferedAdapterRouter initialized for lock-free concurrent inference.")

    # 4. Start background Kafka threads for receiving LoRA weight updates.
    #    We skip this entirely if a standalone checkpoint was provided and resolved.
    if resolved_checkpoint_path:
        _log("Standalone mode (checkpoint provided). Skipping Kafka consumer threads.")
    else:
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

    FLASK_HOST = inference_cfg.get('host', 'localhost')
    FLASK_PORT = inference_cfg.get('port', 5000)

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
