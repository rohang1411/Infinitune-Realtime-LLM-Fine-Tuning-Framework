import time
import io
import json
import argparse
import torch
import threading
import queue
import yaml
from kafka import KafkaConsumer
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
    """Deserializes bytes back into a PyTorch tensor."""
    buffer = io.BytesIO(value_bytes)
    tensor = torch.load(buffer, map_location='cpu') 
    return tensor

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
                    # Trainer sends '__done__' after final weight push to signal
                    # that training is over and no more updates will arrive.
                    if message.key == "__done__":
                        _log("Received training-done signal from trainer. Stopping LoRA listener.")
                        done = True
                        break
                    if message.key and message.value is not None:
                        received_count += 1
                        update_queue.put((message.key, message.value))
                    else:
                        _log("Warning: Received message with missing key or value.")
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

            layer_name, tensor = item
            updates_to_apply = {layer_name: tensor}
            
            # Drain any other updates currently in the queue non-blockingly
            while True:
                try:
                    item = update_queue.get(block=False)
                    if item == _DONE_SENTINEL:
                        _log("Weight application thread received done signal. Applying remaining batch, then stopping.")
                        # Apply what we have, then break both loops
                        break
                    name, t = item
                    updates_to_apply[name] = t
                except queue.Empty:
                    break
            
            if updates_to_apply:
                _log(f"Applying weight updates: tensors={len(updates_to_apply)}, queue_size_before_apply={update_queue.qsize()}")
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

# --------------------------------------------------
# Main Execution Block
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InfiniTune Inference Server")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    config_global = config
    _log(f"Loaded config: {args.config}")

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

    # 1. Load the base model and tokenizer
    BASE_MODEL_NAME = model_cfg['name']
    _log(f"Loading base model: {BASE_MODEL_NAME}")
    # Use a try-except block for robustness during model loading
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32, 
            device_map="auto" # Let accelerate handle device mapping if multiple GPUs/MPS
            # device_map={"" : DEVICE} # Simpler mapping if single device
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token 
    except Exception as e:
        _log(f"FATAL: Failed to load base model or tokenizer: {e}")
        exit(1) # Exit if model loading fails

    # 2. Apply the initial LoRA configuration to the base model
    _log("Applying initial LoRA configuration...")
    try:
        model = get_peft_model(base_model, LORA_CONFIG)
        model.eval() # Set the model to evaluation mode
        _log("Model loaded and LoRA configured.")
        model.print_trainable_parameters() 
    except Exception as e:
        _log(f"FATAL: Failed to apply LoRA config: {e}")
        exit(1)

    # Assign to global variables for access by Flask routes
    model_global = model
    tokenizer_global = tokenizer
    
    # 3. Create shared resources: update queue and model lock
    update_queue = queue.Queue()
    model_lock = threading.Lock()
    model_lock_global = model_lock

    # 4. Start background Kafka threads for receiving LoRA weight updates.
    #    These threads stop automatically when the trainer sends a '__done__' signal.
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
