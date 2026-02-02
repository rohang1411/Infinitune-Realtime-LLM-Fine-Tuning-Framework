import time
import io
import json
import torch
import threading
import queue
from kafka import KafkaConsumer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model, PeftModel
from flask import Flask, request, jsonify # Added Flask imports

# --------------------------------------------------
# Configuration
# --------------------------------------------------
KAFKA_BROKER = "localhost:9092"
LORA_UPDATES_TOPIC = "lora-updates"
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" 
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["self_attn.q_proj", "self_attn.v_proj"], # Must match trainer
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 100 
FLASK_HOST = "localhost" # Use '0.0.0.0' to make accessible on network
FLASK_PORT = 5000

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
def kafka_consumer_thread(update_queue: queue.Queue):
    """
    Listens to the Kafka topic for LoRA weight updates and puts them in a queue.
    (No changes from previous version)
    """
    print(f"Consumer thread started, listening to topic '{LORA_UPDATES_TOPIC}'...")
    consumer = None # Initialize consumer to None
    try:
        consumer = KafkaConsumer(
            LORA_UPDATES_TOPIC,
            bootstrap_servers=[KAFKA_BROKER],
            key_deserializer=lambda k: k.decode("utf-8") if k else None, 
            value_deserializer=deserialize_tensor, 
            group_id="inference-api-group", # Changed group ID slightly
            auto_offset_reset="latest", 
            consumer_timeout_ms=1000 
        )
        
        while True: 
            messages = consumer.poll(timeout_ms=1000) # Poll for messages
            if not messages: # No messages, continue loop
                 time.sleep(0.1)
                 continue
            print("Message received at:", time.time())
            for tp, records in messages.items():
                for message in records:
                    if message.key and message.value is not None:
                        print(f"Received update for layer: {message.key}") # Can be verbose
                        update_queue.put((message.key, message.value))
                    else:
                        print(f"Warning: Received message with missing key or value.")

    except Exception as e:
        print(f"Error in Kafka consumer thread: {e}")
    finally:
        if consumer:
            consumer.close()
        print("Consumer thread finished.")


# --------------------------------------------------
# Weight Application Thread
# --------------------------------------------------
def weight_application_thread(model: PeftModel, update_queue: queue.Queue, model_lock: threading.Lock):
    """
    Applies LoRA weight updates from the queue to the model.
    (No changes from previous version)
    """
    print("Weight application thread started...")
    while True:
        try:
            # Wait for the first update (blocking)
            layer_name, tensor = update_queue.get(block=True, timeout=None) 
            
            updates_to_apply = {layer_name: tensor}
            
            # Process any other updates currently in the queue non-blockingly
            while True:
                try:
                    layer_name, tensor = update_queue.get(block=False)
                    updates_to_apply[layer_name] = tensor
                except queue.Empty:
                    break # No more updates in the queue for now
            
            if updates_to_apply:
                print(f"Applying {len(updates_to_apply)} weight updates...") # Can be verbose
                with model_lock: # Acquire lock before modifying model state
                    updates_to_apply_on_device = {
                        k: v.to(DEVICE) for k, v in updates_to_apply.items()
                    }
                    model.load_state_dict(updates_to_apply_on_device, strict=False) 
                print("Weight updates applied successfully.") # Can be verbose
                
        except queue.Empty:
             # This happens if the initial get times out (if timeout is set)
             continue
        except Exception as e:
            print(f"Error applying weights: {e}")
            time.sleep(1) 

# --------------------------------------------------
# Inference Function (Internal logic unchanged)
# --------------------------------------------------
def generate_text(prompt: str, model: PeftModel, tokenizer: AutoTokenizer, model_lock: threading.Lock):
    """
    Generates text using the current state of the LoRA-adapted model.
    """
    # print(f"\nRunning inference for prompt: '{prompt[:50]}...'") # Logged in API route now
    with model_lock: # Acquire lock to ensure model state is stable during generation
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        generation_config = GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True, 
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id 
        )
        
        with torch.no_grad(): 
            outputs = model.generate(**inputs, generation_config=generation_config)
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    return generated_text

# --------------------------------------------------
# Flask App Setup
# --------------------------------------------------
app = Flask(__name__)

# Global variables to hold the model, tokenizer, and lock
# These will be initialized in the main block
model_global = None
tokenizer_global = None
model_lock_global = None

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
        generated_text = generate_text(prompt, model_global, tokenizer_global, model_lock_global)
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
    print("Initializing inference server with API endpoint...")
    print(f"Using device: {DEVICE}")

    # 1. Load the base model and tokenizer
    print(f"Loading base model: {BASE_MODEL_NAME}")
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
        print(f"FATAL: Failed to load base model or tokenizer: {e}")
        exit(1) # Exit if model loading fails

    # 2. Apply the initial LoRA configuration to the base model
    print("Applying initial LoRA configuration...")
    try:
        model = get_peft_model(base_model, LORA_CONFIG)
        model.eval() # Set the model to evaluation mode
        print("Model loaded and LoRA configured.")
        model.print_trainable_parameters() 
    except Exception as e:
        print(f"FATAL: Failed to apply LoRA config: {e}")
        exit(1)

    # Assign to global variables for access by Flask routes
    model_global = model
    tokenizer_global = tokenizer
    
    # 3. Create shared resources: update queue and model lock
    update_queue = queue.Queue()
    model_lock = threading.Lock()
    model_lock_global = model_lock # Assign lock to global var

    # 4. Start the background threads
    consumer_thread = threading.Thread(
        target=kafka_consumer_thread, 
        args=(update_queue,), 
        daemon=True 
    )
    applier_thread = threading.Thread(
        target=weight_application_thread, 
        args=(model_global, update_queue, model_lock_global), 
        daemon=True
    )
    
    consumer_thread.start()
    applier_thread.start()

    print(f"\nStarting Flask server on http://{FLASK_HOST}:{FLASK_PORT}")
    print("Send POST requests to /generate with JSON body: {'prompt': 'your prompt here'}")
    print("GET /health for health check.")
    
    # 5. Start Flask server (blocking call)
    # Use 'threaded=True' explicitly if needed, though it's often default.
    # 'debug=False' is recommended for stability unless actively debugging Flask itself.
    try:
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True) 
    except Exception as e:
         print(f"FATAL: Failed to start Flask server: {e}")
         exit(1)

    # Code after app.run() executes only when the server stops (e.g., Ctrl+C)
    print("Inference server stopped.")
