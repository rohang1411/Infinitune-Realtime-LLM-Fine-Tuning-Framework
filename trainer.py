import time
import io
import json
import torch
from kafka import KafkaProducer, KafkaConsumer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

# --------------------------------------------------
# LoRAProducer: sends adapter weights via Kafka
# --------------------------------------------------
class LoRAProducer:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=["localhost:9092"],
            value_serializer=lambda v: self.serialize_tensor(v),
            key_serializer=lambda k: b"lora_weights"
        )

    def serialize_tensor(self, tensor):
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()
    
    def send_weights(self, adapter_weights):
        print("Sending weights update...")
        for name, param in adapter_weights.items():
            self.producer.send(
                topic="lora-updates",
                key=name.encode("utf-8") if isinstance(name, str) else name,
                value=param
            )
        self.producer.flush()

# --------------------------------------------------
# Manual training loop function
# --------------------------------------------------
def train_model(model_name, lora_config):
    print("Starting preparation for training")

    # Choose device. (On macs using MPS or defaulting to CPU/GPU.)
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    # device="cuda" if torch.cuda.is_available() else "cpu",

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # For Apple Silicon GPUs
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config,
        # device_map={"" : device}
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure there is a pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loaded model; applying PEFT and LoRA adapter configuration...")

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Set the model to training mode.
    model.train()
    
    # Set up Kafka consumer to stream training data.
    # It is assumed that each message contains an "instruction" and an "output".
    consumer = KafkaConsumer(
        "training-data",
        bootstrap_servers=["localhost:9092"],
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        group_id="trainer-group",
        auto_offset_reset="earliest"
    )
    
    # Training hyperparameters
    training_args = TrainingArguments(
        output_dir="./qlora-output",
        per_device_train_batch_size=4,      # mini-batch size before gradient accumulation
        gradient_accumulation_steps=2,        # number of mini-batches to accumulate before an update
        learning_rate=2e-4,
        logging_steps=1,
        max_steps=1000,                       # count of optimizer updates (not raw iterations)
        fp16=True
    )
    
    # Create an optimizer (here we update all model parameters, but you might want to restrict this)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    
    # DataCollatorForLanguageModeling will help format our batches (with labels equal to inputs for causal LM)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Instantiate our Kafka producer for LoRA updates.
    lora_producer = LoRAProducer()
    
    # Set up timing for sending weights
    last_send_time = time.time()
    
    # Counters for gradient accumulation and optimization steps
    optimization_step = 0
    grad_accum_counter = 0
    
    # Create an iterator from the Kafka consumer to stream new messages.
    message_iterator = iter(consumer)
    print("Starting manual training loop...")
    
    while optimization_step < training_args.max_steps:
        batch_samples = []
        # Assemble a mini-batch of raw examples.
        while len(batch_samples) < training_args.per_device_train_batch_size:
            try:
                message = next(message_iterator)
            except StopIteration:
                # In case the consumer temporarily runs out (or use consumer.poll() if desired)
                time.sleep(0.1)
                continue
    
            # Process the Kafka message to extract text (you may adjust how you build this string)
            # sample_text = f"{message.value['instruction']}\n{message.value['output']}"
            sample_text = f"Review: {message.value['review']}\nSentiment: {message.value['sentiment']}"
            batch_samples.append({"text": sample_text})
    
        # Tokenize each sample (using a max_length of 512 here; adjust if needed)
        tokenized_samples = []
        for sample in batch_samples:
            tokenized = tokenizer(
                sample["text"],
                truncation=True,
                max_length=512,
                padding="longest",
                # return_tensors="pt"
            )
            tokenized_samples.append(tokenized)
    
        # Use the data collator to merge individual tokenizations into a single batch dict.
        batch = data_collator(tokenized_samples)
        # Move all inputs to the target device.
        batch = {k: v.to(device) for k, v in batch.items()}
    
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        # Scale loss appropriately for gradient accumulation.
        loss = loss / training_args.gradient_accumulation_steps
        loss.backward()
        grad_accum_counter += 1
    
        # Once enough mini-batches are accumulated, update the optimizer.
        if grad_accum_counter % training_args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            optimization_step += 1
    
            # Optionally, print loss every logging_steps updates.
            if optimization_step % training_args.logging_steps == 0:
            # Multiply back our loss to get an approximated per-update loss
                print(f"Step {optimization_step}: loss = {loss.item() * training_args.gradient_accumulation_steps:.4f}")
                # lora_producer.send_weights(get_peft_model_state_dict(model))
    
        # Every 60 seconds, send the current LoRA adapter weights.
        if time.time() - last_send_time >= 60:
            print(f"60 seconds elapsed at optimization step {optimization_step}. Sending adapter weights...")
            lora_producer.send_weights(get_peft_model_state_dict(model))
            last_send_time = time.time()
    
    # After training is finished, send one final weights update.
    print("Training complete. Sending final adapter weights.")
    lora_producer.send_weights(get_peft_model_state_dict(model))

# --------------------------------------------------
# Main entry point
# --------------------------------------------------
if __name__ == "__main__":
    # Configuration for the model and LoRA adapter
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["self_attn.q_proj", "self_attn.v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    train_model(model_name, lora_config)
