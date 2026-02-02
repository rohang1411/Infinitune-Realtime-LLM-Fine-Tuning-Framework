import hashlib
from kafka import KafkaProducer
from datasets import load_dataset
import json
import time

def create_producer():
    return KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        # Do not force `api_version`. Let kafka-python auto-detect the broker
        # supported API versions to avoid UnsupportedVersionException.
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: k.encode('utf-8')  # expects key as a string
    )

def compute_hash(text):
    # Compute a stable hash (SHA-256) and return its hexadecimal digest
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def generate_training_examples():
    # Load the IMDb dataset from the Hugging Face hub, using the "train" split
    dataset = load_dataset("imdb", split="train")
    print("loaded dataset")
    for example in dataset:
        yield {
            "review": example["text"],
            "sentiment": "positive" if example["label"] == 1 else "negative",
        }

if __name__ == "__main__":
    producer = create_producer()
    topic_name = "training-data"
    
    print("generating samples")
    for example in generate_training_examples():
        key = compute_hash(example["review"])
        producer.send(
            topic=topic_name,
            key=key,   # this is now a stable hexadecimal string
            value=example
        )
        print("sent sample")
        time.sleep(0.1)  # Control data generation speed

    producer.flush()
    producer.close()
