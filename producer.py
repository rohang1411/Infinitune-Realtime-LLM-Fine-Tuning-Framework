import hashlib
import argparse
from kafka import KafkaProducer
from datasets import load_dataset
import json
import time
import yaml

from utils.stream_filter import StreamFilter
from collections import deque

def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _log(msg):
    print(f"[{_ts()}][PRODUCER] {msg}", flush=True)

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Track delivery failures globally so the main loop can detect them
_delivery_errors = []

def _on_send_error(excp):
    """Callback fired when a message fails to be delivered to Kafka."""
    _delivery_errors.append(excp)
    _log(f"DELIVERY FAILED: {excp}")

def create_producer(config):
    kafka_cfg = config['kafka']
    # NOTE: Do NOT set api_version explicitly — let kafka-python auto-negotiate
    # with the broker.  Pinning api_version=(0,10) causes silent delivery
    # failures on modern Kafka 3.x+ brokers.
    _log("Creating KafkaProducer (api_version=auto-detect)...")
    return KafkaProducer(
        bootstrap_servers=kafka_cfg['bootstrap_servers'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: k.encode('utf-8')  # expects key as a string
    )

def compute_hash(text):
    # Compute a stable hash (SHA-256) and return its hexadecimal digest
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def generate_training_examples(config):
    """
    Load dataset and yield standardized examples with 'input' and 'target' keys.
    Uses column_mapping and label_map from config to transform raw data.
    """
    dataset_cfg = config['dataset']
    col_map = dataset_cfg['column_mapping']
    input_col = col_map['input_col']
    target_col = col_map['target_col']
    label_map = dataset_cfg.get('label_map')

    # Build dataset loading kwargs
    ds_kwargs = {"path": dataset_cfg['name'], "split": dataset_cfg['split']}
    if dataset_cfg.get('config_name'):
        ds_kwargs["name"] = dataset_cfg['config_name']
    if dataset_cfg.get('data_files'):
        ds_kwargs["data_files"] = dataset_cfg.get('data_files')

    dataset = load_dataset(**ds_kwargs)

    # Shuffle if configured (critical for sorted datasets like IMDb where
    # all label-0 examples come before label-1, starving the model of diversity).
    if dataset_cfg.get('shuffle', False):
        shuffle_seed = dataset_cfg.get('shuffle_seed')
        dataset = dataset.shuffle(seed=shuffle_seed)
        _log(f"Dataset shuffled (seed={shuffle_seed}).")

    _log(f"Loaded dataset: {dataset_cfg['name']} split={dataset_cfg['split']} ({len(dataset)} examples)")
    _log(f"Column mapping: input_col='{input_col}', target_col='{target_col}', label_map={'enabled' if label_map is not None else 'disabled'}")

    for example in dataset:
        input_val = str(example[input_col])
        target_val = example[target_col]

        # Apply label mapping if configured (handles both int and str keys from YAML)
        if label_map is not None:
            if target_val in label_map:
                target_val = label_map[target_val]
            elif str(target_val) in label_map:
                target_val = label_map[str(target_val)]

        target_val = str(target_val)

        yield {
            "input": input_val,
            "target": target_val,
        }

def clear_kafka_topic(kafka_cfg, topic):
    """
    Delete all existing messages in the topic up to the latest offset.
    This ensures that restarted runs do not consume stale data left in the topic
    by previous interrupted runs.
    """
    from kafka import KafkaAdminClient, KafkaConsumer, TopicPartition
    try:
        # Import RecordsToDelete locally to avoid issues on older kafka-python versions if any
        from kafka.admin import RecordsToDelete
    except ImportError:
        _log(f"Warning: Installed kafka-python version does not support delete_records. Skipping topic clear.")
        return

    try:
        _log(f"Checking for stale messages in topic '{topic}' to clear...")
        consumer = KafkaConsumer(bootstrap_servers=kafka_cfg['bootstrap_servers'])
        partitions = consumer.partitions_for_topic(topic)
        if not partitions:
            _log("Topic does not exist yet; nothing to clear.")
            return

        tps = [TopicPartition(topic, p) for p in partitions]
        end_offsets = consumer.end_offsets(tps)

        records_to_delete = {}
        total_cleared = 0
        for tp, offset in end_offsets.items():
            if offset > 0:
                records_to_delete[tp] = RecordsToDelete(offset)
                total_cleared += offset

        if records_to_delete:
            admin = KafkaAdminClient(bootstrap_servers=kafka_cfg['bootstrap_servers'])
            admin.delete_records(records_to_delete)
            _log(f"Successfully truncated topic. Cleared {total_cleared} stale messages.")
        else:
            _log("Topic is already empty.")
    except Exception as e:
        _log(f"Warning: Failed to clear old messages from topic: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InfiniTune Data Producer")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    _log(f"Loaded config: {args.config}")
    stream_filter = StreamFilter(config, log_fn=_log)
    
    topic_name = config['kafka']['training_topic']
    # Clear old messages from previous interrupted runs so we start fresh
    clear_kafka_topic(config['kafka'], topic_name)

    producer = create_producer(config)
    send_interval = config['kafka'].get('producer_send_interval', 0.1)
    hash_column = config['preprocessing'].get('hash_column', 'input')

    _log(f"Kafka bootstrap_servers={config['kafka']['bootstrap_servers']}, topic='{topic_name}', send_interval={send_interval}s")
    _log(f"Hashing standardized column '{hash_column}' for Kafka message keys (dedup/log-compaction friendly).")

    # Verify first message is actually delivered before streaming
    _log("Sending verification message to confirm broker connectivity...")
    verify_future = producer.send(topic_name, key="__verify__", value={"_verify": True})
    try:
        record_metadata = verify_future.get(timeout=10)  # block until delivered
        _log(f"Verification OK: topic={record_metadata.topic}, partition={record_metadata.partition}, offset={record_metadata.offset}")
    except Exception as e:
        _log(f"FATAL: Verification message failed — Kafka broker is unreachable or topic is misconfigured: {e}")
        producer.close()
        exit(1)

    sent_count = 0
    confirmed_count = 0
    last_progress_time = time.time()
    progress_every_s = 5.0

    # Filtering telemetry (non-blocking)
    total_records = 0
    dropped_records = 0
    dropped_examples_printed = 0
    dropped_examples_max_to_print = 3
    recent_drop_reasons = deque(maxlen=10)
    last_filter_telemetry_time = time.time()
    filter_telemetry_every_n = 1000
    filter_telemetry_every_s = 60.0

    _log(f"Starting to stream samples to topic '{topic_name}'...")
    for example in generate_training_examples(config):
        total_records += 1

        extracted_text = str(example.get(hash_column, ""))
        ok, reason = stream_filter.validate(raw_record=example, extracted_text=extracted_text)
        if not ok:
            dropped_records += 1
            if reason:
                recent_drop_reasons.append(reason)

            # Print first 3 dropped examples (temporary debugging aid)
            if dropped_examples_printed < dropped_examples_max_to_print:
                # preview = extracted_text.replace("\n", "\\n")
                # if len(preview) > 180:
                #     preview = preview[:180] + "..."
                # _log(f"FILTER DROP EXAMPLE #{dropped_examples_printed + 1}: reason='{reason}' text_preview='{preview}'")
                _log(f"FILTER DROP EXAMPLE #{dropped_examples_printed + 1}: reason='{reason}'")
                dropped_examples_printed += 1

            now = time.time()
            if (
                total_records % filter_telemetry_every_n == 0
                or now - last_filter_telemetry_time >= filter_telemetry_every_s
            ):
                drop_rate = (dropped_records / float(total_records)) * 100.0 if total_records > 0 else 0.0
                common_reasons = ", ".join(list(recent_drop_reasons)) if recent_drop_reasons else "n/a"
                _log(
                    f"FILTER STATS: Ingested={total_records} | Dropped={dropped_records} | "
                    f"Drop Rate={drop_rate:.2f}% | RecentReasons=[{common_reasons}]"
                )
                last_filter_telemetry_time = now
            continue

        key = compute_hash(example[hash_column])
        future = producer.send(
            topic=topic_name,
            key=key,   # this is now a stable hexadecimal string
            value=example
        )
        # Attach error callback so delivery failures are never silent
        future.add_errback(_on_send_error)
        sent_count += 1

        now = time.time()
        if now - last_progress_time >= progress_every_s:
            _log(f"Progress: sent_count={sent_count}, delivery_errors={len(_delivery_errors)}")
            last_progress_time = now
            # Bail out if we're accumulating delivery errors
            if len(_delivery_errors) > 10:
                _log("FATAL: Too many delivery errors — aborting. Check broker logs.")
                break
        time.sleep(send_interval)  # Control data generation speed

    # Signal end-of-stream so the trainer can stop cleanly when no more data is coming.
    try:
        _log("Sending end-of-stream marker...")
        eof_future = producer.send(topic_name, key="__eof__", value={"_eof": True})
        eof_future.get(timeout=10)
        _log("End-of-stream marker delivered.")
    except Exception as e:
        _log(f"WARNING: Failed to send end-of-stream marker: {e}")

    _log(f"Stream finished. Total sent: {sent_count}. Flushing...")
    producer.flush()
    _log("Flush complete. Closing producer.")
    producer.close()
