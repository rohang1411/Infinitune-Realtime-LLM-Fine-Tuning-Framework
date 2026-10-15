#!/usr/bin/env bash
# InfiniTune Google Colab Apache Kafka (KRaft mode) startup script.
# Installs headless JRE, downloads Kafka 3.9.0, formats storage, starts broker, and polls readiness.
set -euo pipefail

KAFKA_VERSION="3.9.0"
SCALA_VERSION="2.13"
KAFKA_DIR="/content/kafka_${SCALA_VERSION}-${KAFKA_VERSION}"
KAFKA_DATA="/content/kafka-data"
KAFKA_LOG="/content/kafka.log"

# Check if Kafka broker is already online
if [ -d "${KAFKA_DIR}" ] && "${KAFKA_DIR}/bin/kafka-topics.sh" --bootstrap-server localhost:9092 --list > /dev/null 2>&1; then
    echo "Apache Kafka broker is already ONLINE and responding on localhost:9092."
    exit 0
fi

echo "=== [1/5] Installing OpenJDK 17 JRE Headless ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq openjdk-17-jre-headless

echo "=== [2/5] Downloading Apache Kafka ${KAFKA_VERSION} ==="
if [ ! -d "${KAFKA_DIR}" ]; then
    ARCHIVE_URL="https://archive.apache.org/dist/kafka/${KAFKA_VERSION}/kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"
    FALLBACK_URL="https://downloads.apache.org/kafka/${KAFKA_VERSION}/kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"
    wget -q "${ARCHIVE_URL}" -O /tmp/kafka.tgz || wget -q "${FALLBACK_URL}" -O /tmp/kafka.tgz
    tar -xzf /tmp/kafka.tgz -C /content
    rm -f /tmp/kafka.tgz
fi

echo "=== [3/5] Configuring KRaft and Formatting Storage ==="
mkdir -p "${KAFKA_DATA}"
KRAFT_CONFIG="${KAFKA_DIR}/config/kraft/server.properties"

# Ensure log directory points to KAFKA_DATA
sed -i "s|log.dirs=.*|log.dirs=${KAFKA_DATA}|g" "${KRAFT_CONFIG}"

# Generate cluster ID and format storage
CLUSTER_ID=$("${KAFKA_DIR}/bin/kafka-storage.sh" random-uuid)
"${KAFKA_DIR}/bin/kafka-storage.sh" format -t "${CLUSTER_ID}" -c "${KRAFT_CONFIG}" --ignore-formatted

echo "=== [4/5] Starting Kafka Broker in Background ==="
export KAFKA_HEAP_OPTS="-Xmx1G -Xms1G"
nohup "${KAFKA_DIR}/bin/kafka-server-start.sh" "${KRAFT_CONFIG}" > "${KAFKA_LOG}" 2>&1 &

echo "=== [5/5] Polling Kafka Broker Readiness (max 60s) ==="
READY=0
for i in $(seq 1 60); do
    if "${KAFKA_DIR}/bin/kafka-topics.sh" --bootstrap-server localhost:9092 --list > /dev/null 2>&1; then
        echo "Apache Kafka broker is ONLINE and ready on localhost:9092 (took ${i}s)."
        READY=1
        break
    fi
    sleep 1
done

if [ ${READY} -eq 0 ]; then
    echo "ERROR: Kafka failed to start within 60 seconds. Tail of ${KAFKA_LOG}:"
    tail -n 50 "${KAFKA_LOG}"
    exit 1
fi
