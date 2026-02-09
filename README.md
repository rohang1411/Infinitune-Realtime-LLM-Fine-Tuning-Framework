# Online LLM Fine-tuning Framework with Kafka and QLoRA

A distributed system for continuous fine-tuning of Large Language Models using Kafka streams and QLoRA (Quantized Low-Rank Adaptation).

## Architecture overview 
```mermaid
graph LR
    A[Data Generator] -->|Training Data| B[Kafka]
    B --> C[QLoRA Trainer]
    C -->|LoRA Weights| D[Kafka]
    D --> E[Inference Server]
```

## How to install and run Kafka on M series Mac
```bash
brew install kafka

brew install zookeeper
zookeeper-server-start /opt/homebrew/etc/kafka/zookeeper.properties


# Start Kafka (KRaft mode - no Zookeeper needed)
# Note: Zookeeper is no longer required or installed by default in newer Kafka versions
kafka-server-start /opt/homebrew/etc/kafka/server.properties

# test kafka (in a new terminal)
kafka-topics --bootstrap-server localhost:9092 --list


nano ~/.zshrc 
export PATH="/opt/homebrew/Cellar/kafka/3.9.0/libexec/bin:$PATH"
source ~/.zshrc

# test kafka
kafka-topics.sh --bootstrap-server localhost:9092 --list
```

## How to install and run Kafka on Windows

### **Prerequisites**
1. **Install Java Development Kit (JDK):**
   - Download and install the latest JDK from Oracle or OpenJDK.
   - Set the `JAVA_HOME` environment variable to point to your JDK installation directory.

2. **Download Kafka:**
   - Visit the [Apache Kafka Downloads page](https://kafka.apache.org/downloads) and download the latest binary `.tgz` file.
   - Extract the downloaded file to a directory (e.g., `C:\kafka`).

---

### **Step-by-Step Installation**
1. **Set Up Kafka Directory:**
   - After extraction, rename the folder to `kafka` for simplicity and place it in `C:\`.

2. **Configure Zookeeper and Kafka:**
   - Kafka requires Zookeeper to run. Configuration files are located in the `config` directory:
     - `zookeeper.properties`: Default settings are sufficient for basic setups.
     - `server.properties`: Update `log.dirs` to a directory where Kafka will store logs (e.g., `C:/kafka/logs`).

3. **Start Zookeeper:**
   - Open a Command Prompt, navigate to the Kafka directory (`C:\kafka`), and run:
     ```bash
     .\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
     ```
   - This starts Zookeeper, which is required for Kafka.

4. **Start Kafka Server:**
   - Open another Command Prompt, navigate to the Kafka directory, and run:
     ```bash
     .\bin\windows\kafka-server-start.bat .\config\server.properties
     ```

5. **Create a Topic:**
   - In a new Command Prompt window, create a topic named `test`:
     ```bash
     .\bin\windows\kafka-topics.bat --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic test
     ```

6. **Produce Messages:**
   - Start a producer to send messages to the topic:
     ```bash
     .\bin\windows\kafka-console-producer.bat --broker-list localhost:9092 --topic test
     ```
   - Type messages into the console and press Enter to send them.

7. **Consume Messages:**
   - Start a consumer to read messages from the topic:
     ```bash
     .\bin\windows\kafka-console-consumer.bat --bootstrap-server localhost:9092 --topic test --from-beginning
     ```

---

### **Troubleshooting Tips**
- If you encounter issues with deprecated tools like `wmic` on Windows 11, consider using Docker or Windows Subsystem for Linux (WSL 2) for better compatibility.
- Ensure that your environment variables (`JAVA_HOME`, `PATH`) are correctly set.



```bash
# Data Generator
python producer.py

# Trainer (in separate terminal)
python trainer.py

# Inference Server (in separate terminal)
python inference.py
```




-------------------------
Running command WITH ARGGS

# Terminal 1 — Start the Producer
python producer.py --config configs/imdb_config.yaml

# Terminal 2 — Start the Trainer
python trainer.py --config configs/imdb_config.yaml

# Terminal 3 — Start the Inference Server
python inference.py --config configs/imdb_config.yaml

# Terminal 4 — Test the API
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Review: This movie was absolutely terrible.\nSentiment:"}'

# Health check
curl http://localhost:5000/health