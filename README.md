# Cyberattack Investigation Support System (HDBSCAN-Based)

## 📌 Project Overview
This project focuses on developing a **Cyberattack Investigation Support System** based on **web server log analysis** using an **unsupervised learning approach**. The system is designed to help security analysts efficiently investigate large-scale web logs by clustering access patterns and isolating anomalous behavior that may indicate cyber attacks.

Unlike traditional signature-based or real-time detection systems, this project emphasizes **post-incident analysis and investigation support**, making it suitable for forensic analysis, threat hunting, and security auditing.

---

## 🎯 Problem Statement
Modern web applications generate **millions of log entries daily**, containing highly diverse and semi-structured request patterns. Manual log analysis is:
- Time-consuming
- Error-prone
- Not scalable

Security analysts need a method to:
- Automatically group normal access behavior
- Isolate low-frequency but high-risk patterns
- Support investigation without relying on labeled data

---

## 💡 Proposed Solution
This project proposes an **HDBSCAN-based clustering pipeline** to analyze web server logs and support cyberattack investigation.

Core ideas:
- Treat **URL access patterns** as behavioral indicators
- Use **transformer-based embeddings** to capture semantic similarity between URLs
- Apply **density-based clustering (HDBSCAN)** to automatically group normal behavior and identify anomalies (noise)

Noise points produced by HDBSCAN are treated as **high-priority candidates for security investigation**.

---

## 🏗️ System Pipeline
1. **Log Preprocessing**
   - Parsing raw web server logs (NGINX format)
   - Filtering bot traffic
   - URL decoding and normalization
   - Masking dynamic parameters (e.g., numbers, IDs)

2. **Pattern Extraction**
   - Extracting normalized URL patterns
   - Deduplicating requests to obtain unique patterns

3. **Scalable Processing**
   - Chunking large datasets to support large-scale log analysis

4. **Feature Representation**
   - Transformer-based URL embedding (semantic representation)

5. **Clustering**
   - Density-based clustering using **HDBSCAN**
   - Automatic detection of clusters and noise (outliers)

6. **Post-Processing**
   - Merging cluster results across chunks
   - Generating investigation reports

---

## 🧠 Why HDBSCAN?
- No need to predefine number of clusters
- Handles varying density distributions
- Naturally detects outliers (noise)
- Suitable for unknown or evolving attack patterns

This makes HDBSCAN well-suited for **cybersecurity investigation scenarios** where attack signatures are incomplete or unknown.

---

## 🛠️ Tech Stack
- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, HDBSCAN
- **Embedding**: Sentence Transformers
- **Data Source**: Public web server log dataset (Harvard Dataverse)

---

## 🔐 Use Cases
- Cybersecurity investigation
- Log-based threat hunting
- Security auditing
- Digital forensics support

---

## ⚠️ Disclaimer
This project is intended for **research and investigation support purposes**. It is not designed as a real-time intrusion detection or prevention system.

---

## 👤 Author
**Naufal Alif Rahmania Akmal**  
Cyber Security Enthusiast