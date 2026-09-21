# RealTimeGuard 🛡️
### Real-Time Transaction Fraud Detection Engine

A low-latency fraud detection pipeline that simulates a live transaction stream, scores each transaction with a machine learning model in real time, and surfaces results on a live monitoring dashboard — built to demonstrate production-grade ML system design, not just model training.

---

## Why this project exists

Most student ML projects stop at "I trained a model and got 95% accuracy in a notebook." That doesn't reflect how fraud detection actually works in production, where a bank or payment processor needs to:
- Score thousands of transactions **per second**
- Respond in **milliseconds**, not seconds
- Handle **highly imbalanced data** (fraud is <1% of all transactions)
- Stay observable — engineers need live visibility into throughput, latency, and system health

RealTimeGuard is built to mirror that reality: a streaming pipeline, a dedicated model-serving layer, and a live dashboard with real performance numbers — not just an offline accuracy score.

---

## Architecture

```
┌─────────────────┐     ┌───────────────┐     ┌────────────────────┐     ┌───────────────┐
│  Transaction      │────▶│  Redis Stream  │────▶│  Node.js Consumer   │────▶│  FastAPI Model │
│  Generator (sim)  │     │  (message bus) │     │  (backpressure,     │────▶│  Server        │
└─────────────────┘     └───────────────┘     │   retries, routing) │     │  (LightGBM)     │
                                                └────────────────────┘     └───────────────┘
                                                          │
                                                          ▼
                                                ┌────────────────────┐
                                                │  MongoDB (results)  │
                                                └────────────────────┘
                                                          │
                                                          ▼
                                                ┌────────────────────┐
                                                │  React Dashboard    │
                                                │  (Socket.IO live    │
                                                │   feed + metrics)   │
                                                └────────────────────┘
```

---

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| ML Model | LightGBM | Fast inference (sub-millisecond), handles imbalanced tabular data natively |
| Streaming | Redis Streams | Buffers and decouples the transaction feed from processing, so bursts don't overwhelm the system |
| Model serving | FastAPI | Lightweight Python service, loads model once at startup (not per-request) |
| Backend | Node.js + Express | Consumer service handling stream reads, retries, and API orchestration |
| Real-time updates | Socket.IO | Pushes live results to the dashboard instead of polling |
| Frontend | React + Recharts | Live transaction feed, throughput/latency graphs |
| Database | MongoDB | Stores processed transaction results |
| Load testing | k6 | Simulates realistic transaction load to measure system limits |

---

## Project status

- [x] **Week 1 — Data + Model**: feature engineering, LightGBM training, inference latency benchmarking
- [ ] **Week 2 — Streaming pipeline**: Redis Streams + FastAPI + Node.js consumer service
- [ ] **Week 3 — Dashboard**: React + Socket.IO live monitoring UI
- [ ] **Week 4 — Load testing + deployment**: k6 benchmarks, cloud deployment, demo video

*(Update this checklist and the results table below as each phase is completed.)*

---

## Model performance

> Fill in with real numbers once trained on the actual PaySim/IEEE-CIS dataset — synthetic-data placeholder results should never be reported here.

| Metric | Value |
|---|---|
| ROC-AUC | _pending real dataset_ |
| Fraud Precision | _pending real dataset_ |
| Fraud Recall | _pending real dataset_ |
| Mean inference latency | _pending real dataset_ |
| p99 inference latency | _pending real dataset_ |

## System performance (Week 4)

| Metric | Value |
|---|---|
| Peak sustained throughput | _pending load test_ |
| p99 end-to-end latency at peak load | _pending load test_ |

---

## Key design decisions (interview talking points)

**Why LightGBM instead of a neural network?**
Tabular transaction data (amounts, balances, categorical types) doesn't benefit much from deep learning, and LightGBM offers inference speeds that are critical for a sub-50ms latency target — the accuracy tradeoff is negligible for this data type.

**Why Redis Streams instead of a direct API call?**
A direct call from generator to model server couples the two tightly — if the model server slows down, the generator blocks too. Redis Streams decouples them: transactions queue up during a slowdown instead of failing outright, and consumer groups allow safe replay if a service crashes mid-processing.

**Why not oversample fraud with SMOTE?**
Class weighting (`scale_pos_weight`) was used instead of synthetic oversampling. SMOTE-generated fraud examples in tabular financial data can create unrealistic balance/amount combinations, so weighting the loss function directly is more faithful to the true data distribution.

**What would change for real production?**
- Kafka instead of Redis Streams for stronger durability guarantees at massive scale
- Model versioning + shadow deployment for safely testing new models against live traffic
- A feature store to keep training/serving feature computation consistent

---

## Running locally

```bash
# 1. Install dependencies
pip install lightgbm scikit-learn pandas numpy imbalanced-learn joblib --break-system-packages

# 2. Generate or provide data
cd scripts
python3 generate_data.py     # synthetic placeholder data
# OR place the real PaySim CSV at data/raw_transactions.csv

# 3. Feature engineering + training
python3 feature_engineering.py
python3 train_model.py
```

*(Streaming, dashboard, and deployment instructions will be added as those phases are built.)*

---

## Dataset

[PaySim1 — Synthetic Financ ial Dataset For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1) (Kaggle)

---

## Author

Anjali Desai — B.Tech CSE (AI/ML), SRM University AP
[GitHub: ADesai2708](https://github.com/ADesai2708)
