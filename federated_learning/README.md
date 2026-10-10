# Federated Learning — 4-Site Non-IID BERT Training

A production-ready federated learning system with 6 Docker containers
communicating over a private network. Built by Tsegai Yhdego, PhD.

---

## Architecture

All containers communicate over an isolated Docker network `fl_network`.
No raw data ever leaves a site container — only model weights are shared.

---

## Quick Start

### Prerequisites
- Docker Desktop installed
- 8GB RAM minimum
- Internet connection (first run downloads DistilBERT ~270MB)

### 1. Clone the repository
```bash
git clone https://github.com/Tsegai20/ai-skills-bootcamp.git
cd ai-skills-bootcamp/federated_learning
```

### 2. Prepare data
```bash
cd data
pip install datasets pandas numpy
python prepare_data.py
cd ..
```

### 3. Create required directories
```bash
mkdir -p mlflow
```

### 4. Build and start all containers
```bash
docker compose build
docker compose up -d
```

### 5. Wait for all containers to be healthy
```bash
docker compose ps
```
All services should show `healthy` status (takes 2-3 minutes on first run).

### 6. Run federated training

**One round:**
```bash
curl -X POST http://localhost:8000/round \
  -H "Content-Type: application/json" \
  -d '{"epochs": 1, "lr": 2e-5}'
```

**Five rounds:**
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"num_rounds": 5, "epochs": 1, "lr": 2e-5}'
```

**Check results:**
```bash
curl http://localhost:8000/history
```

### 7. Open the dashboard
Go to **http://localhost:8501** in your browser.

---

## API Reference

### Site Servers (each site exposes the same API)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check and site status |
| POST | `/train` | Receive global weights, train, return updated weights |

### Aggregator
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/sites` | Poll all 4 sites |
| POST | `/round` | Run one federated round |
| POST | `/train` | Run N rounds |
| GET | `/history` | Full training history |
| GET | `/docs` | Swagger UI |

---

## Results

### 5-Round Convergence (Non-IID, CPU)
| Round | Global Accuracy | Improvement |
|-------|----------------|-------------|
| 1 | 39.8% | +39.8% |
| 2 | 50.5% | +10.7% |
| 3 | 58.2% | +7.7% |
| 4 | 63.7% | +5.5% |
| 5 | 66.7% | +3.0% |

Single-round FedAvg baseline: 33.3%
Centralized training baseline: 84.3%
Non-IID gap vs centralized: 17.6%

---

## Key Concepts

**Non-IID Data** — Each site holds only one news category. This is the
most challenging federated setting, directly mirroring manufacturing
where different sites produce different product types.

**FedAvg** — Federated Averaging weights each site's contribution
proportionally to its data size before averaging model weights.

**Client Drift** — Under extreme Non-IID conditions local models bias
toward their own class. Multi-round training reduces this effect.

**Privacy** — Raw CSV data never leaves the `/data` volume mount.
Only PyTorch model weights travel over the Docker network.

---

## Tech Stack
- **FastAPI** — REST APIs for sites and aggregator
- **DistilBERT** — Lightweight BERT variant (270MB vs 440MB)
- **PyTorch** — Model training with optional GPU acceleration
- **MLflow** — Experiment tracking (SQLite backend)
- **Streamlit** — Real-time training dashboard
- **Docker Compose** — 6-container orchestration with health checks

---

## Research Context

This system demonstrates the federated learning challenges studied in
Tsegai's PhD dissertation on attribute-based cognitive ML for manufacturing
quality control. The Non-IID accuracy gap (17.6%) motivates advanced FL
algorithms including attribute-based knowledge transfer.

**Published:** IEEE T-ASE 2025, ASME JCISE 2024
**Patent:** Filed for privacy-preserving federated learning system

---

## Author

**Tsegai Yhdego, PhD**
Industrial Engineering — Machine Learning and Optimization
Assistant Director, R-SEAT Center, FAMU-FSU
USDOT-Funded Research
