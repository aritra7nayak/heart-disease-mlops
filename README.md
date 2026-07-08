# Heart Disease MLOps — BITS Pilani AIMLCZG523

[![CI/CD](https://github.com/your-org/heart-disease-mlops/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/your-org/heart-disease-mlops/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

End-to-end MLOps project: data acquisition → EDA → model training (with MLflow) → FastAPI serving → Docker containerisation → Kubernetes deployment → Prometheus/Grafana monitoring.

---

## Project Structure

```
heart-disease-mlops/
├── data/
│   ├── download_data.py       # Dataset download & cleaning script
│   └── heart_disease_cleaned.csv
├── notebooks/
│   └── eda.py                 # EDA script (produces all plots)
├── src/
│   ├── preprocessing.py       # Feature engineering & sklearn pipeline
│   ├── train.py               # Training + MLflow experiment tracking
│   ├── inference.py           # Standalone inference helper
│   └── app.py                 # FastAPI serving application
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py  # 20 unit tests
│   ├── test_model.py          # 15 unit tests
│   └── test_api.py            # 23 unit tests (58 total)
├── models/                    # Saved model artefacts (auto-generated)
├── mlruns/                    # MLflow experiment tracking (auto-generated)
├── screenshots/               # EDA visualisations (auto-generated)
├── k8s/
│   ├── deployment.yaml        # Kubernetes Deployment
│   └── service.yaml           # Kubernetes Service + Ingress
├── helm/
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
│       └── deployment.yaml
├── monitoring/
│   ├── prometheus.yml
│   └── docker-compose.yml     # Full monitoring stack
├── .github/workflows/
│   └── ci_cd.yml              # GitHub Actions CI/CD (5 jobs)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start (Local)

### 1. Clone & install

```bash
git clone https://github.com/your-org/heart-disease-mlops.git
cd heart-disease-mlops
pip install -r requirements.txt
```

### 2. Download dataset

```bash
python data/download_data.py
```

### 3. EDA

```bash
python notebooks/eda.py
# Plots saved to screenshots/
```

### 4. Train models (with MLflow tracking)

```bash
MLFLOW_ALLOW_FILE_STORE=true python src/train.py
# Best model: XGBoost  ROC-AUC: 0.9805  Accuracy: 0.9180
```

### 5. Run unit tests

```bash
pytest tests/ -v
# 58 passed
```

### 6. Start the API locally

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age":54,"sex":1,"cp":0,"trestbps":130,"chol":245,
    "fbs":0,"restecg":0,"thalach":150,"exang":0,
    "oldpeak":1.4,"slope":1,"ca":0,"thal":2
  }'

# Metrics (Prometheus format)
curl http://localhost:8000/metrics
```

---

## Docker

```bash
# Build
docker build -t heart-disease-api:1.0.0 .

# Run
docker run -d -p 8000:8000 --name heart-api heart-disease-api:1.0.0

# Test
curl http://localhost:8000/health

# Stop
docker stop heart-api && docker rm heart-api
```

---

## Kubernetes (Minikube)

```bash
# Start Minikube
minikube start

# Apply manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check status
kubectl get pods
kubectl get svc

# Get URL
minikube service heart-disease-api-service --url

# Or use Helm
helm install heart-disease ./helm
```

---

## Monitoring Stack (Docker Compose)

```bash
cd monitoring
docker-compose up -d

# Services:
#   API:        http://localhost:8000
#   Prometheus: http://localhost:9090
#   Grafana:    http://localhost:3000 (admin/admin)
```

## Dataset

- **Source:** UCI Machine Learning Repository — Heart Disease Cleveland Dataset  
- **Rows:** 303 patients  
- **Features:** 13 clinical features (age, sex, chest pain type, BP, cholesterol, etc.)  
- **Target:** Binary (0=No disease, 1=Disease present)

---

## API Endpoints

| Method | Endpoint     | Description                         |
|--------|-------------|-------------------------------------|
| GET    | /           | Service info                        |
| GET    | /health     | Health check                        |
| POST   | /predict    | Heart disease prediction            |
| GET    | /metrics    | Prometheus-format metrics           |
| GET    | /model-info | Model metadata & feature list       |
| GET    | /docs       | Swagger UI (auto-generated)         |

---
