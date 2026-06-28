# MLOps Orchestration Pipeline

An end-to-end machine learning platform running entirely on Docker Compose — from data generation and model training to serving, monitoring, and automated retraining.

---

## What this project does

| Layer | Tool | Role |
|---|---|---|
| **Orchestration** | Apache Airflow | Schedules nightly retraining runs |
| **Experiment tracking** | MLflow | Logs params, metrics, and model artifacts |
| **Model serving** | FastAPI | Serves predictions via a REST API |
| **Monitoring** | Prometheus + Grafana | Scrapes and visualises API metrics |
| **CI/CD** | GitHub Actions | Trains, tests, and builds the image on every push |

The model itself is a **RandomForest classifier** predicting customer churn from four behavioural features. It is intentionally simple — the point is the pipeline, not the model.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Docker Compose                        │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Airflow  (scheduler + webserver :8080)              │ │
│  │  DAG: generate_data ──► train_model                 │ │
│  │         │                      │                     │ │
│  │         ▼                      ▼                     │ │
│  │   data/churn.csv        app/model/model.pkl          │ │
│  └─────────────────────────────────────────────────────┘ │
│                             │                             │
│                    logs run to MLflow                     │
│  ┌──────────────┐           │                            │ │
│  │  MLflow      │◄──────────┘                            │ │
│  │  :5000       │  params, metrics, artifacts            │ │
│  └──────────────┘                                        │ │
│                                                           │
│  ┌──────────────┐   POST /predict    ┌────────────────┐  │
│  │  FastAPI     │◄───────────────────│  curl / client │  │
│  │  :8000       │                    └────────────────┘  │
│  │  GET /metrics│                                        │ │
│  └──────┬───────┘                                        │ │
│         │ scrape every 15s                               │ │
│  ┌──────▼───────┐   query    ┌──────────────┐           │ │
│  │  Prometheus  │◄───────────│   Grafana    │           │ │
│  │  :9090       │            │   :3000      │           │ │
│  └──────────────┘            └──────────────┘           │ │
└──────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose v2)
- 4 GB RAM allocated to Docker (Airflow is memory-hungry)

---

## Running locally

### 1. Clone and start

```bash
git clone <your-repo-url>
cd mlops-orchestration-pipeline
docker compose up --build
```

Wait ~60 seconds for all services to initialise (Airflow runs DB migrations on first boot).

### 2. Generate data and train the model (first time only)

Open a new terminal and run:

```bash
docker compose exec api bash -c "
  pip install scikit-learn pandas mlflow numpy &&
  python scripts/generate_data.py &&
  MLFLOW_TRACKING_URI=http://mlflow:5000 python scripts/train.py
"
```

Or trigger the DAG manually in Airflow (see step 4).

### 3. Hit the prediction API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_spend": 89.99,
    "tenure_months": 3,
    "support_tickets": 7,
    "login_frequency": 2
  }'
```

Expected response:

```json
{
  "churn_probability": 0.8341,
  "will_churn": true,
  "model_version": "1.0"
}
```

### 4. Airflow UI

Open [http://localhost:8080](http://localhost:8080) → login `admin / admin`

Enable the **churn_model_training** DAG and click **Trigger DAG** to run it immediately.

### 5. MLflow UI

Open [http://localhost:5000](http://localhost:5000) to see experiment runs, metrics, and registered models.

### 6. Grafana dashboards

Open [http://localhost:3000](http://localhost:3000) → login `admin / admin`

Navigate to **Dashboards → Churn Prediction API** to see live request rates and latency percentiles.

---

## Project layout

```
.
├── scripts/
│   ├── generate_data.py    # Synthesise churn dataset → data/churn.csv
│   └── train.py            # Train RandomForest, log to MLflow, save model
├── dags/
│   └── training_dag.py     # Airflow DAG: generate → train (nightly)
├── app/
│   ├── main.py             # FastAPI service: /predict, /metrics, /health
│   └── model/              # Saved model.pkl (gitignored, built in CI)
├── monitoring/
│   ├── prometheus.yml              # Scrape config
│   ├── dashboard.json              # Grafana dashboard (auto-provisioned)
│   ├── grafana-datasource.yml      # Prometheus datasource
│   └── grafana-dashboard-provider.yml
├── tests/
│   └── test_api.py         # pytest suite (no running server needed)
├── .github/workflows/
│   └── ci.yml              # GitHub Actions: test + Docker build
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Running tests locally

```bash
pip install -r requirements.txt
python scripts/generate_data.py
MODEL_OUTPUT_PATH=app/model/model.pkl python scripts/train.py
pytest tests/ -v
```

---

## Why each tool?

**Airflow** — industry-standard DAG orchestrator. Even for a two-task pipeline it demonstrates how retraining jobs are scheduled and monitored in production.

**MLflow** — tracks every training run (hyperparameters, metrics, artefacts) so you can compare runs and roll back to a known-good model.

**FastAPI** — async Python web framework with automatic OpenAPI docs. Pydantic validates inputs so bad payloads are rejected before they hit the model.

**Prometheus + Grafana** — the de-facto open-source observability stack. The `/metrics` endpoint follows the OpenMetrics standard, making the API compatible with any Prometheus-based monitoring system.
