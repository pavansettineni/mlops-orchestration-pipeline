# MLOps Orchestration Pipeline

A small end-to-end ML pipeline I built to practice the full lifecycle of getting a model from raw data into a monitored, automatically retrained service — not just training something in a notebook and leaving it there.

It predicts customer churn from four basic behavioral features. The model itself is intentionally simple (a RandomForest classifier) — the point of this project isn't the model, it's everything around it: scheduling, tracking, serving, monitoring, and CI/CD, wired together the way they'd actually work in production.

---

## What's actually in here

- **Airflow** schedules a nightly job that regenerates training data and retrains the model
- **MLflow** logs every training run — parameters, metrics, the model artifact itself — so nothing gets lost between runs
- **FastAPI** serves the trained model behind a `/predict` endpoint
- **Prometheus + Grafana** watch the API while it's running — request rates, latency, error rates
- **GitHub Actions** runs tests and builds the Docker image on every push

Everything runs locally through Docker Compose. No cloud account needed.

---

## How it fits together

```
Airflow (scheduler + webserver, :8080)
  DAG: generate_data → train_model
         │                  │
         ▼                  ▼
   data/churn.csv     app/model/model.pkl
                            │
                   logged to MLflow (:5000)

FastAPI (:8000)
  POST /predict   ← curl / any client
  GET  /metrics   ← scraped by Prometheus every 15s

Prometheus (:9090) ──→ Grafana (:3000)
```

---

## Running it

**You'll need:** Docker Desktop installed, and ideally 4GB+ RAM given to Docker — Airflow is the heavy part.

**1. Start everything**

```bash
git clone <your-repo-url>
cd mlops-orchestration-pipeline
docker compose up --build
```

Give it about a minute on first boot — Airflow runs its DB migrations before anything else comes up.

**2. Train the model the first time**

In a separate terminal:

```bash
docker compose exec api bash -c "
  pip install scikit-learn pandas mlflow numpy &&
  python scripts/generate_data.py &&
  MLFLOW_TRACKING_URI=http://mlflow:5000 python scripts/train.py
"
```

(Or just trigger the DAG manually from the Airflow UI — see below.)

**3. Try a prediction**

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

```json
{
  "churn_probability": 0.8341,
  "will_churn": true,
  "model_version": "1.0"
}
```

**4. Look around the UIs**

- Airflow → [localhost:8080](http://localhost:8080) (`admin` / `admin`) — enable the `churn_model_training` DAG, trigger it manually if you want to see a run happen live
- MLflow → [localhost:5000](http://localhost:5000) — every training run, with its metrics and saved model
- Grafana → [localhost:3000](http://localhost:3000) (`admin` / `admin`) → **Dashboards → Churn Prediction API** for live request rate and latency

---

## Project layout

```
scripts/
  generate_data.py    synthesizes the churn dataset
  train.py            trains the model, logs to MLflow, saves it
dags/
  training_dag.py     Airflow DAG — generate → train, nightly
app/
  main.py             FastAPI service: /predict, /metrics, /health
  model/              saved model.pkl (gitignored, rebuilt by CI)
monitoring/
  prometheus.yml                scrape config
  dashboard.json                Grafana dashboard
  grafana-datasource.yml
  grafana-dashboard-provider.yml
tests/
  test_api.py          pytest suite, doesn't need a running server
.github/workflows/
  ci.yml               GitHub Actions: test + Docker build
Dockerfile
docker-compose.yml
requirements.txt
```

---

## Running the tests yourself

```bash
pip install -r requirements.txt
python scripts/generate_data.py
MODEL_OUTPUT_PATH=app/model/model.pkl python scripts/train.py
pytest tests/ -v
```

---

## Why these tools specifically

**Airflow** because it's the standard for scheduling and monitoring retraining jobs in production — even with just two tasks in the DAG, it shows the pattern.

**MLflow** so every run is logged and comparable, and you can roll back to a known-good model instead of losing track of what changed between versions.

**FastAPI** for the async support and automatic docs, and because Pydantic validation rejects bad input before it ever reaches the model.

**Prometheus + Grafana** because it's the open-source standard for this, and the `/metrics` endpoint follows OpenMetrics, so it'd plug into any existing monitoring setup, not just this one.
