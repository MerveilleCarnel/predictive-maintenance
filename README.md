# Predictive Maintenance — Valve Condition

Systeme de maintenance predictive pour predire si la condition de la valve
d'un cycle de production hydraulique est optimale (100%) ou non.

## Dataset
[UCI — Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)

## Structure du projet
```
predictive-maintenance/
├── data/
│   ├── raw/            # PS2.txt, FS1.txt, profile.txt
│   └── processed/      # X_train.csv, X_test.csv, y_train.csv, y_test.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── api/
│       └── app.py
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── models/             # valve_model.pkl
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .gitignore
```

## Lancement rapide

### 1. Installation locale
```bash
pip install -r requirements.txt
```

### 2. Preparer les donnees et entrainer le modele
```bash
# Executer les notebooks dans l'ordre
jupyter notebook notebooks/
```

### 3. Lancer l'API
```bash
uvicorn src.api.app:app --reload --port 8000
```

### 4. Lancer avec Docker
```bash
docker-compose up --build
```

## API Endpoints

| Methode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Etat de l'API et du modele |
| POST | `/predict` | Prediction par index de cycle |
| GET | `/predict/{cycle_index}` | Prediction via URL |
| GET | `/cycles/info` | Infos sur les cycles disponibles |
| GET | `/docs` | Documentation interactive (Swagger) |
| GET | `/metrics` | Metriques Prometheus |

## Exemple d'utilisation

```bash
# Prediction pour le cycle 42
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"cycle_index": 42}'

# Reponse
{
  "cycle_index": 42,
  "valve_optimal": 1,
  "label": "Optimal",
  "probability": 0.9823,
  "confidence": "Haute"
}
```

## Monitoring
- Prometheus : http://localhost:9090
- Grafana    : http://localhost:3000  (admin / admin)
