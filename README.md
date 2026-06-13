# Workflow-CI - Housing Price Prediction

Repository CI/CD pipeline untuk retraining model Machine Learning secara otomatis menggunakan GitHub Actions dan MLflow Project.

## Struktur Repository

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci_pipeline.yml         # GitHub Actions CI workflow
└── MLProject/
    ├── MLProject                   # MLflow Project config
    ├── conda.yaml                  # Conda environment
    ├── modelling.py                # Training script
    └── housing_preprocessing/      # Dataset preprocessed
        ├── train.csv
        └── test.csv
```

## Cara Kerja CI Pipeline

```
Push to main
    │
    ▼
Checkout repo
    │
    ▼
Setup Python 3.12
    │
    ▼
Install dependencies
    │
    ▼
Prepare dataset
    │
    ▼
mlflow run MLProject/
    │
    ▼
Upload artifacts (GitHub)
    │
    ▼
Commit artifacts to repo
    │
    ▼
Build Docker image (mlflow build-docker)
    │
    ▼
Push to Docker Hub
```

## GitHub Secrets yang Diperlukan

| Secret | Deskripsi |
|--------|-----------|
| `DOCKERHUB_USERNAME` | Username Docker Hub |
| `DOCKERHUB_TOKEN` | Access token Docker Hub |
| `MLFLOW_TRACKING_URI` | URI MLflow tracking server |
| `DAGSHUB_TOKEN` | Token autentikasi DagsHub |

## Docker Image

Image tersedia di Docker Hub:
```bash
docker pull <DOCKERHUB_USERNAME>/housing-price-prediction:latest

# Jalankan serving
docker run -p 8080:8080 <DOCKERHUB_USERNAME>/housing-price-prediction:latest
```

## Menjalankan MLflow Project Secara Lokal

```bash
mlflow run MLProject/ --env-manager local \
  -P n_estimators=100 \
  -P max_depth=10
```
