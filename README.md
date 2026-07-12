# Workflow-CI — Vina (Kriteria 3, Advance)

CI menggunakan **MLflow Project** untuk re-training otomatis, mengunggah artefak model,
lalu membangun **Docker image** dengan `mlflow build-docker` dan push ke **Docker Hub**.

## Struktur
```
Workflow-CI/
├── .github/workflows/ci.yml     # workflow CI (Actions)
└── MLProject/
    ├── MLProject                # definisi MLflow Project
    ├── conda.yaml               # environment
    ├── modelling.py             # entry point training
    └── breast_cancer_preprocessing/   # dataset siap latih
```

## Docker Hub
Image hasil CI: **https://hub.docker.com/r/vinanamira/cancer-model**

```bash
docker pull vinanamira/cancer-model:latest
docker run -p 5001:8080 vinanamira/cancer-model:latest
# inference:
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

## Secrets yang harus diset di GitHub
Settings → Secrets and variables → Actions:
- `DOCKERHUB_USERNAME` = `vinanamira`
- `DOCKERHUB_TOKEN`    = Docker Hub access token (hub.docker.com → Account Settings → Security)

## Menjalankan MLflow Project secara lokal
```bash
cd MLProject
mlflow run . --env-manager local
```
