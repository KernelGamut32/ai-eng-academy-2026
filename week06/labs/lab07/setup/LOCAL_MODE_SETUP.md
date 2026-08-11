# Local Mode Setup

For interface with a model, "local" mode, "ollama" mode, or "lmstudio" modes are available. Metric checks in the notebook will be based on "offline" but calls to an actual model may be more meaningful from a learning perspective (process based on performance).

## 1. MLflow Local (Docker)

If not already executed, download **mlflow-local-v3.15.1.tar** from the **images** folder at <https://gamuttechnologysvcs-my.sharepoint.com/:f:/p/asanders/IgD_SIVCz8YJQYh7BL3DUy4ZAVgU8-9SO8Lo3boIy-wwV8g?e=D4X53b>. From the location where **mlflow-local-v3.15.1.tar** has been stored, run `docker load -i mlflow-local-v3.15.1.tar` to load the image to local cache

From the **lab07** root folder:

```bash
docker run -d --name cordwell-mlflow -p 5000:5000 -v "$PWD/mlflow_server_data:/mlflow" ghcr.io/mlflow/mlflow:v3.15.1 mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////mlflow/mlflow.db --artifacts-destination /mlflow/artifacts
```

MLflow UI will be accessible from <http://localhost:5000>.
