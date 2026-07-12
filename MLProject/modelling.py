"""
modelling.py  (Kriteria 3 - MLflow Project entry point)
Melatih model Breast Cancer dan mencatatnya ke MLflow. Dipanggil oleh
`mlflow run` di dalam workflow CI.
"""

import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_data(data_path):
    train = pd.read_csv(os.path.join(data_path, "train.csv"))
    test = pd.read_csv(os.path.join(data_path, "test.csv"))
    X_train, y_train = train.drop(columns=["target"]), train["target"]
    X_test, y_test = test.drop(columns=["target"]), test["target"]
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="breast_cancer_preprocessing")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data(args.data_path)

    # `mlflow run` sudah membuka run aktif; pakai run itu. Bila dijalankan
    # langsung (python modelling.py), buka run baru sendiri.
    active_run = mlflow.active_run()
    own_run = active_run is None
    if own_run:
        active_run = mlflow.start_run()

    try:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
        mlflow.log_metric("precision", precision_score(y_test, preds))
        mlflow.log_metric("recall", recall_score(y_test, preds))
        mlflow.log_metric("f1", f1_score(y_test, preds))

        # Simpan model agar dapat dibungkus menjadi Docker image (mlflow build-docker).
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("Accuracy:", accuracy_score(y_test, preds))
        print("Run ID:", active_run.info.run_id)
    finally:
        if own_run:
            mlflow.end_run()


if __name__ == "__main__":
    main()
