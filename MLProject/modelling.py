"""
modelling.py (MLProject version)
Digunakan dalam workflow CI via mlflow run.
Mendukung argumen CLI untuk hyperparameter.
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Housing Price Prediction Model")
    parser.add_argument('--n_estimators',      type=int,   default=100)
    parser.add_argument('--max_depth',         type=int,   default=10)
    parser.add_argument('--min_samples_split', type=int,   default=2)
    parser.add_argument('--min_samples_leaf',  type=int,   default=1)
    parser.add_argument('--data_dir',          type=str,   default="housing_preprocessing")
    return parser.parse_args()


def load_data(data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    X_train = train_df.drop('MedHouseVal', axis=1)
    y_train = train_df['MedHouseVal']
    X_test  = test_df.drop('MedHouseVal', axis=1)
    y_test  = test_df['MedHouseVal']
    return X_train, X_test, y_train, y_test


def save_feature_importance_plot(model, feature_names, path="feature_importance.png"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def save_prediction_plot(y_test, y_pred, path="prediction_vs_actual.png"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.4, s=10)
    m, M = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([m, M], [m, M], 'r--', lw=2)
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.title("Prediction vs Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path

def main():
    args = parse_args()

    # MLflow Experiment
    mlflow.set_experiment("Housing_CI_Pipeline")

    # Load data
    X_train, X_test, y_train, y_test = load_data(args.data_dir)

    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'random_state': 42,
        'n_jobs': -1,
    }

    logger.info("Training RandomForestRegressor...")

    # Train Model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # ==========================
    # Manual MLflow Logging
    # ==========================
    mlflow.log_params(params)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Log Model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    # ==========================
    # Artifact 1
    # ==========================
    fi_path = save_feature_importance_plot(
        model,
        list(X_train.columns)
    )

    mlflow.log_artifact(
        fi_path,
        artifact_path="plots"
    )

    # ==========================
    # Artifact 2
    # ==========================
    pred_path = save_prediction_plot(
        y_test,
        y_pred
    )

    mlflow.log_artifact(
        pred_path,
        artifact_path="plots"
    )

    # ==========================
    # Artifact 3
    # ==========================
    params_path = "run_params.json"

    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)

    mlflow.log_artifact(params_path)

    # ==========================
    # Local Artifact
    # ==========================
    os.makedirs("artifacts", exist_ok=True)

    mlflow.sklearn.save_model(
        sk_model=model,
        path="artifacts/model"
    )

    logger.info(f"MSE  : {mse:.4f}")
    logger.info(f"RMSE : {rmse:.4f}")
    logger.info(f"MAE  : {mae:.4f}")
    logger.info(f"R2   : {r2:.4f}")

    print("\n=== Training Completed ===")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R2   : {r2:.4f}")
    print("Artifacts saved to: artifacts/")