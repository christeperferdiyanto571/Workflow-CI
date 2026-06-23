import argparse
import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# ARGUMENT
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="housing_preprocessing")
    return parser.parse_args()

# =========================
# LOAD DATA
# =========================
def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop("MedHouseVal", axis=1)
    y_train = train_df["MedHouseVal"]

    X_test = test_df.drop("MedHouseVal", axis=1)
    y_test = test_df["MedHouseVal"]

    return X_train, X_test, y_train, y_test

# =========================
# MAIN
# =========================
def main():
    args = parse_args()

    X_train, X_test, y_train, y_test = load_data(args.data_dir)

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": 42,
        "n_jobs": -1
    }

    logger.info("Training RandomForestRegressor...")

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # =========================
    # MLflow
    # =========================
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # ✅ TAMBAH: Export model ke folder artifacts/model
        os.makedirs("artifacts", exist_ok=True)
        mlflow.sklearn.save_model(model, "artifacts/model")
        
        # Log ke MLflow juga
        mlflow.sklearn.log_model(model, "model")

        # simpan params lokal
        with open("artifacts/params.json", "w") as f:
            json.dump(params, f, indent=4)

        mlflow.log_artifact("artifacts/params.json")

    print("\n=== TRAINING DONE ===")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R2  : {r2:.4f}")

if __name__ == "__main__":
    main()
