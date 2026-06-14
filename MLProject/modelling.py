import os
import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="housing_preprocessing")

    args = parser.parse_args()

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
    }

    # contoh load data (sesuaikan punyamu)
    data = pd.read_csv(f"{args.data_dir}/data.csv")

    X = data.drop("target", axis=1)
    y = data["target"]

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42
    )

    mlflow.start_run()

    print("Training RandomForestRegressor...")
    model.fit(X, y)

    preds = model.predict(X)

    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    # log params
    mlflow.log_params(params)

    # log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # ✅ INI YANG BENAR (FIX ERROR KAMU SEBELUMNYA)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    print("\n=== TRAINING DONE ===")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R2   : {r2:.4f}")

    mlflow.end_run()

if __name__ == "__main__":
    main()
