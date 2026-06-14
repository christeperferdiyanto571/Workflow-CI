import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import argparse

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
        "min_samples_leaf": args.min_samples_leaf
    }

    # contoh data dummy (ganti sesuai dataset kamu)
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)
    X_test = np.random.rand(20, 5)
    y_test = np.random.rand(20)

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42
    )

    # 🔥 INI FIX UTAMA
    with mlflow.start_run():

        print("Training RandomForestRegressor...")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # log parameter
        mlflow.log_params(params)

        # log metrics
        mlflow.log_metrics({
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })

        # log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("\n=== TRAINING DONE ===")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAE  : {mae:.4f}")
        print(f"R2   : {r2:.4f}")


if __name__ == "__main__":
    main()
