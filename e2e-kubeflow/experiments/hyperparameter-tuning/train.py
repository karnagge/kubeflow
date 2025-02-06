import argparse
import json
import os

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score


def load_data():
    """Load preprocessed data."""
    X_train = pd.read_csv("/data/processed/X_train.csv")
    y_train = pd.read_csv("/data/processed/y_train.csv")["target"]
    X_val = pd.read_csv("/data/processed/X_val.csv")
    y_val = pd.read_csv("/data/processed/y_val.csv")["target"]

    return X_train, y_train, X_val, y_val


def train_and_evaluate(args):
    """Train XGBoost model and evaluate performance."""
    # Load data
    X_train, y_train, X_val, y_val = load_data()

    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Set up parameters
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["error", "auc", "logloss"],
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "random_state": 42,
    }

    # Create watchlist for evaluation
    watchlist = [(dtrain, "train"), (dval, "validation")]

    # Train model with early stopping
    model = xgb.train(
        params,
        dtrain,
        params["n_estimators"],
        watchlist,
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Make predictions
    y_train_pred = model.predict(dtrain)
    y_val_pred = model.predict(dval)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred > 0.5)
    val_accuracy = accuracy_score(y_val, y_val_pred > 0.5)
    train_auc = roc_auc_score(y_train, y_train_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)

    # Print metrics for Katib to collect
    print(f"Training-Accuracy={train_accuracy}")
    print(f"Training-AUC={train_auc}")
    print(f"Validation-Accuracy={val_accuracy}")
    print(f"Validation-AUC={val_auc}")

    # Save model if validation accuracy meets target
    if val_accuracy >= 0.90:
        os.makedirs("/output", exist_ok=True)
        model.save_model("/output/model.json")

        # Save model configuration
        config = {
            "parameters": params,
            "best_iteration": model.best_iteration,
            "feature_names": list(X_train.columns),
            "metrics": {
                "train_accuracy": float(train_accuracy),
                "train_auc": float(train_auc),
                "val_accuracy": float(val_accuracy),
                "val_auc": float(val_auc),
            },
        }

        with open("/output/model_config.json", "w") as f:
            json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--min_child_weight", type=int, default=1)
    parser.add_argument("--subsample", type=float, default=0.8)
    args = parser.parse_args()

    train_and_evaluate(args)


if __name__ == "__main__":
    main()
