import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "src"
    / "feature_repo"
    / "data"
    / "transactions.parquet"
)

MODEL_PATH = PROJECT_ROOT / "model" / "fraud_model.pkl"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

EXPERIMENT_NAME = "FraudDetection"
REGISTERED_MODEL_NAME = "fraud_model"

FEATURES = [
    "amount",
    "transaction_time",
    "user_age",
    "is_international",
    "avg_30d_spending",
]


# --------------------------------------------------
# Load and validate data
# --------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load transaction data and create a synthetic fraud target."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}\n"
            "Generate the transaction data before training."
        )

    df = pd.read_parquet(DATA_PATH)

    missing_columns = [
        column
        for column in FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required feature columns: {missing_columns}"
        )

    if df[FEATURES].isnull().any().any():
        raise ValueError(
            "Training data contains missing feature values."
        )

    # Synthetic labels for demonstration purposes only.
    df["fraud"] = (
        (
            (df["amount"] > 4000)
            & (df["is_international"] == 1)
        )
        | (
            df["amount"] > df["avg_30d_spending"] * 2.0
        )
    ).astype(int)

    class_counts = df["fraud"].value_counts().to_dict()

    print(f"Loaded {len(df)} records from {DATA_PATH}")
    print(f"Class distribution: {class_counts}")

    if df.empty or df["fraud"].nunique() != 2:
        raise ValueError(
            "Training requires non-empty data containing both "
            "fraud and non-fraud examples."
        )

    if df["fraud"].value_counts().min() < 2:
        raise ValueError(
            "Each target class needs at least two records "
            "for a stratified train/test split."
        )

    return df


# --------------------------------------------------
# Train, evaluate, and register model
# --------------------------------------------------

def train_model() -> None:
    """Train the model, log metrics, and register it in MLflow."""

    df = load_data()

    X = df[FEATURES]
    y = df["fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
    }

    print("\nEvaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Save a local backup independently of MLflow.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nLocal model saved: {MODEL_PATH}")

    # Configure MLflow tracking.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # Log training details and register the model.
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "n_estimators": 100,
                "random_state": 42,
                "test_size": 0.2,
                "features": ",".join(FEATURES),
            }
        )

        mlflow.log_metrics(metrics)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="fraud_model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train.head(5),
        )

        print(f"\nMLflow run ID: {run.info.run_id}")
        print(f"Model artifact URI: {model_info.model_uri}")

    # Verify that registration and version creation succeeded.
    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    registered_model = client.get_registered_model(
        REGISTERED_MODEL_NAME
    )

    if not registered_model.latest_versions:
        raise RuntimeError(
            f"Model '{REGISTERED_MODEL_NAME}' exists, "
            "but no registered versions were found."
        )

    latest_version = max(
        registered_model.latest_versions,
        key=lambda version: int(version.version),
    )

    print("\nSUCCESS: Model registered in MLflow.")
    print(f"Registered name: {registered_model.name}")
    print(f"Latest version: {latest_version.version}")
    print(f"Version status: {latest_version.status}")


if __name__ == "__main__":
    train_model()
