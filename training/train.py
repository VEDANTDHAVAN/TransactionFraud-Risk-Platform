import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from mlflow import sklearn as mlflow_sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "fraudTrain.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_fraud_model.pkl"

RANDOM_STATE = 42


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place fraudTrain.csv inside the data/ folder."
        )

    df = pd.read_csv(DATA_PATH)
    return df


def feature_engineering(df):
    df = df.copy()

    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    df["transaction_hour"] = df["trans_date_trans_time"].dt.hour
    df["transaction_day"] = df["trans_date_trans_time"].dt.day
    df["transaction_month"] = df["trans_date_trans_time"].dt.month
    df["transaction_dayofweek"] = df["trans_date_trans_time"].dt.dayofweek

    df["age"] = df["trans_date_trans_time"].dt.year - df["dob"].dt.year

    df["distance_km"] = haversine_distance(
        df["lat"],
        df["long"],
        df["merch_lat"],
        df["merch_long"]
    )

    df["transactions_per_user"] = df.groupby("cc_num")["cc_num"].transform("count")

    df["avg_amt_per_user"] = df.groupby("cc_num")["amt"].transform("mean")

    df["amt_deviation"] = df["amt"] - df["avg_amt_per_user"]

    merchant_fraud_rate = df.groupby("merchant")["is_fraud"].mean()
    df["merchant_risk"] = df["merchant"].map(merchant_fraud_rate)

    hourly_fraud_rate = df.groupby("transaction_hour")["is_fraud"].mean()
    df["hourly_risk"] = df["transaction_hour"].map(hourly_fraud_rate)

    drop_cols = [
        "trans_date_trans_time",
        "cc_num",
        "first",
        "last",
        "street",
        "city",
        "state",
        "zip",
        "dob",
        "trans_num",
        "unix_time",
        "merchant",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
    ]

    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def build_model(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols
            ),
        ]
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics


def train():
    MODEL_DIR.mkdir(exist_ok=True)

    print("Loading dataset...")
    raw_df = load_data()

    print("Running feature engineering...")
    df = feature_engineering(raw_df)

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("Building model...")
    model = build_model(X_train)

    mlflow.set_experiment("fraud-risk-detection")

    with mlflow.start_run():
        print("Training model...")
        model.fit(X_train, y_train)

        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("subsample", 0.8)
        mlflow.log_param("colsample_bytree", 0.8)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        print(f"Saving model to {MODEL_PATH}...")
        joblib.dump(model, MODEL_PATH)

        mlflow_sklearn.log_model(
            sk_model=model,
            artifact_path="fraud_model"
        )

    print("\nTraining complete.")
    print(f"Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train()