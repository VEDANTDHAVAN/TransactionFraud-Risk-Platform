import joblib
from pathlib import Path

MODEL_PATH = Path("models/best_fraud_model.pkl")

model = joblib.load(MODEL_PATH)

def get_model():
    return model