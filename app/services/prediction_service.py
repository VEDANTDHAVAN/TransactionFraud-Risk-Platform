import pandas as pd
from app.model_loader import get_model

def get_risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "HIGH"
    if probability >= 0.5:
        return "MEDIUM"
    return "LOW"

def predict_transaction(data: dict):
    model = get_model()

    input_df = pd.DataFrame([data]) 

    probability = model.predict_proba(input_df)[0][1]
    prediction = int(probability >= 0.5)

    return {
        "fraud_probability": round(float(probability), 4),
        "prediction": prediction,
        "risk_level": get_risk_level(probability)
    }