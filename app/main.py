from fastapi import FastAPI
from app.schemas import TransactionInput, PredictionResponse
from app.services.prediction_service import predict_transaction

app = FastAPI(
    title="Fraud Risk Detection API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Fraud Risk Detection API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    return predict_transaction(transaction.model_dump())