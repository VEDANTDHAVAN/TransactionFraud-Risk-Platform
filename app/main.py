from fastapi import FastAPI
from app.schemas import TransactionInput, PredictionResponse, ExplainationResponse
from app.services.prediction_service import predict_transaction
from app.services.explain_service import explain_transaction

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

@app.post("/explain", response_model=ExplainationResponse)
def explain(transaction: TransactionInput):
    return explain_transaction(transaction.model_dump())