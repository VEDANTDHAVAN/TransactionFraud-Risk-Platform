from pydantic import BaseModel

class TransactionInput(BaseModel):
    category: str
    amt: float
    gender: str
    city_pop: int
    job: str
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_dayofweek: int
    age: int
    distance_km: float
    transactions_per_user: int
    avg_amt_per_user: float
    amt_deviation: float
    merchant_risk: float
    hourly_risk: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: int
    risk_level: str