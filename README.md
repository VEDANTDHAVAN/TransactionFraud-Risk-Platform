# Transaction Fraud Risk Detection Platform

An end-to-end Machine Learning system for detecting fraudulent financial transactions using advanced feature engineering, XGBoost, Explainable AI (SHAP), FastAPI, Streamlit, and Docker.

This project demonstrates the complete lifecycle of a production-style ML application, from data exploration and feature engineering to model training, explainability, API deployment, containerization, and interactive dashboards.

---

# Problem Statement

Financial fraud results in billions of dollars in losses every year. Traditional rule-based systems often struggle to identify evolving fraud patterns and generate a high number of false positives.

The goal of this project is to build an intelligent fraud detection system capable of:

* Detecting fraudulent transactions accurately
* Handling highly imbalanced datasets
* Providing explainable predictions
* Serving predictions through APIs
* Delivering insights through an interactive dashboard
* Supporting deployment in production environments

---

# Key Features

## Machine Learning

* Fraud detection using supervised learning
* Imbalanced classification handling
* Feature engineering
* Model benchmarking
* Hyperparameter experimentation
* Probability-based risk scoring

## Explainable AI

* SHAP-based explanations
* Feature importance analysis
* Individual transaction reasoning
* Global model interpretation

## Backend

* FastAPI REST API
* Pydantic validation
* Prediction endpoint
* Explainability endpoint
* Health checks

## Frontend

* Streamlit dashboard
* Interactive transaction input
* Fraud probability visualization
* Risk level display
* Explanation display

## Deployment

* Dockerized services
* Docker Compose orchestration
* Reproducible environments

---

# Dataset

Dataset used:

Fraud Detection Dataset by Kartik2112

Contains approximately:

* 1.29 Million transactions
* Customer demographics
* Merchant information
* Transaction metadata
* Geographical coordinates
* Fraud labels

Target Variable:

```txt
is_fraud

0 → Legitimate Transaction
1 → Fraudulent Transaction
```

Dataset is intentionally excluded from GitHub because of size limitations.

Download manually and place inside:

```txt
data/
```

Example:

```txt
data/
└── fraudTrain.csv
```

---

# Project Architecture

```txt
Raw Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Data Preprocessing
      │
      ▼
Model Training
(LogReg / RF / XGBoost)
      │
      ▼
Model Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
SHAP Explainability
      │
      ▼
FastAPI Backend
      │
      ▼
Streamlit Dashboard
      │
      ▼
Docker Deployment
```

---

# Project Structure

```txt
fraud-risk-platform/
│
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── model_loader.py
│   └── services/
│       ├── prediction_service.py
│       └── explain_service.py
│
├── frontend/
│   └── streamlit_app.py
│
├── models/
│   └── best_fraud_model.pkl
│
├── notebooks/
│   ├── phase1-eda.ipynb
│   ├── phase2-baselines.ipynb
│   ├── phase3-xgboost.ipynb
│   └── phase4-explainability.ipynb
│
├── assets/
│   ├── dashboard.png
│   ├── shap-summary.png
│   ├── swagger-ui.png
│   └── model-comparison.png
│
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements-api.txt
├── requirements-frontend.txt
└── README.md
```

---

# Exploratory Data Analysis

Performed analysis on:

* Fraud distribution
* Transaction amount distribution
* Time-based transaction patterns
* Merchant categories
* Customer demographics
* Geographic patterns

Important finding:

The dataset is highly imbalanced, requiring specialized handling techniques.

---

# Feature Engineering

Several domain-inspired features were created.

## Time Features

```txt
transaction_hour
transaction_day
transaction_month
transaction_dayofweek
```

Purpose:

Capture temporal fraud behavior.

---

## Customer Age

```txt
age
```

Derived from:

```txt
transaction date - date of birth
```

---

## Geographic Distance

Using Haversine Distance:

```txt
distance_km
```

Captures:

* Customer location
* Merchant location
* Geographic anomalies

---

## Transaction Velocity

```txt
transactions_per_user
```

Measures:

Customer activity frequency.

---

## Spending Behavior

```txt
avg_amt_per_user
amt_deviation
```

Captures:

Deviation from normal spending patterns.

---

## Merchant Risk

```txt
merchant_risk
```

Represents:

Historical fraud tendency associated with merchants.

---

## Hourly Risk

```txt
hourly_risk
```

Represents:

Historical fraud likelihood by hour.

---

# Models Evaluated

## Logistic Regression

Used as baseline model.

Advantages:

* Fast
* Interpretable
* Probabilistic

---

## Random Forest

Ensemble tree-based model.

Advantages:

* Non-linear learning
* Feature importance
* Better generalization

---

## XGBoost

Final selected model.

Advantages:

* Gradient boosting
* Excellent performance on tabular datasets
* Industry standard

---

# Model Results

| Model                          | Precision | Recall | F1 Score | ROC-AUC |
| ------------------------------ | --------- | ------ | -------- | ------- |
| Logistic Regression (Balanced) | 0.0386    | 0.7961 | 0.0736   | 0.9291  |
| Random Forest (Balanced)       | 0.2167    | 0.8055 | 0.3415   | 0.9775  |
| XGBoost                        | 0.9400    | 0.7835 | 0.8547   | 0.9969  |
| XGBoost (Balanced)             | 0.2433    | 0.9707 | 0.3891   | 0.9979  |

---

# Performance Analysis

### Best Overall Model

XGBoost

Achieved:

```txt
Precision : 94.00%
Recall    : 78.35%
F1 Score  : 85.47%
ROC-AUC   : 99.69%
```

### Key Insight

Balanced XGBoost achieved:

```txt
Recall = 97.07%
```

but at the cost of significantly lower precision.

This demonstrates the classic fraud detection tradeoff:

* High Recall → Catch more fraud
* High Precision → Reduce false alarms

---

# Explainable AI

SHAP was used to explain model predictions.

Capabilities:

* Global feature importance
* Local prediction explanations
* Feature contribution analysis

Questions answered:

* Why was a transaction flagged?
* Which features increased risk?
* Which features reduced risk?

---

# FastAPI Backend

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

---

## Prediction Endpoint

```http
POST /predict
```

Returns:

```json
{
  "fraud_probability": 0.91,
  "prediction": 1,
  "risk_level": "HIGH"
}
```

---

## Explainability Endpoint

```http
POST /explain
```

Returns:

```json
{
  "top_reasons": [
    {
      "feature": "amt_deviation",
      "impact": 0.53
    }
  ]
}
```

---

# Streamlit Dashboard

The dashboard provides:

* Transaction input form
* Fraud prediction
* Risk visualization
* SHAP explanation display

---

# Docker Deployment

Build and run:

```bash
docker compose up --build
```

Frontend:

```txt
http://localhost:8501
```

Backend:

```txt
http://localhost:8000/docs
```

---

# Local Setup

Clone repository:

```bash
git clone <repo-url>
cd fraud-risk-platform
```

Create environment:

```bash
python -m venv venv
```

Activate:

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r requirements-api.txt
```

Run API:

```bash
uvicorn app.main:app --reload
```

Run dashboard:

```bash
streamlit run frontend/streamlit_app.py
```

---

# Future Enhancements

* Real-time fraud monitoring
* Kafka event streaming
* Drift detection
* MLflow experiment tracking
* User authentication
* PostgreSQL storage
* Feature store integration
* Online learning pipelines
* Alerting and notifications
* Cloud deployment (AWS/GCP/Azure)

---

# Skills Demonstrated

Machine Learning

* Classification
* Feature Engineering
* Imbalanced Learning
* Model Evaluation
* Explainable AI

Data Science

* EDA
* Statistical Analysis
* Visualization

Backend Engineering

* FastAPI
* REST APIs
* Pydantic

Deployment

* Docker
* Docker Compose

Software Engineering

* Modular Architecture
* Reproducibility
* Version Control
* Documentation

---

# Author

Vedant Dhavan

Computer Engineering Student | AI Systems Builder | Machine Learning Engineer

Focus Areas:

* Machine Learning
* Agentic AI
* RAG & GraphRAG
* Explainable AI
* AI Security
* Production AI Systems
