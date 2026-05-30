import streamlit as st
import requests

API_URL = "http://api:8000"

st.set_page_config(
    page_title="Fraud Risk Detection",
    layout="wide"
)

st.title("Transaction Fraud Risk Detection Dashboard")

st.write("Enter transaction details to predict fraud risk.")

with st.form("transaction_form"):
    category = st.selectbox("Category", [
       "shopping_net", "grocery_pos", "misc_net", "gas_transport",
       "entertainment", "food_dining", "personal_care", "health_fitness" 
    ])

    amt = st.number_input("Transaction Amount", min_value=0.0, value=500.0)
    gender = st.selectbox("Gender", ["M", "F"])
    city_pop = st.number_input("City Population", min_value=0, value=10000)
    job = st.text_input("Job", value="Engineer")

    transaction_hour = st.slider("Transaction Hour", 0, 23, 12)
    transaction_day = st.slider("Transaction Day", 1, 31, 15)
    transaction_month = st.slider("Transaction Month", 1, 12, 6)
    transaction_dayofweek = st.slider("Day of Week", 0, 6, 2)

    age = st.number_input("Age", min_value=1, value=35)
    distance_km = st.number_input("Distance KM", min_value=0.0, value=120.0)

    transactions_per_user = st.number_input("Transactions Per User", min_value=0, value=100)
    avg_amt_per_user = st.number_input("Average Amount Per User", min_value=0.0, value=70.0)
    amt_deviation = amt - avg_amt_per_user

    merchant_risk = st.slider("Merchant Risk", 0.0, 1.0, 0.05)
    hourly_risk = st.slider("Hourly Risk", 0.0, 1.0, 0.01)

    submitted = st.form_submit_button("Predict Fraud Risk")

if submitted:
    payload = {
        "category": category,
        "amt": amt,
        "gender": gender,
        "city_pop": city_pop,
        "job": job,
        "transaction_hour": transaction_hour,
        "transaction_day": transaction_day,
        "transaction_month": transaction_month,
        "transaction_dayofweek": transaction_dayofweek,
        "age": age,
        "distance_km": distance_km,
        "transactions_per_user": transactions_per_user,
        "avg_amt_per_user": avg_amt_per_user,
        "amt_deviation": amt_deviation,
        "merchant_risk": merchant_risk,
        "hourly_risk": hourly_risk
    }

    pred_res = requests.post(f"{API_URL}/predict", json=payload)
    explain_res = requests.post(f"{API_URL}/explain", json=payload)

    if pred_res.status_code == 200:
        result = pred_res.json()

        st.subheader("Prediction Result")

        st.metric(
            "Fraud Probability",
            f"{result['fraud_probability'] * 100:.2f}%"
        )

        st.metric("Risk Level", result["risk_level"])

        if result["prediction"] == 1:
            st.error("Transaction flagged as possible fraud.")
        else:
            st.success("Transaction appears normal.")

    else:
        st.error(pred_res.text)

    if explain_res.status_code == 200:
        explanation = explain_res.json()

        st.subheader("Top Explanation Factors")

        for item in explanation["top_reasons"]:
            st.write(
                f"**{item['feature']}** → impact: `{item['impact']}`"
            )
    else:
        st.error(explain_res.text)