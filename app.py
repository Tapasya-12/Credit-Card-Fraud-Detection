import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Fraud Risk Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <h1 style="margin-bottom:4px;"> 💳 Credit Card Fraud Detection</h1>
    <p style="color:grey; margin-top:0;">
    </p>
    """,
    unsafe_allow_html=True
)

MODELS = {
    "Logistic Regression": "models/logistic.pkl",
    "Random Forest": "models/random_forest.pkl",
    "SVM": "models/svm.pkl",
    "XGBoost": "models/xgboost.pkl",
    "LightGBM": "models/lightgbm.pkl",
    "Neural Network": "models/mlp.pkl",
    "Isolation Forest (Anomaly)": "models/isolation_forest.pkl"
}

with st.sidebar:
    st.header("Model Selection")
    model_name = st.radio("Choose a model", list(MODELS.keys()))

model = joblib.load(MODELS[model_name])

st.markdown("## Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Amounts")
    amount = float(st.text_input("Transaction Amount", "1200"))
    avg_amount = float(st.text_input("Avg Transaction Amount", "1800"))
    tx_count = st.slider("Transactions in last 24 hours", 0, 50, 1)

with col2:
    st.markdown("### Card & Merchant")
    account_age = st.number_input(
        "Account Age (days)",
        min_value=30,
        max_value=6000,
        value=500,
        step=1
    )
    tx_type = st.selectbox("Transaction Type", ["Online", "POS", "ATM"])
    merchant = st.selectbox(
        "Merchant Category",
        ["Grocery", "Electronics", "Travel", "Food", "Healthcare"]
    )


with col3:
    st.markdown("### Device & Location")
    tx_country = st.selectbox("Transaction Country", ["India", "USA", "UK", "UAE"])
    bill_country = st.selectbox("Billing Country", ["India", "USA", "UK", "UAE"])
    device = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
    browser = st.selectbox("Browser", ["Chrome", "Firefox", "Safari", "Edge"])

input_df = pd.DataFrame([{
    "amount": amount,
    "avg_transaction_amount": avg_amount,
    "transaction_count_24h": tx_count,
    "account_age_days": account_age,
    "amount_to_avg_ratio": amount / (avg_amount + 1),
    "high_velocity_flag": int(tx_count >= 5),
    "foreign_tx_flag": int(tx_country != bill_country),
    "new_account_flag": int(account_age < 180),
    "transaction_type": tx_type,
    "merchant_category": merchant,
    "currency": "INR",
    "card_type": "Credit",
    "customer_region": "North",
    "transaction_country": tx_country,
    "billing_country": bill_country,
    "device_type": device,
    "browser": browser
}])

def risk_level_from_prob(prob):
    if prob >= 0.75:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


def system_decision_from_prob(prob):
    if prob >= 0.75:
        return "BLOCK & MANUAL REVIEW"
    elif prob >= 0.40:
        return "STEP-UP AUTHENTICATION"
    else:
        return "ALLOW TRANSACTION"


def risk_level_from_anomaly(score):
    if score < -0.10:
        return "HIGH"
    elif score < 0.05:
        return "MEDIUM"
    else:
        return "LOW"


def system_decision_from_anomaly(score):
    if score < -0.10:
        return "BLOCK & MANUAL REVIEW"
    elif score < 0.05:
        return "STEP-UP AUTHENTICATION"
    else:
        return "ALLOW TRANSACTION"

if st.button("Analyze Transaction", use_container_width=True):

    st.markdown(
        "<h2 style='margin-bottom:6px;'>Risk Assessment Result</h2>",
        unsafe_allow_html=True
    )

    colA, colB, colC = st.columns(3)

    if "Isolation" in model_name:
        score = model.named_steps["model"].decision_function(
            model.named_steps["preprocess"].transform(input_df)
        )[0]

        risk = risk_level_from_anomaly(score)
        decision = system_decision_from_anomaly(score)

        colA.markdown(
            f"""
            <div>
                <p style="font-size:20px; color:grey; margin-bottom:2px;">Anomaly Score</p>
                <p style="font-size:26px; font-weight:600; margin-top:0;">{round(score,4)}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        prob = model.predict_proba(input_df)[0][1]
        risk = risk_level_from_prob(prob)
        decision = system_decision_from_prob(prob)

        colA.markdown(
            f"""
            <div>
                <p style="font-size:20px; color:grey; margin-bottom:2px;">Fraud Probability</p>
                <p style="font-size:26px; font-weight:600; margin-top:0;">{prob:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    colB.markdown(
        f"""
        <div>
            <p style="font-size:20px; color:grey; margin-bottom:2px;">Risk Level</p>
            <p style="font-size:26px; font-weight:600; margin-top:0;">{risk}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    colC.markdown(
        f"""
        <div>
            <p style="font-size:20px; color:grey; margin-bottom:2px;">System Decision</p>
            <p style="font-size:26px; font-weight:600; margin-top:0;">{decision}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
