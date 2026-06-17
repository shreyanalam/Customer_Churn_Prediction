"""
app.py
------------------------
Streamlit web app for the Customer Churn Prediction project.

Two tabs:
1. "Predict Churn" - a form where you enter a customer's details and get
   an instant churn prediction with probability.
2. "Insights Dashboard" - key charts and stats from the dataset, the kind
   of thing a retention team would actually look at.

Run with:
    streamlit run app.py
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict_utils import ChurnPredictor

# --- Page setup ---
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

DATA_PATH = "data/customer_churn_data.csv"


@st.cache_resource
def get_predictor():
    """Loads the trained model once and reuses it across reruns/sessions."""
    return ChurnPredictor()


@st.cache_data
def get_dataset():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


# --- Header ---
st.title("📉 Customer Churn Prediction")
st.caption(
    "An intermediate-level machine learning project that predicts whether a "
    "telecom customer is likely to churn, based on their account and service details."
)

tab_predict, tab_dashboard = st.tabs(["🔮 Predict Churn", "📊 Insights Dashboard"])

# ======================================================================
# TAB 1: PREDICTION FORM
# ======================================================================
with tab_predict:
    st.subheader("Enter Customer Details")

    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months with company)", 0, 72, 12)

    with col2:
        st.markdown("**Services**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("**Account & Billing**")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(monthly_charges * tenure), step=10.0
        )

    st.write("")
    if st.button("Predict Churn", type="primary", use_container_width=True):
        customer = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        result = predictor.predict(customer)
        probability_pct = result["churn_probability"] * 100

        st.write("---")
        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            if result["prediction"] == "Churn":
                st.error(f"⚠️ Prediction: **{result['prediction']}**")
            else:
                st.success(f"✅ Prediction: **{result['prediction']}**")
            st.metric("Churn Probability", f"{probability_pct:.1f}%")
            st.caption(f"Model used: {result['model_used']}")

        with result_col2:
            risk_level = "High" if probability_pct >= 60 else "Medium" if probability_pct >= 30 else "Low"
            risk_color = {"High": "#E63946", "Medium": "#F4A261", "Low": "#2A9D8F"}[risk_level]
            st.markdown(f"**Risk Level:** <span style='color:{risk_color}; font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)
            st.progress(min(int(probability_pct), 100))
            if risk_level == "High":
                st.write("Suggested action: prioritize this customer for a retention offer (discount, contract upgrade incentive, or support outreach).")
            elif risk_level == "Medium":
                st.write("Suggested action: monitor this customer and consider a proactive check-in.")
            else:
                st.write("Suggested action: no action needed -- this customer profile looks stable.")

# ======================================================================
# TAB 2: INSIGHTS DASHBOARD
# ======================================================================
with tab_dashboard:
    st.subheader("Dataset Overview")

    if not os.path.exists(DATA_PATH):
        st.warning("Dataset not found. Run 'python generate_dataset.py' first.")
        st.stop()

    df = get_dataset()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{len(df):,}")
    k2.metric("Churn Rate", f"{(df['Churn'] == 'Yes').mean() * 100:.1f}%")
    k3.metric("Avg. Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
    k4.metric("Avg. Tenure", f"{df['tenure'].mean():.1f} months")

    st.write("---")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        fig = px.pie(
            churn_counts, names="Churn", values="Count", hole=0.45,
            color="Churn", color_discrete_map={"Yes": "#E63946", "No": "#2A9D8F"},
            title="Overall Churn Split",
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        contract_churn = (
            df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        )
        contract_churn.columns = ["Contract", "Churn Rate (%)"]
        fig = px.bar(
            contract_churn, x="Contract", y="Churn Rate (%)",
            title="Churn Rate by Contract Type", color="Contract",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        fig = px.histogram(
            df, x="tenure", color="Churn", nbins=30, barmode="overlay",
            color_discrete_map={"Yes": "#E63946", "No": "#2A9D8F"},
            title="Tenure Distribution by Churn",
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col4:
        fig = px.box(
            df, x="Churn", y="MonthlyCharges", color="Churn",
            color_discrete_map={"Yes": "#E63946", "No": "#2A9D8F"},
            title="Monthly Charges by Churn Status",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("---")
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(20), use_container_width=True)