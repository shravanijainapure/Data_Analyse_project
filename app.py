import os
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import streamlit as st

# 1. SET PAGE CONFIGURATION
st.set_page_config(page_title="Corporate Workforce Analytics", layout="wide")
st.title("📊 Enterprise Workforce Analytics & Attrition Dashboard")

# 2. RESOLVE ABSOLUTE PATHS FOR REPOSITORY DEPLOYMENT
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 3. LOAD DATASET AND MACHINE LEARNING ARTIFACTS SECURELY
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv"))

try:
    df = load_data()
    model = joblib.load(os.path.join(BASE_DIR, "attrition_model.pkl"))
    encoders = joblib.load(os.path.join(BASE_DIR, "label_encoders.pkl"))
    feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
except FileNotFoundError as e:
    st.error(f"❌ Initialization Error: Could not load data or model files. Details: {e}")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Dashboard Filters")
department_filter = st.sidebar.multiselect(
    "Select Department", options=df["Department"].unique(), default=df["Department"].unique()
)
gender_filter = st.sidebar.multiselect(
    "Select Gender", options=df["Gender"].unique(), default=df["Gender"].unique()
)

# Filter Dataframe based on selection
filtered_df = df[(df["Department"].isin(department_filter)) & (df["Gender"].isin(gender_filter))]

# --- KPI METRICS ---
total_emp = len(filtered_df)
attrition_rate = (filtered_df["Attrition"] == "Yes").sum() / total_emp * 100 if total_emp > 0 else 0
avg_satisfaction = filtered_df["JobSatisfaction"].mean() if total_emp > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Headcount", f"{total_emp}")
col2.metric("Attrition Rate", f"{attrition_rate:.1f}%")
col3.metric("Avg Job Satisfaction", f"{avg_satisfaction:.2f} / 4")

st.markdown("---")

# --- INTERACTIVE VISUALIZATIONS ---
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Monthly Income vs. Age by Attrition")
    fig1 = px.scatter(filtered_df, x="Age", y="MonthlyIncome", color="Attrition",
                      hover_data=["JobRole"], color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"})
    st.plotly_chart(fig1, use_container_width=True)

with chart_col2:
    st.subheader("Attrition Count by Job Role")
    fig2 = px.histogram(filtered_df, x="JobRole", color="Attrition", barmode="group",
                         color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"})
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# --- RISK PREDICTION SIMULATOR ---
st.subheader("🔮 Real-Time Employee Retention Risk Simulator")
st.write("Adjust parameters to calculate the attrition risk probability for an individual profile.")

sim_col1, sim_col2, sim_col3 = st.columns(3)

with sim_col1:
    sim_age = st.slider("Age", 18, 60, 35)
    sim_ot = st.selectbox("Overtime", ["Yes", "No"])
    sim_hike = st.slider("Percent Salary Hike", 10, 25, 14)

with sim_col2:
    # Max bound updated to 25000 to match realistic monthly constraints of the dataset
    sim_income = st.slider("Monthly Income ($)", 1000, 25000, 5000)
    sim_years = st.slider("Years At Company", 0, 40, 5)
    sim_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

with sim_col3:
    sim_satisfaction = st.slider("Environment Satisfaction Score", 1, 4, 3)
    sim_role = st.selectbox("Job Role", options=df["JobRole"].unique())
    sim_balance = st.slider("Work-Life Balance Score", 1, 4, 3)

if st.button("Run Attrition Risk Assessment"):
    # Build dynamic dictionary mapping to exactly match training shape
    input_data = {}
    for col in feature_cols:
        input_data[col] = 0  # Initialize defaults
    
    # Map interactive UI inputs
    input_data['Age'] = sim_age
    input_data['MonthlyIncome'] = sim_income
    input_data['YearsAtCompany'] = sim_years
    input_data['PercentSalaryHike'] = sim_hike
    input_data['EnvironmentSatisfaction'] = sim_satisfaction
    input_data['WorkLifeBalance'] = sim_balance
    input_data['OverTime'] = encoders['OverTime'].transform([sim_ot])[0]
    input_data['BusinessTravel'] = encoders['BusinessTravel'].transform([sim_travel])[0]
    input_data['JobRole'] = encoders['JobRole'].transform([sim_role])[0]
    
    # Convert input structure to DataFrame vector aligned with the model features
    input_df = pd.DataFrame([input_data])[feature_cols]
    
    # Predict Probability
    prob = model.predict_proba(input_df)[0][1] * 100
    
    if prob < 35:
        st.success(f"🟢 **Low Retention Risk:** This profile shows a {prob:.1f}% risk score. Retention probability is secure.")
    elif prob < 70:
        st.warning(f"🟡 **Moderate Retention Risk:** This profile shows a {prob:.1f}% risk score. Monitor work-life parameters.")
    else:
        st.error(f"🔴 **High Attrition Risk Alert:** This profile shows a {prob:.1f}% probability of leaving. Intervention suggested.")
