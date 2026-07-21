import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import skops.io as sio
import urllib.request
import pandas as pd
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Educational ROI & Career Recommender",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Educational Program ROI & Career Decision Support Suite")
st.markdown(
    "Explore program financial outcomes, predict career earnings, and find your ideal academic path using machine learning."
)


# --- LOAD ASSETS ---
RF_MODEL_URL = "https://drive.google.com/uc?export=download&id=1jK-BU8u3JhywC5lcLsgrw8objTwA1GI2"

@st.cache_resource
def load_model():
    # Loads local XGBoost pipeline
    return sio.load("xgb_salary_model.skops", trusted=True)

@st.cache_data
def load_data():
    # Loads local data subset
    return pd.read_csv("streamlit_prog_data.csv")

@st.cache_resource
def load_rf_model():
    # Assuming you still download it from Drive or load locally
    return sio.load("rf_payoff_model.skops", trusted=True

@st.cache_resource
def load_scaler():
    # If you also export your scaler with skops
    return sio.load("recommender_scaler.skops", trusted=True)

# Initialize everything with the correct variable names
xgb_model = load_model()
prog_df = load_data()
rf_model = load_rf_model()
scaler = load_scaler()

# --- SIDEBAR INPUTS ---
st.sidebar.header("🎯 Your Target Profile & Constraints")

job_zone = st.sidebar.selectbox("Job Zone Level", [1, 2, 3, 4, 5], index=2)
tuition = st.sidebar.number_input(
    "Target In-State Tuition ($)", min_value=0, max_value=60000, value=10000, step=500
)
debt = st.sidebar.number_input(
    "Target Effective Debt ($)", min_value=0, max_value=300000, value=5000, step=1000
)
salary = st.sidebar.number_input(
    "Target Median Salary ($)", min_value=0, max_value=250000, value=100000, step=5000
)

# --- MAIN DASHBOARD TABS ---
tab1, tab2 = st.tabs(
    ["🔮 Model Predictions & Insights", "🎯 Program Recommendation Engine"]
)

with tab1:
  st.subheader("Predictive Analytics")
  st.markdown(
      "Using our optimized **Random Forest** and **XGBoost** pipelines to"
      " estimate career payoff times and earning potential."
  )

  if st.button("Run Predictions"):
    # Prepare input dataframe for models (must match training feature names)
    input_data = pd.DataFrame({
        "JOB_ZONE": [job_zone],
        "TUITION_IN_STATE": [tuition],
        "EFFECTIVE_DEBT": [debt],
        "YEARS_TO_PAYOFF": [
            3.0
        ],  # Baseline estimate for salary prediction pipeline if required
        "MEDIAN_SALARY": [salary],  # Baseline for payoff prediction if required
    })

    # Predict Salary using XGBoost pipeline
    # Note: Ensure columns match what xgb_salary_model expects ('JOB_ZONE', 'TUITION_IN_STATE', 'EFFECTIVE_DEBT', 'YEARS_TO_PAYOFF')
    xgb_input = input_data[
        ["JOB_ZONE", "TUITION_IN_STATE", "EFFECTIVE_DEBT", "YEARS_TO_PAYOFF"]
    ]
    pred_salary = xgb_model.predict(xgb_input)[0]

    # Predict Payoff Years using Random Forest pipeline
    rf_input = input_data[
        ["JOB_ZONE", "TUITION_IN_STATE", "EFFECTIVE_DEBT", "MEDIAN_SALARY"]
    ]
    pred_payoff = rf_model.predict(rf_input)[0]

    col1, col2 = st.columns(2)
    with col1:
      st.metric(
          label="Estimated Median Salary",
          value=f"${pred_salary:,.2f}",
          delta="XGBoost Prediction",
      )
    with col2:
      st.metric(
          label="Estimated Years to Payoff",
          value=f"{pred_payoff:.1f} Years",
          delta="Random Forest Prediction",
      )

with tab2:
  st.subheader("Top Program Matches")
  st.markdown(
      "Finding the closest program alignment based on your financial"
      " specifications using **Cosine Similarity**."
  )

  if st.button("Find Matching Programs"):
    # Scale user input
    user_vector = np.array([[tuition, debt, salary]])
    user_vector_scaled = scaler.transform(user_vector)

    # Scale database program features
    program_vectors = prog_df[
        ["TUITION_IN_STATE", "EFFECTIVE_DEBT", "MEDIAN_SALARY"]
    ].values
    program_vectors_scaled = scaler.transform(program_vectors)

    # Calculate similarity
    similarities = cosine_similarity(
        user_vector_scaled, program_vectors_scaled
    ).flatten()

    results_df = prog_df.copy()
    results_df["Match_Score"] = similarities
    top_matches = (
        results_df.sort_values(by="Match_Score", ascending=False)
        .head(5)[
            [
                "PROGRAM_NAME",
                "MEDIAN_SALARY",
                "EFFECTIVE_DEBT",
                "TUITION_IN_STATE",
                "Match_Score",
            ]
        ]
        .reset_index(drop=True)
    )

    # Clean display formatting
    top_matches["MEDIAN_SALARY"] = top_matches["MEDIAN_SALARY"].map(
        "${:,.2f}".format
    )
    top_matches["EFFECTIVE_DEBT"] = top_matches["EFFECTIVE_DEBT"].map(
        "${:,.2f}".format
    )
    top_matches["TUITION_IN_STATE"] = top_matches["TUITION_IN_STATE"].map(
        "${:,.2f}".format
    )
    top_matches["Match_Score"] = (top_matches["Match_Score"] * 100).map(
        "{:.2f}%".format
    )

    st.dataframe(top_matches, use_container_width=True)
