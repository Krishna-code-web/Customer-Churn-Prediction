import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import shap
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# PAGE CONFIG + GLOBAL STYLES
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Overall background */
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }

    /* Headings */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        color: #f8fafc;
    }

    /* Custom hero header */
    .hero {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #1e293b, #0ea5e9 180%);
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
    }
    .hero p {
        color: #cbd5e1;
        margin-top: 0.3rem;
        font-size: 0.95rem;
    }

    /* Card-like containers */
    .card {
        background: #1e293b;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }

    /* Risk badges */
    .badge-high {
        background: #7f1d1d; color: #fecaca;
        padding: 0.5rem 1rem; border-radius: 999px;
        font-weight: 600; display: inline-block;
    }
    .badge-medium {
        background: #78350f; color: #fde68a;
        padding: 0.5rem 1rem; border-radius: 999px;
        font-weight: 600; display: inline-block;
    }
    .badge-low {
        background: #14532d; color: #bbf7d0;
        padding: 0.5rem 1rem; border-radius: 999px;
        font-weight: 600; display: inline-block;
    }

    /* Sidebar tweak */
    section[data-testid="stSidebar"] {
        background-color: #0b1220;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
    }

    /* Hide default streamlit footer/menu for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------------------------
@st.cache_resource
def load_models():
    logistic_model = load("../models/logistic_model.pkl")
    rf_model = load("../models/rf_model.pkl")
    xgb_model = load("../models/xgb_model.pkl")
    return logistic_model, rf_model, xgb_model

logistic_model, rf_model, xgb_model = load_models()
preprocessor = rf_model.named_steps["preprocessor"]
rf_classifier = rf_model.named_steps["model"]

# ----------------------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📡 Churn Radar</h1>
        <p>Predict subscriber churn risk in real time and understand exactly why, with model-backed explanations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["🔮  Prediction", "📊  Model Insights"])

# ----------------------------------------------------------------------
# SIDEBAR — model selection lives here instead of buried in the form
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox(
        "Prediction model",
        ["Logistic Regression", "Random Forest", "XGBoost"],
        help="Switch models to compare predicted churn probability and explanations."
    )
    st.markdown("---")
    st.caption("Model performance snapshot")
    st.markdown(
        """
        | Model | ROC AUC | F1 |
        |---|---|---|
        | Logistic Reg. | 0.86 | 0.64 |
        | Random Forest | 0.85 | 0.65 |
        | XGBoost | 0.85 | 0.63 |
        """
    )

# ----------------------------------------------------------------------
# TAB 1 — PREDICTION
# ----------------------------------------------------------------------
with tab1:
    st.markdown("#### Customer Profile")

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            partner = st.selectbox("Partner", ["Yes", "No"])
        with c2:
            senior = st.selectbox("Senior Citizen", [0, 1])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
        with c3:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with c4:
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📞  Phone & Internet Services", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
            multiplelines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        with c2:
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            onlinesecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        with c3:
            onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            deviceprotection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

    with st.expander("📺  Streaming & Support", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streamingtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        with c2:
            streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with st.expander("📄  Billing & Contract", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with c2:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        with c3:
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )

    data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phoneservice],
        "MultipleLines": [multiplelines],
        "InternetService": [internet],
        "OnlineSecurity": [onlinesecurity],
        "OnlineBackup": [onlinebackup],
        "DeviceProtection": [deviceprotection],
        "TechSupport": [techsupport],
        "StreamingTV": [streamingtv],
        "StreamingMovies": [streamingmovies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total]
    })

    if model_choice == "Logistic Regression":
        model = logistic_model
    elif model_choice == "Random Forest":
        model = rf_model
    else:
        model = xgb_model

    st.write("")
    predict_clicked = st.button("🚀  Predict Churn", width='stretch')

    if predict_clicked:
        prob = model.predict_proba(data)[0][1]

        result_col, gauge_col = st.columns([1, 1])

        with result_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Churn Probability", f"{prob*100:.1f}%")
            st.progress(float(prob))

            if prob > 0.6:
                st.markdown('<span class="badge-high">🔴 High Risk Customer</span>', unsafe_allow_html=True)
            elif prob > 0.3:
                st.markdown('<span class="badge-medium">🟠 Medium Risk Customer</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-low">🟢 Low Risk Customer</span>', unsafe_allow_html=True)

            st.caption(f"Model used: **{model_choice}**")
            st.markdown('</div>', unsafe_allow_html=True)

        with gauge_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Quick read**")
            if prob > 0.6:
                st.write("This customer shows strong churn signals. Consider a proactive retention offer.")
            elif prob > 0.3:
                st.write("Moderate churn risk — worth a check-in or loyalty incentive.")
            else:
                st.write("Low churn risk. No immediate action needed.")
            st.markdown('</div>', unsafe_allow_html=True)

        # st.markdown("#### Why this prediction?")

        # X_transformed = preprocessor.transform(data)
        # feature_names = preprocessor.get_feature_names_out()
        # X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

        # explainer = shap.Explainer(rf_classifier)
        # shap_values = explainer(X_transformed_df)

        # fig = plt.figure(facecolor="#1e293b")
        # shap.plots.waterfall(shap_values[0, :, 1], show=False)
        # fig.patch.set_facecolor("#1e293b")
        # st.pyplot(fig, width='stretch')

# ----------------------------------------------------------------------
# TAB 2 — MODEL INSIGHTS
# ----------------------------------------------------------------------
with tab2:
    feature_names = preprocessor.get_feature_names_out()
    feature_importance = rf_classifier.feature_importances_

    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importance
    }).sort_values("importance", ascending=False)

    top_features = feat_imp.head(15)

    st.markdown("#### Top Drivers of Customer Churn")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(top_features["feature"], top_features["importance"], color="#0ea5e9")
    ax.invert_yaxis()
    ax.set_facecolor("#1e293b")
    fig.patch.set_facecolor("#1e293b")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    st.pyplot(fig, width='stretch')

    shap_col1, shap_col2 = st.columns(2)

    # Build a small synthetic sample purely for global SHAP plots
    base_row = pd.DataFrame({
        "gender": ["Female"], "SeniorCitizen": [0], "Partner": ["Yes"], "Dependents": ["No"],
        "tenure": [12], "PhoneService": ["Yes"], "MultipleLines": ["No"],
        "InternetService": ["Fiber optic"], "OnlineSecurity": ["No"], "OnlineBackup": ["No"],
        "DeviceProtection": ["No"], "TechSupport": ["No"], "StreamingTV": ["No"],
        "StreamingMovies": ["No"], "Contract": ["Month-to-month"], "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"], "MonthlyCharges": [70.0], "TotalCharges": [1000.0]
    })
    sample_data = pd.concat([base_row] * 50, ignore_index=True)
    sample_data["tenure"] = np.random.randint(0, 72, size=50)
    sample_data["MonthlyCharges"] = np.random.uniform(20, 120, size=50)
    sample_data["TotalCharges"] = np.random.uniform(100, 5000, size=50)

    X_sample_transformed = preprocessor.transform(sample_data)
    X_sample_df = pd.DataFrame(X_sample_transformed, columns=feature_names)
    explainer = shap.Explainer(rf_classifier)
    shap_values = explainer(X_sample_df)

    with shap_col1:
        st.markdown("#### SHAP Summary")
        fig = plt.figure(facecolor="#1e293b")
        shap.plots.beeswarm(shap_values[:, :, 1], max_display=12, show=False)
        st.pyplot(fig, width='stretch')

    with shap_col2:
        st.markdown("#### SHAP Feature Importance")
        fig = plt.figure(facecolor="#1e293b")
        shap.plots.bar(shap_values[:, :, 1], max_display=12, show=False)
        st.pyplot(fig, width='stretch')

    st.markdown("#### Model Performance")
    performance_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "ROC AUC": [0.86, 0.85, 0.85],
        "F1 Score": [0.64, 0.65, 0.63],
        "Precision": [0.52, 0.56, 0.55],
        "Recall": [0.84, 0.78, 0.75]
    })
    st.dataframe(performance_df, width='stretch', hide_index=True)

    st.markdown("#### Business Insights")

    insight_cols = st.columns(3)
    insights = [
        ("⏳ Tenure", "Short-tenure customers are far more likely to churn than long-term subscribers."),
        ("📃 Contract Type", "Month-to-month plans carry the highest churn risk; one/two-year contracts cut it sharply."),
        ("💵 Monthly Charges", "Higher bills correlate with higher churn — customers shop around for better value."),
        ("🌐 Internet Type", "Fiber optic subscribers churn more than DSL users."),
        ("🛡️ Add-on Services", "Customers without security, tech support, or device protection churn more."),
        ("🎯 Recommendation", "Bundle retention offers, push longer contracts, and target low-tenure customers first."),
    ]
    for i, (title, text) in enumerate(insights):
        with insight_cols[i % 3]:
            st.markdown(f'<div class="card"><b>{title}</b><br><span style="color:#cbd5e1;font-size:0.9rem;">{text}</span></div>', unsafe_allow_html=True)