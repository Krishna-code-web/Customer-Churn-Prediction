import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit Branding
hide_menu = """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
"""

st.markdown(hide_menu, unsafe_allow_html=True)

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📊 Customer Churn Prediction System")

st.markdown("""
Welcome to the **Customer Churn Prediction Dashboard**.

👈 Select a page from the sidebar.

### Features

- Predict Customer Churn
- Multiple ML Models
- Probability Score
- Risk Analysis
- Model Comparison
- Download Prediction Report
- Professional Dashboard
""")

st.image(
    "https://images.unsplash.com/photo-1551288049-bebda4e38f71",
    use_container_width=True
)

st.info("Choose **Prediction** from the sidebar to begin.")