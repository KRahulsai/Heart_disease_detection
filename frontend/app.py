"""
Heart Disease Detection – Streamlit Frontend
=============================================
Auto-generates input fields based on the trained model's metadata
and sends predictions to the Flask backend API.
"""

import json
import os
import sys

# Add project root to path so config can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
import streamlit as st
import plotly.graph_objects as go
import config

# Use centralized configuration
API_URL = config.API_URL
META_PATH = config.METADATA_PATH


def get_metadata() -> dict | None:
    """Try to fetch metadata from the API; fall back to local file."""
    try:
        resp = requests.get(f"{API_URL}/metadata", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("metadata")
    except requests.ConnectionError:
        pass

    # Fallback: read local file
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            return json.load(f)
    return None


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Heart Disease Detector",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for premium look
# ──────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    .result-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
    }

    .result-positive {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
    }

    .result-negative {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
    }

    .result-card h2 {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }

    .result-card p {
        font-size: 1.1rem;
        opacity: 0.95;
    }

    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }

    .metric-card h4 {
        color: #495057;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }

    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #212529;
    }

    .sidebar .stSelectbox, .sidebar .stNumberInput {
        margin-bottom: 0.5rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>❤️ Heart Disease Detection</h1>
    <p>AI-powered prediction using advanced classification algorithms</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Load metadata
# ──────────────────────────────────────────────

meta = get_metadata()

if meta is None:
    st.error(
        "⚠️ Could not load model metadata. "
        "Please ensure you have run `python train.py --data data/heart.csv` first, "
        "and that the Flask backend is running (`python -m backend.app`)."
    )
    st.stop()

feature_names = meta["feature_names"]
cat_features = meta.get("categorical_features", [])
num_features = meta.get("numerical_features", [])
cat_encodings = meta.get("cat_encodings", {})
best_model = meta.get("best_model_name", "Unknown")
results = meta.get("results", {})

# ──────────────────────────────────────────────
# Sidebar – Model Info
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧠 Model Info")
    st.markdown(f"**Best Model:** {best_model}")

    if results:
        best_result = results.get(best_model, {})
        st.markdown(f"**Accuracy:** {best_result.get('accuracy', 'N/A')}")
        st.markdown(f"**F1 Score:** {best_result.get('f1', 'N/A')}")

    st.markdown("---")
    st.markdown("### 📊 All Model Results")
    for model_name, scores in results.items():
        with st.expander(f"{'🏆 ' if model_name == best_model else ''}{model_name}"):
            for metric, value in scores.items():
                st.metric(metric.capitalize(), f"{value:.4f}")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; opacity:0.6; font-size:0.85rem;'>"
        "Built with ❤️ using Streamlit + Flask + Scikit-learn"
        "</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# Input form
# ──────────────────────────────────────────────

st.markdown("### 📝 Enter Patient Details")
st.markdown("Fill in the features below and click **Predict** to get the result.")

input_data = {}

# Determine number of columns for layout
n_cols = 3
cols = st.columns(n_cols)

for idx, feat in enumerate(feature_names):
    col = cols[idx % n_cols]
    with col:
        if feat == 'sex':
            sex_choice = st.selectbox(
                f"🏷️ {feat}",
                options=["0 - Female", "1 - Male"],
                key=f"input_{feat}",
            )
            input_data[feat] = int(sex_choice.split()[0])
        elif feat in cat_features:
            options = cat_encodings.get(feat, [])
            input_data[feat] = st.selectbox(
                f"🏷️ {feat}",
                options=options,
                key=f"input_{feat}",
            )
        else:
            input_data[feat] = st.number_input(
                f"🔢 {feat}",
                value=0.0,
                format="%.4f",
                key=f"input_{feat}",
            )

st.markdown("")

# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────

if st.button("🫀 Predict Heart Disease", use_container_width=True):
    with st.spinner("Analyzing..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=input_data, timeout=10)
            result = resp.json()

            if result.get("status") == "ok":
                prediction = result["prediction"]
                label = result["label"]
                probability = result.get("probability")
                reason = result.get("reason")

                if reason:
                    st.warning(f"⚠️ {reason}")
                    st.info("Please provide more medical details for accurate prediction.")
                else:
                    # Determine if positive/negative
                    is_positive = prediction == 1

                    card_class = "result-positive" if is_positive else "result-negative"
                    icon = "⚠️" if is_positive else "✅"
                    title = "Heart Disease Detected" if is_positive else "No Heart Disease Detected"
                    desc = (
                        "The model indicates a risk of heart disease. Please consult a medical professional."
                        if is_positive
                        else "The model indicates no significant risk of heart disease."
                    )

                    st.markdown(
                        f"""
                        <div class="result-card {card_class}">
                            <h2>{icon} {title}</h2>
                            <p>{desc}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Probabilities
                    if probability:
                        st.markdown("### 📊 Class Probabilities")
                        target_classes = meta.get("target_classes", [str(i) for i in range(len(probability))])
                        prob_cols = st.columns(len(probability))
                        for i, (cls, prob) in enumerate(zip(target_classes, probability)):
                            with prob_cols[i]:
                                st.metric(f"Class {cls}", f"{prob * 100:.1f}%")
                                
                        # Risk visualization chart
                        positive_prob = probability[-1] * 100
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=positive_prob,
                            title={'text': "Risk of Heart Disease (%)", 'font': {'size': 20}},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#e74c3c" if positive_prob >= 50 else "#2ecc71"},
                                'steps': [
                                    {'range': [0, 30], 'color': "#e8f8f5"},
                                    {'range': [30, 50], 'color': "#fdf2e9"},
                                    {'range': [50, 100], 'color': "#fdedec"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"❌ Error: {result.get('message', 'Unknown error')}")

        except requests.ConnectionError:
            st.error(
                "🔌 Cannot connect to the backend API. "
                "Make sure Flask is running: `python -m backend.app`"
            )
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# ──────────────────────────────────────────────
# Footer with charts
# ──────────────────────────────────────────────

st.markdown("---")
st.markdown("### 📈 Model Training Visualizations")

charts_dir = os.path.join(os.path.dirname(__file__), "..", "model")

chart_files = {
    "Model Comparison": "model_comparison.png",
    "Correlation Heatmap": "correlation_heatmap.png",
    "Target Distribution": "target_distribution.png",
}

chart_cols = st.columns(len(chart_files))
for i, (title, fname) in enumerate(chart_files.items()):
    fpath = os.path.join(charts_dir, fname)
    with chart_cols[i]:
        if os.path.exists(fpath):
            st.image(fpath, caption=title, use_container_width=True)
        else:
            st.info(f"{title} not available yet.")
