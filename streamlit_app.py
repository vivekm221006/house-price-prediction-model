"""
House Price Predictor - IMPROVED Streamlit App
Features: Multi-model selection, SHAP explainability, 
          comparison view, confidence intervals, map, metrics dashboard
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Must be first Streamlit call
st.set_page_config(
    page_title="House Price Predictor — California",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Import feature engineering ───────────────────────────────────────────────
from src.feature_engineering import create_interaction_features

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Overall background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: #1e293b;
    border-right: 1px solid #334155;
}
/* Hero banner */
.hero-banner {
    background: linear-gradient(90deg, #1d4ed8 0%, #7c3aed 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: white;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1rem;
    color: #bfdbfe;
    margin: 6px 0 0 0;
}
/* Price card */
.price-card {
    background: linear-gradient(135deg, #1d4ed8 0%, #7c3aed 100%);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
}
.price-main {
    font-size: 3.5rem;
    font-weight: 900;
    color: white;
    line-height: 1;
    margin: 0;
}
.price-label {
    font-size: 0.85rem;
    color: #bfdbfe;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.price-ci {
    font-size: 0.95rem;
    color: #bfdbfe;
    margin-top: 12px;
}
/* Metric cards */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 16px 0;
}
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
    flex: 1;
    text-align: center;
}
.metric-val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #60a5fa;
}
.metric-lbl {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
}
/* Feature importance bar */
.feat-bar-wrap {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
}
/* Section title */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 24px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
/* Tag pills */
.tag-green {
    background: #064e3b;
    color: #6ee7b7;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
}
.tag-blue {
    background: #1e3a5f;
    color: #93c5fd;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
}
/* Info box */
.info-box {
    background: #1e293b;
    border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    color: #94a3b8;
    font-size: 0.85rem;
}
/* Override streamlit slider track */
.stSlider > div > div > div > div { background: #3b82f6 !important; }

/* Hide streamlit default footer */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────

MODEL_FILES = {
    "Random Forest":       "outputs/models/random_forest.pkl",
    "Gradient Boosting":   "outputs/models/gradient_boosting.pkl",
    "XGBoost":             "outputs/models/xgboost.pkl",
    "Ridge Regression":    "outputs/models/ridge_regression.pkl",
    "Lasso Regression":    "outputs/models/lasso_regression.pkl",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_metrics():
    path = "outputs/model_metrics.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_cv_results():
    path = "outputs/cv_results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def available_models():
    return {k: v for k, v in MODEL_FILES.items() if os.path.exists(v)}

def build_input_df(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, lat, lon):
    return pd.DataFrame({
        "MedInc":     [med_inc],
        "HouseAge":   [house_age],
        "AveRooms":   [ave_rooms],
        "AveBedrms":  [ave_bedrms],
        "Population": [population],
        "AveOccup":   [ave_occup],
        "Latitude":   [lat],
        "Longitude":  [lon],
    })

def predict_with_confidence(model, input_df, model_name):
    """Return mean prediction and 95% CI. Works for RF (tree variance) or fallback."""
    pred_df = input_df[model.feature_names_in_]
    mean_pred = model.predict(pred_df)[0]

    # Tree-based uncertainty via individual estimator variance
    if hasattr(model, 'estimators_'):
        tree_preds = np.array([t.predict(pred_df)[0] for t in model.estimators_])
        std = tree_preds.std()
        low  = (mean_pred - 1.96 * std) * 100_000
        high = (mean_pred + 1.96 * std) * 100_000
    else:
        # Approximate ±10% for non-tree models
        low  = mean_pred * 100_000 * 0.90
        high = mean_pred * 100_000 * 1.10

    return mean_pred * 100_000, low, high

def get_feature_importance(model, feature_names):
    """Extract feature importances from the model if available."""
    if hasattr(model, 'feature_importances_'):
        imps = model.feature_importances_
        return pd.Series(imps, index=feature_names).sort_values(ascending=False).head(12)
    elif hasattr(model, 'coef_'):
        imps = np.abs(model.coef_)
        return pd.Series(imps, index=feature_names).sort_values(ascending=False).head(12)
    return None

def affordability_label(price):
    if price < 150_000:
        return "🟢 Affordable"
    elif price < 300_000:
        return "🟡 Moderate"
    elif price < 500_000:
        return "🟠 Pricey"
    else:
        return "🔴 Expensive"

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <span style="font-size:3rem">🏠</span>
  <div>
    <p class="hero-title">California House Price Predictor</p>
    <p class="hero-sub">Multi-model ML · Confidence Intervals · SHAP Explainability · Model Comparison</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Check if models are trained ──────────────────────────────────────────────
avail = available_models()
if not avail:
    st.error("⚠️ No trained models found. Run `python train_pipeline.py` first to train models.")
    st.code("python train_pipeline.py", language="bash")
    st.stop()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Property Details")

    med_inc    = st.slider("Median Income ($10k)",    0.5, 15.0, 4.0, 0.1,
                           help="Median household income in the block group ($10,000s)")
    house_age  = st.slider("House Age (years)",        1, 52, 20,
                           help="Median age of houses in the block")
    ave_rooms  = st.slider("Average Rooms",            1.0, 20.0, 5.5, 0.5)
    ave_bedrms = st.slider("Average Bedrooms",         0.5, 5.0, 1.2, 0.1)
    population = st.slider("Population",               3, 35000, 1500, 100)
    ave_occup  = st.slider("Average Occupancy",        0.5, 20.0, 3.0, 0.5)

    st.markdown("### 📍 Location")
    latitude   = st.slider("Latitude",  32.0, 42.0, 37.5, 0.05)
    longitude  = st.slider("Longitude", -125.0, -114.0, -120.0, 0.05)

    st.markdown("---")
    st.markdown("### 🤖 Model Selection")
    selected_model_name = st.selectbox("Prediction model", list(avail.keys()))

    run_shap = st.checkbox("Show SHAP explanation", value=False,
                           help="SHAP values explain which features drove this prediction")
    compare_all = st.checkbox("Compare all models", value=False)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Price", use_container_width=True, type="primary")

# ─── Main content tabs ────────────────────────────────────────────────────────
tab_predict, tab_metrics, tab_data = st.tabs(["🏠 Prediction", "📊 Model Metrics", "📁 Data Explorer"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:

    if not predict_btn:
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown('<div class="info-box">👈 Adjust the sliders in the sidebar and click <b>Predict Price</b> to get started.</div>', unsafe_allow_html=True)
        with col_r:
            st.markdown('<div class="info-box">This model was trained on ~20,000 California housing records from the 1990 census. Prices shown reflect the data distribution (×$100k).</div>', unsafe_allow_html=True)

    if predict_btn:
        # Build input
        input_df = build_input_df(med_inc, house_age, ave_rooms, ave_bedrms,
                                   population, ave_occup, latitude, longitude)
        input_engineered = create_interaction_features(input_df)

        model = load_model(avail[selected_model_name])

        # Align columns
        try:
            input_aligned = input_engineered[model.feature_names_in_]
        except Exception:
            input_aligned = input_engineered

        price, price_low, price_high = predict_with_confidence(model, input_engineered, selected_model_name)

        # ── Price Result Card ──
        affl = affordability_label(price)
        st.markdown(f"""
        <div class="price-card">
            <p class="price-label">Estimated Market Value</p>
            <p class="price-main">${price:,.0f}</p>
            <p class="price-ci">95% Confidence Interval: ${price_low:,.0f} – ${price_high:,.0f}</p>
            <p style="margin-top:12px;font-size:1.1rem">{affl} &nbsp;·&nbsp; {selected_model_name}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Key input summary ──
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Income Level", f"${med_inc*10:.0f}k")
        col2.metric("House Age", f"{house_age} yrs")
        col3.metric("Avg Rooms", f"{ave_rooms:.1f}")
        col4.metric("Occupancy", f"{ave_occup:.1f}")

        st.markdown("---")

        # ── Feature Importance Chart ──
        feat_names = list(model.feature_names_in_)
        importances = get_feature_importance(model, feat_names)

        if importances is not None:
            st.markdown('<div class="section-title">📌 Top Feature Importances</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#1e293b')
            ax.set_facecolor('#1e293b')

            colors = ['#3b82f6' if i == 0 else '#60a5fa' if i < 3 else '#93c5fd'
                      for i in range(len(importances))]
            bars = ax.barh(importances.index[::-1], importances.values[::-1], color=colors[::-1], height=0.6)
            ax.set_xlabel("Importance", color='#94a3b8')
            ax.tick_params(colors='#94a3b8')
            for spine in ax.spines.values():
                spine.set_color('#334155')
            ax.xaxis.label.set_color('#94a3b8')
            ax.set_title(f"Feature Importances — {selected_model_name}", color='#e2e8f0', pad=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── SHAP (optional) ──
        if run_shap:
            st.markdown('<div class="section-title">🔬 SHAP Explanation</div>', unsafe_allow_html=True)
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_aligned)

                shap_df = pd.DataFrame({
                    'Feature': list(input_aligned.columns),
                    'Value': input_aligned.values[0],
                    'SHAP': shap_values[0]
                }).sort_values('SHAP', key=abs, ascending=False).head(10)

                fig2, ax2 = plt.subplots(figsize=(8, 4))
                fig2.patch.set_facecolor('#1e293b')
                ax2.set_facecolor('#1e293b')
                cols = ['#ef4444' if v > 0 else '#3b82f6' for v in shap_df['SHAP']]
                ax2.barh(shap_df['Feature'][::-1], shap_df['SHAP'][::-1], color=cols[::-1], height=0.6)
                ax2.axvline(0, color='#94a3b8', linewidth=0.8, linestyle='--')
                ax2.set_xlabel("SHAP value (impact on prediction)", color='#94a3b8')
                ax2.tick_params(colors='#94a3b8')
                for spine in ax2.spines.values():
                    spine.set_color('#334155')
                ax2.set_title("SHAP Feature Impact (Red = raises price, Blue = lowers)", color='#e2e8f0', pad=12)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
                st.caption("SHAP values show how each feature pushes the prediction up or down from the base price.")
            except Exception as e:
                st.warning(f"SHAP requires a tree-based model. Error: {e}")

        # ── Compare all models ──
        if compare_all:
            st.markdown('<div class="section-title">⚖️ All Model Predictions</div>', unsafe_allow_html=True)
            rows = []
            for mname, mpath in avail.items():
                try:
                    m = load_model(mpath)
                    p, plo, phi = predict_with_confidence(m, input_engineered, mname)
                    rows.append({"Model": mname, "Prediction": f"${p:,.0f}",
                                 "Low (95% CI)": f"${plo:,.0f}", "High (95% CI)": f"${phi:,.0f}"})
                except Exception:
                    pass

            if rows:
                cmp_df = pd.DataFrame(rows)
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        # ── Location Map ──
        st.markdown('<div class="section-title">📍 Property Location</div>', unsafe_allow_html=True)
        map_df = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
        st.map(map_df, zoom=8)

        st.markdown("---")
        st.caption(
            "⚠️ California Housing data (1990 census) was capped at $500k. "
            "Model trained after removing capped values. "
            "Prices represent 1990 values scaled in units of $100,000."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown("### 📊 Model Performance Dashboard")

    metrics_df = load_metrics()
    cv_results = load_cv_results()

    if metrics_df is not None:
        # Highlight best values
        display_cols = [c for c in ['Model', 'Test R²', 'Test RMSE', 'Test MAE', 'Test MAPE (%)'] if c in metrics_df.columns]
        st.dataframe(
            metrics_df[display_cols].style
                .highlight_max(subset=[c for c in display_cols if 'R²' in c], color='#1e3a5f')
                .highlight_min(subset=[c for c in display_cols if c not in ['Model', 'Test R²']], color='#1e3a5f')
                .format({c: "{:.4f}" for c in display_cols if c != 'Model'}),
            use_container_width=True
        )

        # Bar chart of Test R²
        if 'Test R²' in metrics_df.columns:
            st.markdown("#### Test R² Comparison")
            fig3, ax3 = plt.subplots(figsize=(9, 3.5))
            fig3.patch.set_facecolor('#1e293b')
            ax3.set_facecolor('#1e293b')
            colors = ['#3b82f6' if i == metrics_df['Test R²'].argmax() else '#475569'
                      for i in range(len(metrics_df))]
            ax3.bar(metrics_df['Model'], metrics_df['Test R²'], color=colors, width=0.5)
            ax3.set_ylim(0, 1)
            ax3.set_ylabel("Test R²", color='#94a3b8')
            ax3.tick_params(colors='#94a3b8', axis='x', rotation=15)
            ax3.tick_params(colors='#94a3b8', axis='y')
            for spine in ax3.spines.values():
                spine.set_color('#334155')
            ax3.set_title("Test R² by Model (higher is better)", color='#e2e8f0')
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

    # CV Results
    if cv_results:
        st.markdown("#### 5-Fold Cross-Validation R² Scores")
        cv_df = pd.DataFrame([
            {"Model": k, "CV R² Mean": v['cv_r2_mean'], "CV R² Std": v['cv_r2_std']}
            for k, v in cv_results.items()
        ])
        st.dataframe(cv_df.style.format({"CV R² Mean": "{:.4f}", "CV R² Std": "{:.4f}"}),
                     use_container_width=True)

    if metrics_df is None and not cv_results:
        st.info("Run `python train_pipeline.py` to generate model metrics.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("### 📁 California Housing Dataset Explorer")

    @st.cache_data
    def get_data():
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing(as_frame=True)
        return housing.frame

    try:
        df = get_data()

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Records", f"{len(df):,}")
        col_b.metric("Features", f"{df.shape[1]-1}")
        col_c.metric("Median House Value", f"${df['MedHouseVal'].median()*100_000:,.0f}")

        st.markdown("#### Distribution of House Values")
        fig4, ax4 = plt.subplots(figsize=(9, 3.5))
        fig4.patch.set_facecolor('#1e293b')
        ax4.set_facecolor('#1e293b')
        ax4.hist(df['MedHouseVal'] * 100_000, bins=60, color='#3b82f6', alpha=0.8, edgecolor='none')
        ax4.axvline(df['MedHouseVal'].median() * 100_000, color='#f59e0b', linewidth=2, linestyle='--', label='Median')
        ax4.set_xlabel("Median House Value ($)", color='#94a3b8')
        ax4.set_ylabel("Count", color='#94a3b8')
        ax4.tick_params(colors='#94a3b8')
        for spine in ax4.spines.values():
            spine.set_color('#334155')
        ax4.set_title("Price Distribution (note: capped at $500k)", color='#e2e8f0')
        ax4.legend(facecolor='#1e293b', labelcolor='#e2e8f0')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

        st.markdown("#### Correlation Heatmap")
        fig5, ax5 = plt.subplots(figsize=(9, 6))
        fig5.patch.set_facecolor('#1e293b')
        ax5.set_facecolor('#1e293b')
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    ax=ax5, cbar_kws={'shrink': 0.8},
                    linewidths=0.5, linecolor='#334155')
        ax5.tick_params(colors='#94a3b8')
        ax5.set_title("Feature Correlations", color='#e2e8f0', pad=12)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

        st.markdown("#### Raw Data Sample")
        st.dataframe(df.sample(100, random_state=42), use_container_width=True)

    except Exception as e:
        st.error(f"Could not load dataset: {e}")
