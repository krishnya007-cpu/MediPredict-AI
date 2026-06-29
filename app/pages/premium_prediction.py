# ============================================================
# app/pages/premium_prediction.py — PROFESSIONAL VERSION
# All values in Indian Rupees (₹)
# Predict on button click only
# ============================================================

import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

APP_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.insert(0, ROOT_DIR); sys.path.insert(0, APP_DIR)

from utils.helpers import (
    load_models, transform_input, validate_input,
    fmt_inr, usd_to_inr, generate_premium_explanation,
)

# Premium category thresholds (INR)
def get_premium_category(inr: float):
    if inr < 15_000:
        return "Very Low", "#10B981"
    elif inr < 40_000:
        return "Low", "#3B82F6"
    elif inr < 80_000:
        return "Medium", "#F59E0B"
    elif inr < 1_50_000:
        return "High", "#EF4444"
    else:
        return "Very High", "#7C3AED"


def render_premium_gauge(inr_val: float) -> go.Figure:
    max_val = 5_00_000
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=inr_val,
        title={"text": "Annual Premium (₹)",
               "font": {"size": 13, "color": "#64748B"}},
        number={"prefix": "₹", "valueformat": ",.0f",
                "font": {"size": 18, "color": "#8B5CF6"}},
        gauge={
            "axis": {
                "range": [0, max_val],
                "tickvals": [0, 100000, 200000, 300000, 400000, 500000],
                "ticktext": ["₹0","₹1L","₹2L","₹3L","₹4L","₹5L"],
                "tickcolor": "#CBD5E1",
            },
            "bar": {"color": "#8B5CF6", "thickness": 0.3},
            "steps": [
                {"range": [0,        40_000],  "color": "#D1FAE5"},
                {"range": [40_000,   1_50_000],"color": "#FEF3C7"},
                {"range": [1_50_000, 5_00_000],"color": "#FEE2E2"},
            ],
            "threshold": {
                "line": {"color": "#8B5CF6", "width": 4},
                "thickness": 0.8, "value": min(inr_val, max_val),
            },
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="white",
    )
    return fig


def render_comparison_chart(predicted_inr: float) -> go.Figure:
    categories = [
        "Industry Min", "Avg Low Risk",
        "Your Premium", "Avg High Risk", "Maximum"
    ]
    values = [8_000, 25_000, predicted_inr, 2_00_000, 5_00_000]
    colors = ["#10B981","#3B82F6","#8B5CF6","#F59E0B","#EF4444"]

    fig = go.Figure()
    for cat, val, col in zip(categories, values, colors):
        fig.add_bar(
            x=[cat], y=[val],
            marker_color=col,
            text=[fmt_inr(val)],
            textposition="outside",
            name=cat,
        )
    fig.update_layout(
        title="Your Premium vs Benchmark Ranges",
        yaxis_title="Annual Premium (₹)",
        yaxis_tickformat=",.0f",
        showlegend=False,
        height=350,
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(gridcolor="#F1F5F9"),
    )
    return fig


def show(user_input: dict):
    st.markdown("""
        <h2 style='color:#1E293B; font-weight:800; margin-bottom:4px;'>
            💰 Insurance Premium Prediction
        </h2>
        <p style='color:#64748B; margin-bottom:20px;'>
            Predict your annual insurance premium in
            <b>Indian Rupees (₹)</b> using an XGBoost Regressor.
            All amounts are in INR.
        </p>
    """, unsafe_allow_html=True)

    # Validation
    errors = validate_input(user_input)
    if errors:
        for e in errors: st.error(f"❌ {e}")
        return

    # Input summary
    st.markdown("#### 📋 Policyholder Profile Summary")
    cols = st.columns(8)
    items = [
        ("Age",        user_input["age"]),
        ("BMI",        f"{user_input['bmi']:.1f}"),
        ("Smoker",     user_input["smoker"]),
        ("Chronic",    "Yes" if user_input["chronic_disease"] else "No"),
        ("Coverage",   user_input["coverage_type"]),
        ("Claims",     user_input["num_prev_claims"]),
        ("Income",     fmt_inr(user_input["annual_income_inr"])),
        ("Credit",     user_input["credit_score"]),
    ]
    for col, (label, val) in zip(cols, items):
        with col:
            st.markdown(f"""
                <div style='background:#F8FAFC; border-radius:8px;
                            padding:10px 6px; text-align:center;
                            border:1px solid #E2E8F0;'>
                    <div style='font-size:9px; color:#94A3B8;
                                text-transform:uppercase;'>{label}</div>
                    <div style='font-size:13px; font-weight:700;
                                color:#1E293B; margin-top:2px;'>{val}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Predict button
    if st.button("💰 Calculate Premium", type="primary",
                 use_container_width=True, key="premium_btn"):

        with st.spinner("Calculating your premium..."):
            models        = load_models()
            encoders      = models["encoders"]
            X             = transform_input(user_input, encoders)
            pred_usd      = float(models["premium_model"].predict(X)[0])
            pred_usd      = max(500.0, pred_usd)
            pred_inr      = usd_to_inr(pred_usd)
            pred_monthly  = pred_inr / 12
            category, cat_color = get_premium_category(pred_inr)
            expl = generate_premium_explanation(user_input, pred_inr)

        st.markdown("---")
        st.markdown("### 🎯 Premium Prediction Result")

        # Main result cards
        r1, r2, r3 = st.columns([1, 1, 1])

        with r1:
            st.markdown(f"""
                <div style='background:white; border-radius:16px;
                            padding:28px; box-shadow:0 4px 24px rgba(0,0,0,0.1);
                            border-top:5px solid #8B5CF6; text-align:center;'>
                    <div style='font-size:42px;'>💰</div>
                    <div style='font-size:12px; color:#94A3B8; margin-top:8px;
                                text-transform:uppercase; letter-spacing:1px;'>
                        Annual Premium
                    </div>
                    <div style='font-size:32px; font-weight:800;
                                color:#8B5CF6; margin-top:4px;'>
                        {fmt_inr(pred_inr)}
                    </div>
                    <div style='font-size:13px; color:#64748B; margin-top:8px;'>
                        Monthly: <strong>{fmt_inr(pred_monthly)}</strong>
                    </div>
                    <div style='margin-top:12px;'>
                        <span style='background:{cat_color}20; color:{cat_color};
                                     padding:4px 14px; border-radius:20px;
                                     font-weight:700; font-size:13px;'>
                            {category} Premium
                        </span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
                <div style='background:white; border-radius:16px;
                            padding:20px; box-shadow:0 4px 24px rgba(0,0,0,0.1);
                            height:200px;'>
                    <div style='font-size:13px; font-weight:600; color:#1E293B;
                                margin-bottom:12px;'>📊 Premium Breakdown</div>
                    <div style='display:flex; justify-content:space-between;
                                padding:8px 0; border-bottom:1px solid #F1F5F9;'>
                        <span style='font-size:13px; color:#64748B;'>Annual</span>
                        <span style='font-size:13px; font-weight:700;
                                     color:#8B5CF6;'>{fmt_inr(pred_inr)}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between;
                                padding:8px 0; border-bottom:1px solid #F1F5F9;'>
                        <span style='font-size:13px; color:#64748B;'>Monthly</span>
                        <span style='font-size:13px; font-weight:700;
                                     color:#3B82F6;'>{fmt_inr(pred_monthly)}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between;
                                padding:8px 0; border-bottom:1px solid #F1F5F9;'>
                        <span style='font-size:13px; color:#64748B;'>Quarterly</span>
                        <span style='font-size:13px; font-weight:700;
                                     color:#10B981;'>{fmt_inr(pred_inr/4)}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between;
                                padding:8px 0;'>
                        <span style='font-size:13px; color:#64748B;'>Category</span>
                        <span style='font-size:13px; font-weight:700;
                                     color:{cat_color};'>{category}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with r3:
            # Premium interpretation
            if pred_inr < 15_000:
                msg = "✅ Excellent! Very low premium — you have an outstanding risk profile."
                msg_type = "success"
            elif pred_inr < 40_000:
                msg = "✅ Good. Low premium — healthy profile with minor risk factors."
                msg_type = "success"
            elif pred_inr < 80_000:
                msg = "ℹ️ Average premium — some risk factors are present."
                msg_type = "info"
            elif pred_inr < 1_50_000:
                msg = "⚠️ Above average — multiple risk factors increasing your premium."
                msg_type = "warning"
            else:
                msg = "🚨 High premium — significant risk profile. Lifestyle changes recommended."
                msg_type = "error"

            st.markdown(f"""
                <div style='background:white; border-radius:16px;
                            padding:20px; box-shadow:0 4px 24px rgba(0,0,0,0.1);
                            height:200px; display:flex; flex-direction:column;
                            justify-content:center;'>
                    <div style='font-size:13px; font-weight:600;
                                color:#1E293B; margin-bottom:10px;'>
                        💡 Premium Assessment
                    </div>
                    <div style='font-size:13px; color:#374151;
                                line-height:1.6;'>{msg}</div>
                </div>
            """, unsafe_allow_html=True)

        # Gauge + Comparison
        st.markdown("<br>", unsafe_allow_html=True)
        g1, g2 = st.columns([1, 2])
        with g1:
            st.plotly_chart(render_premium_gauge(pred_inr),
                            use_container_width=True)
        with g2:
            st.plotly_chart(render_comparison_chart(pred_inr),
                            use_container_width=True)

        st.markdown("---")

        # Why this premium
        st.markdown("### 🤖 Why Is Your Premium This Amount?")
        reason_cols = st.columns(min(3, len(expl["reasons"])))
        for i, reason in enumerate(expl["reasons"]):
            with reason_cols[i % 3]:
                st.markdown(f"""
                    <div style='background:#FFF7ED; border-left:3px solid #F59E0B;
                                padding:10px 12px; border-radius:8px;
                                margin-bottom:8px; font-size:13px; color:#92400E;'>
                        📌 {reason}
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Recommendations
        st.markdown("### 💡 AI Recommendations to Reduce Your Premium")
        for rec in expl["recommendations"]:
            st.markdown(f"""
                <div style='background:#EFF6FF; border-left:3px solid #3B82F6;
                            padding:10px 14px; border-radius:8px;
                            margin-bottom:6px; font-size:13px; color:#1D4ED8;'>
                    💡 {rec}
                </div>
            """, unsafe_allow_html=True)

        # Download
        st.markdown("<br>", unsafe_allow_html=True)
        result_df = pd.DataFrame([{
            "Age": user_input["age"],
            "BMI": user_input["bmi"],
            "Smoker": user_input["smoker"],
            "Coverage": user_input["coverage_type"],
            "Annual_Premium_INR": round(pred_inr, 2),
            "Monthly_Premium_INR": round(pred_monthly, 2),
            "Premium_Category": category,
        }])
        st.download_button(
            "📥 Download Premium Report as CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="premium_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )