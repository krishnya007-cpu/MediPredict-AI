# ============================================================
# app/pages/fraud_detection.py — PROFESSIONAL VERSION
# Simple, clean UI — predict on button click only
# ============================================================

import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

APP_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.insert(0, ROOT_DIR); sys.path.insert(0, APP_DIR)

from utils.helpers import (
    load_models, transform_input, validate_input,
    generate_fraud_explanation, fraud_badge, fmt_percent,
)


def render_fraud_meter(prob: float) -> go.Figure:
    pct = prob * 100
    color = "#EF4444" if prob > 0.5 else "#F59E0B" if prob > 0.3 else "#10B981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        title={"text": "Fraud Probability",
               "font": {"size": 13, "color": "#64748B"}},
        number={"suffix": "%", "valueformat": ".1f",
                "font": {"size": 22, "color": color}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0,  30],  "color": "#D1FAE5"},
                {"range": [30, 60],  "color": "#FEF3C7"},
                {"range": [60, 100], "color": "#FEE2E2"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8, "value": pct,
            },
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="white",
    )
    return fig


def show(user_input: dict):
    st.markdown("""
        <h2 style='color:#1E293B; font-weight:800; margin-bottom:4px;'>
            🚨 Insurance Fraud Detection
        </h2>
        <p style='color:#64748B; margin-bottom:20px;'>
            Detect potentially fraudulent insurance claims using an
            XGBoost Classifier with class-imbalance handling.
            Fraud costs the industry <b>₹80,000+ Crore annually</b>.
        </p>
    """, unsafe_allow_html=True)

    errors = validate_input(user_input)
    if errors:
        for e in errors: st.error(f"❌ {e}")
        return

    # Input summary
    st.markdown("#### 📋 Claim Profile Being Analysed")
    cols = st.columns(6)
    items = [
        ("Prior Claims",  user_input["num_prev_claims"]),
        ("Policy Yrs",    user_input["policy_duration"]),
        ("Hospitalisations", user_input["prev_hospitalizations"]),
        ("Medications",   user_input["num_medications"]),
        ("Credit Score",  user_input["credit_score"]),
        ("Smoker",        user_input["smoker"]),
    ]
    for col, (label, val) in zip(cols, items):
        with col:
            st.markdown(f"""
                <div style='background:#F8FAFC; border-radius:8px;
                            padding:10px; text-align:center;
                            border:1px solid #E2E8F0;'>
                    <div style='font-size:9px; color:#94A3B8;
                                text-transform:uppercase;'>{label}</div>
                    <div style='font-size:15px; font-weight:700;
                                color:#1E293B;'>{val}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Analyse Fraud Risk", type="primary",
                 use_container_width=True, key="fraud_btn"):

        with st.spinner("Analysing claim patterns..."):
            models      = load_models()
            encoders    = models["encoders"]
            X           = transform_input(user_input, encoders)
            pred        = int(models["fraud_model"].predict(X)[0])
            proba       = models["fraud_model"].predict_proba(X)[0]
            fraud_prob  = float(proba[1])
            is_fraud    = pred == 1
            expl        = generate_fraud_explanation(user_input, is_fraud, fraud_prob)

        st.markdown("---")
        st.markdown("### 🎯 Fraud Analysis Result")

        r1, r2 = st.columns([1.2, 1])

        with r1:
            bg    = "#FEE2E2" if is_fraud else "#D1FAE5"
            bc    = "#EF4444" if is_fraud else "#10B981"
            icon  = "🚨" if is_fraud else "✅"
            title = "POTENTIAL FRAUD DETECTED" if is_fraud else "CLAIM APPEARS GENUINE"

            # Risk level label
            if fraud_prob >= 0.70:
                risk_lbl, risk_col = "HIGH RISK", "#EF4444"
            elif fraud_prob >= 0.40:
                risk_lbl, risk_col = "MEDIUM RISK", "#F59E0B"
            else:
                risk_lbl, risk_col = "LOW RISK", "#10B981"

            st.markdown(f"""
                <div style='background:white; border-radius:16px;
                            padding:28px; box-shadow:0 4px 24px rgba(0,0,0,0.1);
                            border-top:5px solid {bc};'>
                    <div style='text-align:center; margin-bottom:16px;'>
                        <div style='font-size:52px;'>{icon}</div>
                        <div style='margin-top:8px;'>{fraud_badge(is_fraud)}</div>
                        <div style='font-size:22px; font-weight:800;
                                    color:{bc}; margin-top:10px;'>{title}</div>
                    </div>
                    <hr style='border-color:#F1F5F9; margin:16px 0;'>
                    <div style='display:flex; justify-content:space-between;
                                align-items:center; padding:6px 0;'>
                        <span style='color:#64748B; font-size:13px;'>
                            Fraud Probability
                        </span>
                        <span style='font-weight:700; font-size:16px;
                                     color:{bc};'>{fraud_prob:.1%}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between;
                                align-items:center; padding:6px 0;'>
                        <span style='color:#64748B; font-size:13px;'>
                            Genuine Probability
                        </span>
                        <span style='font-weight:700; font-size:16px;
                                     color:#10B981;'>{float(proba[0]):.1%}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between;
                                align-items:center; padding:6px 0;'>
                        <span style='color:#64748B; font-size:13px;'>
                            Risk Level
                        </span>
                        <span style='font-weight:700; font-size:14px;
                                     color:{risk_col};'>{risk_lbl}</span>
                    </div>
                    <hr style='border-color:#F1F5F9; margin:16px 0;'>
                    <div style='background:{bg}; border-radius:10px;
                                padding:12px 16px; font-size:13px;
                                color:#374151; border-left:4px solid {bc};'>
                        <strong>Recommended Action:</strong><br>
                        {expl["action"]}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with r2:
            st.plotly_chart(render_fraud_meter(fraud_prob),
                            use_container_width=True)

            # Simple prob bar
            fig = go.Figure(go.Bar(
                x=["Genuine", "Fraud"],
                y=[float(proba[0]), fraud_prob],
                marker_color=["#10B981", "#EF4444"],
                text=[f"{float(proba[0]):.1%}", f"{fraud_prob:.1%}"],
                textposition="outside",
            ))
            fig.update_layout(
                title="Prediction Probability",
                yaxis=dict(range=[0, 1.15], tickformat=".0%",
                           gridcolor="#F1F5F9"),
                height=240,
                paper_bgcolor="white", plot_bgcolor="white",
                showlegend=False,
                margin=dict(t=36, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Fraud indicators
        st.markdown("### 🔍 Fraud Indicator Analysis")
        f1, f2 = st.columns(2)

        with f1:
            st.markdown("**🚩 Suspicious Indicators**")
            if expl["flags"]:
                for flag in expl["flags"]:
                    st.markdown(f"""
                        <div style='background:#FEF2F2;
                                    border-left:3px solid #EF4444;
                                    padding:8px 12px; border-radius:6px;
                                    margin-bottom:6px; font-size:13px;
                                    color:#991B1B;'>⚠️ {flag}</div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style='background:#F0FDF4;
                                border-left:3px solid #10B981;
                                padding:10px 14px; border-radius:6px;
                                font-size:13px; color:#065F46;'>
                        ✅ No suspicious indicators found
                    </div>
                """, unsafe_allow_html=True)

        with f2:
            st.markdown("**✅ Normal Indicators**")
            for clr in expl["clear"]:
                st.markdown(f"""
                    <div style='background:#F0FDF4;
                                border-left:3px solid #10B981;
                                padding:8px 12px; border-radius:6px;
                                margin-bottom:6px; font-size:13px;
                                color:#065F46;'>✅ {clr}</div>
                """, unsafe_allow_html=True)

        # Download
        st.markdown("<br>", unsafe_allow_html=True)
        result_df = pd.DataFrame([{
            "Age": user_input["age"],
            "Prior_Claims": user_input["num_prev_claims"],
            "Policy_Duration": user_input["policy_duration"],
            "Fraud_Prediction": "Fraud" if is_fraud else "Genuine",
            "Fraud_Probability": f"{fraud_prob:.1%}",
            "Risk_Level": risk_lbl,
            "Recommended_Action": expl["action"].replace("**","").replace("\n",""),
        }])
        st.download_button(
            "📥 Download Fraud Analysis as CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="fraud_detection_result.csv",
            mime="text/csv",
            use_container_width=True,
        )