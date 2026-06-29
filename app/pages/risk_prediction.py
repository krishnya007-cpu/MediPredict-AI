# ============================================================
# app/pages/risk_prediction.py — COMPLETE FIXED VERSION
# No cuts, no placeholders, fully working
# ============================================================

import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

APP_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, APP_DIR)

from utils.helpers import (
    load_models, transform_input, transform_batch,
    validate_input, generate_risk_explanation,
    risk_color, risk_badge, fmt_percent,
)


# ==============================================================
# CHART HELPERS
# ==============================================================

def render_gauge(risk_label: str, confidence: float) -> go.Figure:
    val   = {"Low": 15, "Medium": 50, "High": 85}.get(risk_label, 50)
    color = risk_color(risk_label)
    fig   = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        title={"text": "Risk Score", "font": {"size": 14, "color": "#64748B"}},
        number={"suffix": " / 100", "font": {"size": 20, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#CBD5E1"},
            "bar":  {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0,  35],  "color": "#D1FAE5"},
                {"range": [35, 65],  "color": "#FEF3C7"},
                {"range": [65, 100], "color": "#FEE2E2"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8,
                "value": val,
            },
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="white",
    )
    return fig


def render_proba_chart(proba: np.ndarray, classes: list) -> go.Figure:
    colors = {"High": "#EF4444", "Low": "#10B981", "Medium": "#F59E0B"}
    fig = go.Figure(go.Bar(
        x=proba,
        y=classes,
        orientation="h",
        marker_color=[colors.get(c, "#6B7280") for c in classes],
        text=[f"{p:.1%}" for p in proba],
        textposition="outside",
    ))
    fig.update_layout(
        title="Class Probability",
        xaxis=dict(
            range=[0, 1],
            tickformat=".0%",
            showgrid=True,
            gridcolor="#F1F5F9",
        ),
        yaxis=dict(showgrid=False),
        height=200,
        margin=dict(t=36, b=10, l=10, r=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


# ==============================================================
# MAIN PAGE
# ==============================================================

def show(user_input: dict):

    # ── Page header ────────────────────────────────────────────
    st.markdown("""
        <h2 style='color:#1E293B; font-weight:800; margin-bottom:4px;'>
            ⚠️ Risk Level Prediction
        </h2>
        <p style='color:#64748B; margin-bottom:20px;'>
            Classify a policyholder into <b>Low</b>, <b>Medium</b>, or
            <b>High</b> risk using a Random Forest Classifier trained on
            10,000 insurance records.
        </p>
    """, unsafe_allow_html=True)

    # ── Two tabs: Individual and Bulk ──────────────────────────
    tab_ind, tab_bulk = st.tabs([
        "👤 Individual Prediction",
        "📂 Bulk CSV Prediction",
    ])

    # ==========================================================
    # TAB 1 — INDIVIDUAL PREDICTION
    # ==========================================================
    with tab_ind:

        # Validate inputs
        errors = validate_input(user_input)
        if errors:
            for e in errors:
                st.error(f"❌ {e}")
            return

        # ── Input summary card ─────────────────────────────────
        st.markdown("#### 📋 Entered Policyholder Details")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        summary_items = [
            ("Age",      user_input["age"]),
            ("BMI",      f"{user_input['bmi']:.1f}"),
            ("Smoker",   user_input["smoker"]),
            ("Chronic",  "Yes" if user_input["chronic_disease"] else "No"),
            ("Claims",   user_input["num_prev_claims"]),
            ("Exercise", f"{user_input['exercise_frequency']}d/wk"),
        ]
        for col, (label, val) in zip([c1, c2, c3, c4, c5, c6], summary_items):
            with col:
                st.markdown(f"""
                    <div style='background:#F8FAFC; border-radius:8px;
                                padding:10px; text-align:center;
                                border:1px solid #E2E8F0;'>
                        <div style='font-size:10px; color:#94A3B8;
                                    text-transform:uppercase;'>{label}</div>
                        <div style='font-size:16px; font-weight:700;
                                    color:#1E293B; margin-top:2px;'>{val}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Predict button ─────────────────────────────────────
        predict_clicked = st.button(
            "🔍 Predict Risk Level",
            type="primary",
            use_container_width=True,
            key="risk_predict_btn",
        )

        if predict_clicked:

            with st.spinner("Analysing risk profile..."):
                models    = load_models()
                encoders  = models["encoders"]
                label_enc = encoders["risk_label_encoder"]
                X         = transform_input(user_input, encoders)

                pred_enc   = models["risk_model"].predict(X)[0]
                pred_label = label_enc.inverse_transform([pred_enc])[0]
                pred_proba = models["risk_model"].predict_proba(X)[0]
                classes    = list(label_enc.classes_)
                confidence = float(max(pred_proba))
                risk_score = {"Low": 18, "Medium": 52, "High": 86}.get(pred_label, 50)

                expl = generate_risk_explanation(user_input, pred_label)

            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")

            # ── Result card + gauge ────────────────────────────
            r1, r2 = st.columns([1, 1])

            with r1:
                color = risk_color(pred_label)
                emoji = {"Low": "✅", "Medium": "⚠️", "High": "🚨"}.get(pred_label, "❓")
                st.markdown(f"""
                    <div style='background:white; border-radius:16px;
                                padding:30px; box-shadow:0 4px 24px rgba(0,0,0,0.1);
                                border-top:5px solid {color}; text-align:center;'>
                        <div style='font-size:52px;'>{emoji}</div>
                        <div style='margin-top:8px;'>{risk_badge(pred_label)}</div>
                        <div style='font-size:13px; color:#94A3B8; margin-top:12px;'>
                            Confidence
                        </div>
                        <div style='font-size:22px; font-weight:700; color:{color};'>
                            {confidence:.1%}
                        </div>
                        <div style='font-size:13px; color:#94A3B8; margin-top:8px;'>
                            Risk Score
                        </div>
                        <div style='font-size:28px; font-weight:800; color:{color};'>
                            {risk_score} / 100
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with r2:
                st.plotly_chart(
                    render_gauge(pred_label, confidence),
                    use_container_width=True,
                )

            # ── Probability chart ──────────────────────────────
            st.plotly_chart(
                render_proba_chart(pred_proba, classes),
                use_container_width=True,
            )

            st.markdown("---")

            # ── AI Explanation ─────────────────────────────────
            st.markdown("### 🤖 AI Explanation — Why This Prediction?")
            e1, e2 = st.columns(2)

            with e1:
                st.markdown("**✅ Positive Factors**")
                for item in expl["positive"]:
                    st.markdown(f"""
                        <div style='background:#F0FDF4;
                                    border-left:3px solid #10B981;
                                    padding:8px 12px; border-radius:6px;
                                    margin-bottom:6px; font-size:13px;
                                    color:#065F46;'>{item}</div>
                    """, unsafe_allow_html=True)

            with e2:
                st.markdown("**⚠️ Risk Factors**")
                if expl["negative"]:
                    for item in expl["negative"]:
                        st.markdown(f"""
                            <div style='background:#FFF7ED;
                                        border-left:3px solid #F59E0B;
                                        padding:8px 12px; border-radius:6px;
                                        margin-bottom:6px; font-size:13px;
                                        color:#92400E;'>{item}</div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style='background:#F0FDF4;
                                    border-left:3px solid #10B981;
                                    padding:10px 14px; border-radius:6px;
                                    font-size:13px; color:#065F46;'>
                            ✅ No significant risk factors detected.
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Recommendations ────────────────────────────────
            st.markdown("### 💡 AI Recommendations")

            recommendations = expl.get("recommendations", [])

            if recommendations:
                rec_cols = st.columns(min(len(recommendations), 4))

                for i, rec in enumerate(recommendations):
                    with rec_cols[i % len(rec_cols)]:
                        st.markdown(f"""
                            <div style='background:white;
                                        border-radius:10px;
                                        padding:14px;
                                        box-shadow:0 2px 8px rgba(0,0,0,0.06);
                                        border-left:3px solid #3B82F6;
                                        font-size:13px;
                                        color:#1E293B;
                                        min-height:80px;'>
                                💡 {rec}
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No recommendations available.")

            # ── Download CSV ───────────────────────────────────
            result_df = pd.DataFrame([{
                "Age":              user_input["age"],
                "Gender":           user_input["gender"],
                "BMI":              user_input["bmi"],
                "Smoker":           user_input["smoker"],
                "Chronic_Disease":  user_input["chronic_disease"],
                "Prev_Claims":      user_input["num_prev_claims"],
                "Predicted_Risk":   pred_label,
                "Confidence":       f"{confidence:.1%}",
                "Risk_Score":       risk_score,
            }])
            st.download_button(
                label="📥 Download Result as CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="risk_prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ==========================================================
    # TAB 2 — BULK CSV PREDICTION
    # ==========================================================
    with tab_bulk:

        st.markdown("""
            #### 📂 Bulk Risk Prediction
            Upload a CSV file with multiple policyholders.
            The model will predict risk level for every row.
        """)

        st.info("""
            **Required columns in your CSV:**
            age, gender, region, marital_status, num_dependents,
            bmi, smoker, chronic_disease, prev_hospitalizations,
            num_medications, exercise_frequency, coverage_type,
            policy_duration, deductible, num_prev_claims,
            annual_income, credit_score
        """)

        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            key="bulk_risk_upload",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.success(
                    f"✅ File loaded: **{uploaded_file.name}** — "
                    f"{len(df_upload):,} records"
                )
                st.dataframe(df_upload.head(5), use_container_width=True)

                bulk_btn = st.button(
                    "🔍 Predict Risk for All Rows",
                    type="primary",
                    key="bulk_risk_btn",
                )

                if bulk_btn:
                    with st.spinner("Running bulk predictions..."):
                        models    = load_models()
                        encoders  = models["encoders"]
                        label_enc = encoders["risk_label_encoder"]

                        X_batch = transform_batch(df_upload, encoders)
                        preds   = models["risk_model"].predict(X_batch)
                        probas  = models["risk_model"].predict_proba(X_batch)

                        df_result = df_upload.copy()
                        df_result["Predicted_Risk"] = label_enc.inverse_transform(preds)
                        df_result["Risk_Score"] = [
                            {"Low": 18, "Medium": 52, "High": 86}.get(r, 50)
                            for r in df_result["Predicted_Risk"]
                        ]
                        df_result["Confidence"] = [
                            f"{max(p):.1%}" for p in probas
                        ]

                    st.markdown("### 📊 Bulk Prediction Results")

                    # Summary KPIs
                    rc = df_result["Predicted_Risk"].value_counts()
                    s1, s2, s3 = st.columns(3)
                    s1.metric("🟢 Low Risk",    int(rc.get("Low", 0)))
                    s2.metric("🟡 Medium Risk", int(rc.get("Medium", 0)))
                    s3.metric("🔴 High Risk",   int(rc.get("High", 0)))

                    # Pie chart
                    fig = px.pie(
                        values=rc.values,
                        names=rc.index,
                        color=rc.index,
                        color_discrete_map={
                            "Low":    "#10B981",
                            "Medium": "#F59E0B",
                            "High":   "#EF4444",
                        },
                        title="Risk Distribution in Uploaded Dataset",
                        hole=0.45,
                    )
                    fig.update_layout(height=320, paper_bgcolor="white")
                    st.plotly_chart(fig, use_container_width=True)

                    # Full results table
                    st.markdown("**Full Predictions Table:**")
                    st.dataframe(df_result, use_container_width=True)

                    # Download button
                    csv_out = df_result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download All Predictions as CSV",
                        data=csv_out,
                        file_name="bulk_risk_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                st.info(
                    "Please make sure all required columns are present "
                    "and the file is a valid CSV."
                )
        else:
            # Upload placeholder
            st.markdown("""
                <div style='background:#F8FAFC; border:2px dashed #CBD5E1;
                            border-radius:12px; padding:40px;
                            text-align:center; color:#94A3B8;'>
                    <div style='font-size:40px;'>📂</div>
                    <div style='font-size:15px; margin:10px 0; color:#64748B;'>
                        Drop your CSV file here
                    </div>
                    <div style='font-size:12px;'>
                        Supports up to 10,000 rows for batch prediction
                    </div>
                </div>
            """, unsafe_allow_html=True)