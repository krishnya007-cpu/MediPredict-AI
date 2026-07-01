# ============================================================
# app/pages/claim_prediction.py — PROFESSIONAL VERSION
# All values in INR — predict on button click only
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
    fmt_inr, usd_to_inr, generate_claim_explanation,transform_batch,
)


def get_severity(inr: float):
    if inr == 0:
        return "None",   "#10B981", "✅", "No claim expected"
    elif inr < 30_000:
        return "Low",    "#3B82F6", "🟢", "Minor medical expenses"
    elif inr < 1_00_000:
        return "Medium", "#F59E0B", "🟡", "Moderate claim expected"
    elif inr < 3_00_000:
        return "High",   "#EF4444", "🟠", "Significant claim expected"
    else:
        return "Critical","#7C3AED","🔴", "Major claim — escalate for review"


def render_claim_gauge(inr_val: float) -> go.Figure:
    max_v = 10_00_000
    sev, color, _, _ = get_severity(inr_val)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=inr_val,
        title={"text": "Expected Claim (₹)",
               "font": {"size": 13, "color": "#64748B"}},
        number={"prefix": "₹", "valueformat": ",.0f",
                "font": {"size": 18, "color": color}},
        gauge={
            "axis": {
                "range": [0, max_v],
                "tickvals": [0,200000,400000,600000,800000,1000000],
                "ticktext": ["₹0","₹2L","₹4L","₹6L","₹8L","₹10L"],
                "tickcolor": "#CBD5E1",
            },
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0,       1_00_000], "color": "#D1FAE5"},
                {"range": [1_00_000,3_00_000], "color": "#FEF3C7"},
                {"range": [3_00_000,10_00_000],"color": "#FEE2E2"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8, "value": min(inr_val, max_v),
            },
        },
    ))
    fig.update_layout(
        height=250, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="white",
    )
    return fig


def render_comparison_chart(pred_inr: float) -> go.Figure:
    labels = ["Expected (You)", "Avg Claim", "High Risk Avg", "Maximum"]
    values = [pred_inr, 75_000, 2_50_000, 10_00_000]
    colors = ["#8B5CF6", "#3B82F6", "#F59E0B", "#EF4444"]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[fmt_inr(v) for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Your Claim vs Benchmarks (₹)",
        yaxis_title="Claim Amount (₹)",
        yaxis_tickformat=",.0f",
        showlegend=False, height=300,
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#F1F5F9"),
        margin=dict(t=40, b=30, l=10, r=10),
    )
    return fig


def render_scenario_chart(user_input, claim_model, encoders) -> go.Figure:
    """What-if: claim vs number of prior claims."""
    x_vals, y_vals = [], []
    for n in range(0, 8):
        mod = user_input.copy()
        mod["num_prev_claims"] = n
        from utils.helpers import transform_input as ti
        X   = ti(mod, encoders)
        val = float(claim_model.predict(X)[0])
        x_vals.append(n)
        y_vals.append(max(0, usd_to_inr(val)))

    current = user_input["num_prev_claims"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="lines+markers",
        line=dict(color="#8B5CF6", width=3),
        marker=dict(size=8, color="#8B5CF6"),
        name="Predicted Claim",
        fill="tozeroy",
        fillcolor="rgba(139,92,246,0.1)",
    ))
    fig.add_vline(
        x=current, line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"You: {current} claims",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Scenario: Claim Amount vs Prior Claims Count",
        xaxis_title="Number of Previous Claims",
        yaxis_title="Predicted Claim (₹)",
        yaxis_tickformat=",.0f",
        height=300,
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#F1F5F9"),
        showlegend=False,
    )
    return fig


def show(user_input: dict):
    st.markdown("""
        <h2 style='color:#1E293B; font-weight:800; margin-bottom:4px;'>
            📋 Claim Amount Prediction
        </h2>
        <p style='color:#64748B; margin-bottom:20px;'>
            Predict the expected insurance claim amount in
            <b>Indian Rupees (₹)</b> using XGBoost Regressor.
            Used for <b>loss reserving</b> and <b>actuarial budgeting</b>.
        </p>
    """, unsafe_allow_html=True)

    tab_individual, tab_portfolio = st.tabs(
        [
            "👤 Individual Prediction",
            "🏢 Portfolio Analytics"
        ]
    )
    with tab_individual:
        errors = validate_input(user_input)
        if errors:
            for e in errors: st.error(f"❌ {e}")
            return

    # Input summary
    st.markdown("#### 📋 Policyholder Profile")
    cols = st.columns(6)
    items = [
        ("Age",         user_input["age"]),
        ("BMI",         f"{user_input['bmi']:.1f}"),
        ("Prior Claims",user_input["num_prev_claims"]),
        ("Hosp.",       user_input["prev_hospitalizations"]),
        ("Smoker",      user_input["smoker"]),
        ("Chronic",     "Yes" if user_input["chronic_disease"] else "No"),
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

    if st.button("📋 Predict Claim Amount", type="primary",
                 use_container_width=True, key="claim_btn"):

        with st.spinner("Calculating expected claim amount..."):
            models      = load_models()
            encoders    = models["encoders"]
            X           = transform_input(user_input, encoders)
            pred_usd    = float(models["claim_model"].predict(X)[0])
            pred_inr    = max(0.0, usd_to_inr(pred_usd))
            reserve_inr = pred_inr * 1.25
            sev, color, icon, sev_desc = get_severity(pred_inr)
            expl = generate_claim_explanation(user_input, pred_inr)

        st.markdown("---")
        st.markdown("### 🎯 Claim Prediction Result")

        r1, r2 = st.columns([1.2, 1])

        with r1:
            st.markdown(f"""
                <div style='background:white; border-radius:16px;
                            padding:28px; box-shadow:0 4px 24px rgba(0,0,0,0.1);
                            border-top:5px solid {color};'>
                    <div style='text-align:center; margin-bottom:16px;'>
                        <div style='font-size:48px;'>{icon}</div>
                        <div style='font-size:12px; color:#94A3B8; margin-top:8px;
                                    text-transform:uppercase; letter-spacing:1px;'>
                            Expected Claim Amount
                        </div>
                        <div style='font-size:36px; font-weight:800;
                                    color:{color}; margin-top:6px;'>
                            {fmt_inr(pred_inr)}
                        </div>
                        <div style='margin-top:10px;'>
                            <span style='background:{color}20; color:{color};
                                         padding:4px 14px; border-radius:20px;
                                         font-weight:700; font-size:13px;'>
                                {sev} Severity
                            </span>
                        </div>
                        <div style='font-size:13px; color:#64748B; margin-top:8px;'>
                            {sev_desc}
                        </div>
                    </div>
                    <hr style='border-color:#F1F5F9; margin:16px 0;'>
                    <div style='display:flex; justify-content:space-between;
                                padding:6px 0; border-bottom:1px solid #F1F5F9;'>
                        <span style='font-size:13px; color:#64748B;'>
                            Recommended Reserve
                        </span>
                        <span style='font-weight:700; font-size:14px;
                                     color:#F59E0B;'>{fmt_inr(reserve_inr)}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between;
                                padding:6px 0; border-bottom:1px solid #F1F5F9;'>
                        <span style='font-size:13px; color:#64748B;'>
                            Prudential Margin
                        </span>
                        <span style='font-weight:700; font-size:14px;
                                     color:#64748B;'>25%</span>
                    </div>
                    <div style='display:flex; justify-content:space-between;
                                padding:6px 0;'>
                        <span style='font-size:13px; color:#64748B;'>
                            Est. Processing Time
                        </span>
                        <span style='font-weight:700; font-size:13px;
                                     color:#3B82F6;'>
                            {expl["processing_time"]}
                        </span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with r2:
            st.plotly_chart(render_claim_gauge(pred_inr),
                            use_container_width=True)

        # Reserve KPIs
        st.markdown("<br>", unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Expected Claim",      fmt_inr(pred_inr))
        k2.metric("Reserve (25% margin)",fmt_inr(reserve_inr))
        k3.metric("Severity",            sev)
        k4.metric("Processing Time",     expl["processing_time"])

        # Charts
        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(render_comparison_chart(pred_inr),
                            use_container_width=True)
        with ch2:
            st.plotly_chart(
                render_scenario_chart(user_input,
                                      models["claim_model"], encoders),
                use_container_width=True
            )

        st.markdown("---")

        # What's driving the claim
        st.markdown("### 🤖 What Is Driving This Claim?")
        for driver in expl["drivers"]:
            st.markdown(f"""
                <div style='background:#FFF7ED; border-left:3px solid #F59E0B;
                            padding:10px 14px; border-radius:8px;
                            margin-bottom:6px; font-size:13px; color:#92400E;'>
                    📌 {driver}
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Recommendations
        st.markdown("### 💡 AI Recommendations")
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
            "Chronic_Disease": user_input["chronic_disease"],
            "Prior_Claims": user_input["num_prev_claims"],
            "Predicted_Claim_INR": round(pred_inr, 2),
            "Reserve_INR": round(reserve_inr, 2),
            "Severity": sev,
            "Processing_Time": expl["processing_time"],
        }])
        st.download_button(
            "📥 Download Claim Report as CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="claim_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with tab_portfolio:

        st.subheader("🏢 Portfolio Claim Analytics")

        st.write(
            "Upload a CSV containing multiple policyholders "
            "to estimate the total expected claims for the portfolio."
        )

        uploaded_file = st.file_uploader(
            "Upload Portfolio CSV",
            type=["csv"],
            key="portfolio_claim_csv"
        )

        if uploaded_file is not None:

            df_upload = pd.read_csv(uploaded_file)

            st.success(
                f"Loaded {len(df_upload):,} policyholders."
            )

            st.dataframe(
                df_upload.head(),
                use_container_width=True
            )

            if st.button(
                "🚀 Analyse Portfolio",
                key="portfolio_predict",
                type="primary",
            ):

                

                models = load_models()

                encoders = models["encoders"]

                X = transform_batch(
                    df_upload,
                    encoders
                )

                pred = models["claim_model"].predict(X)

                pred = np.maximum(
                    0,
                    usd_to_inr(pred)
                )

                df_upload["Predicted Claim"] = pred
                # =====================================
                # Portfolio KPIs
                # =====================================

                total_claim = df_upload["Predicted Claim"].sum()

                average_claim = df_upload["Predicted Claim"].mean()

                highest_claim = df_upload["Predicted Claim"].max()

                reserve = total_claim * 1.10
                st.markdown("---")

                st.subheader("📊 Portfolio Dashboard")

                c1, c2, c3, c4, c5 = st.columns(5)

                c1.metric("Policies", len(df_upload))
                c2.metric("Expected Claims", f"₹{total_claim:,.0f}")
                c3.metric("Average Claim", f"₹{average_claim:,.0f}")
                c4.metric("Highest Claim", f"₹{highest_claim:,.0f}")
                c5.metric("Reserve", f"₹{reserve:,.0f}")
                st.markdown("---")

                st.subheader("🔝 Highest Expected Claims")

                top10 = (
                    df_upload
                    .sort_values(
                        "Predicted Claim",
                        ascending=False
                    )
                    .head(10)
                )

                st.dataframe(
                    top10,
                    use_container_width=True
                )
                # =====================================
                # Claim Distribution Histogram
                # =====================================

                st.markdown("---")

                st.subheader("📊 Claim Distribution")

                fig = px.histogram(
                    df_upload,
                    x="Predicted Claim",
                    nbins=20,
                    title="Distribution of Predicted Claims",
                    color_discrete_sequence=["#3B82F6"],
                )

                fig.update_layout(
                    xaxis_title="Predicted Claim (₹)",
                    yaxis_title="Number of Policyholders",
                    template="plotly_white",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )