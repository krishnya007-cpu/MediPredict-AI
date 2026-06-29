# ============================================================
# app/pages/home.py — MEDIPREDICT AI
# Redesigned layout:
#   Hero (compact, 180px) → Platform Status → AI Prediction
#   Dashboard → Dataset Upload → Results → Charts → Footer
# ML logic, prediction code, CSV processing unchanged.
# ============================================================

import os, sys
from typing import Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

APP_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, APP_DIR)

from utils.helpers import (
    load_models, transform_input, transform_batch,
    usd_to_inr, fmt_inr, fmt_inr_plain, USD_TO_INR,
    FEATURE_COLS, risk_color, get_logo_svg, validate_input,
    section_header, white_card,
)

DATA_DIR    = os.path.join(ROOT_DIR, "data")
RISK_COLORS = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}


# ==============================================================
# DATA UTILITIES  (unchanged — same ML logic)
# ==============================================================

@st.cache_data(show_spinner=False)
def load_default_dataset():
    p = os.path.join(DATA_DIR, "insurance_data.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def get_first_row_input(df: pd.DataFrame) -> dict:
    row = df.iloc[0]

    def g(col, d):
        return row[col] if col in df.columns else d

    inc_raw = float(g("annual_income", 600_000))
    inc_usd = inc_raw / USD_TO_INR if inc_raw > 10_000 else inc_raw
    inc_inr = inc_raw if inc_raw > 10_000 else inc_raw * USD_TO_INR
    return {
        "age":                   int(g("age", 35)),
        "gender":                str(g("gender", "Male")),
        "region":                str(g("region", "Northeast")),
        "marital_status":        str(g("marital_status", "Single")),
        "num_dependents":        int(g("num_dependents", 0)),
        "bmi":                   float(g("bmi", 25.0)),
        "smoker":                str(g("smoker", "No")),
        "chronic_disease":       int(g("chronic_disease", 0)),
        "prev_hospitalizations": int(g("prev_hospitalizations", 0)),
        "num_medications":       int(g("num_medications", 0)),
        "exercise_frequency":    int(g("exercise_frequency", 3)),
        "coverage_type":         str(g("coverage_type", "Standard")),
        "policy_duration":       int(g("policy_duration", 5)),
        "deductible":            int(g("deductible", 1000)),
        "num_prev_claims":       int(g("num_prev_claims", 0)),
        "annual_income":         inc_usd,
        "annual_income_inr":     inc_inr,
        "credit_score":          int(g("credit_score", 680)),
    }


def run_predictions(ui: dict) -> dict:
    models    = load_models()
    encoders  = models["encoders"]
    label_enc = encoders["risk_label_encoder"]
    X         = transform_input(ui, encoders)

    risk_enc   = models["risk_model"].predict(X)[0]
    risk_label = label_enc.inverse_transform([risk_enc])[0]
    risk_proba = models["risk_model"].predict_proba(X)[0]

    prem_usd = float(models["premium_model"].predict(X)[0])
    prem_inr = usd_to_inr(max(500.0, prem_usd))

    fraud_pred  = int(models["fraud_model"].predict(X)[0])
    fraud_proba = models["fraud_model"].predict_proba(X)[0]
    fraud_prob  = float(fraud_proba[1])

    claim_usd = float(models["claim_model"].predict(X)[0])
    claim_inr = usd_to_inr(max(0.0, claim_usd))

    return {
        "risk_label": risk_label,
        "risk_conf":  float(max(risk_proba)),
        "prem_inr":   prem_inr,
        "fraud_prob": fraud_prob,
        "is_fraud":   fraud_pred == 1,
        "claim_inr":  claim_inr,
    }


def run_batch(df: pd.DataFrame) -> pd.DataFrame:
    models    = load_models()
    encoders  = models["encoders"]
    label_enc = encoders["risk_label_encoder"]

    df_m = df.copy()
    if "annual_income" in df_m.columns and df_m["annual_income"].mean() > 10_000:
        df_m["annual_income"] = df_m["annual_income"] / USD_TO_INR

    X_b = transform_batch(df_m, encoders)
    out = df.copy()

    out["Predicted_Risk"]        = label_enc.inverse_transform(
        models["risk_model"].predict(X_b))
    out["Risk_Confidence"]       = (
        models["risk_model"].predict_proba(X_b).max(axis=1) * 100).round(1)
    out["Premium_INR"]           = (
        np.clip(models["premium_model"].predict(X_b), 500, None) * USD_TO_INR
    ).round(0)
    out["Fraud_Probability_Pct"] = (
        models["fraud_model"].predict_proba(X_b)[:, 1] * 100).round(1)
    out["Claim_INR"]             = (
        np.clip(models["claim_model"].predict(X_b), 0, None) * USD_TO_INR
    ).round(0)
    return out


# ==============================================================
# SECTION 1 — COMPACT HERO + PLATFORM STATUS
# ==============================================================

def render_hero():
    hero_col, status_col = st.columns([3, 1], gap="large")

    with hero_col:
        # Pre-render SVG safely outside f-string
        logo = get_logo_svg(54)

        hero_html = (
            '<div style="'
            'background:linear-gradient(135deg,#0F172A 0%,#1E3A8A 55%,#2563EB 100%);'
            'border-radius:22px;'
            'padding:30px 36px 28px;'
            'box-shadow:0 12px 40px rgba(37,99,235,0.28);'
            'border:1px solid rgba(255,255,255,0.09);'
            'position:relative;'
            'overflow:hidden;'
            'min-height:185px;">'

            # Decorative circles
            '<div style="position:absolute;top:-30px;right:-30px;'
            'width:140px;height:140px;'
            'background:rgba(37,99,235,0.18);border-radius:50%;"></div>'
            '<div style="position:absolute;bottom:-40px;right:100px;'
            'width:110px;height:110px;'
            'background:rgba(124,58,237,0.12);border-radius:50%;"></div>'

            # Logo + Title row
            '<div style="display:flex;align-items:center;'
            'gap:16px;margin-bottom:14px;">'
            '<div style="flex-shrink:0;">'
            + logo +
            '</div>'
            '<div>'
            '<div style="font-size:30px;font-weight:900;color:white;'
            'letter-spacing:2.5px;line-height:1.1;text-transform:uppercase;">'
            'MEDIPREDICT AI'
            '</div>'
            '<div style="font-size:13px;color:#93C5FD;font-weight:500;'
            'margin-top:5px;letter-spacing:0.3px;">'
            'AI Powered Healthcare Insurance Analytics Platform'
            '</div>'
            '</div>'
            '</div>'

            # Description
            '<div style="color:#CBD5E1;font-size:13px;line-height:1.65;'
            'margin-bottom:16px;max-width:580px;">'
            'A comprehensive AI platform for healthcare insurance risk '
            'assessment, premium prediction, fraud detection and claim '
            'prediction.'
            '</div>'

            # Badges
            '<div style="display:flex;gap:8px;flex-wrap:wrap;">'

            '<span style="background:rgba(255,255,255,0.11);color:#E2E8F0;'
            'padding:5px 13px;border-radius:30px;font-size:11px;'
            'font-weight:500;border:1px solid rgba(255,255,255,0.16);">'
            '&#10003; Machine Learning</span>'

            '<span style="background:rgba(255,255,255,0.11);color:#E2E8F0;'
            'padding:5px 13px;border-radius:30px;font-size:11px;'
            'font-weight:500;border:1px solid rgba(255,255,255,0.16);">'
            '&#10003; Predictive Analytics</span>'

            '<span style="background:rgba(255,255,255,0.11);color:#E2E8F0;'
            'padding:5px 13px;border-radius:30px;font-size:11px;'
            'font-weight:500;border:1px solid rgba(255,255,255,0.16);">'
            '&#10003; Healthcare Insurance</span>'

            '<span style="background:rgba(255,255,255,0.11);color:#E2E8F0;'
            'padding:5px 13px;border-radius:30px;font-size:11px;'
            'font-weight:500;border:1px solid rgba(255,255,255,0.16);">'
            '&#10003; AI Dashboard</span>'

            '<span style="background:rgba(255,255,255,0.11);color:#E2E8F0;'
            'padding:5px 13px;border-radius:30px;font-size:11px;'
            'font-weight:500;border:1px solid rgba(255,255,255,0.16);">'
            '&#10003; Interactive Reports</span>'

            '</div>'
            '</div>'
        )

        st.markdown(hero_html, unsafe_allow_html=True)

    with status_col:
        # Header
        st.markdown(
            '<div style="background:white;border-radius:20px;'
            'padding:22px 20px;'
            'box-shadow:0 4px 20px rgba(37,99,235,0.09);'
            'border:1px solid #E2E8F0;">',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:10px;font-weight:800;color:#94A3B8;'
            'text-transform:uppercase;letter-spacing:1.5px;'
            'margin-bottom:16px;text-align:center;">'
            'Platform Status'
            '</div>',
            unsafe_allow_html=True,
        )

        # Status rows — each as its own st.markdown call
        rows = [
            ("🤖 AI Models",   "Online", "#10B981", True),
            ("📊 Analytics",   "Active", "#2563EB", True),
            ("🔒 Security",    "Secure", "#10B981", True),
            ("⚡ Performance", "Fast",   "#F59E0B", False),
        ]
        for label, status, color, has_border in rows:
            border = "border-bottom:1px solid #F1F5F9;" if has_border else ""
            st.markdown(
                '<div style="display:flex;justify-content:space-between;'
                'align-items:center;padding:9px 0;' + border + '">'
                '<span style="font-size:12px;color:#475569;font-weight:500;">'
                + label +
                '</span>'
                '<span style="font-size:11px;color:' + color + ';font-weight:700;'
                'background:' + color + '1A;padding:2px 10px;'
                'border-radius:20px;">'
                + status +
                '</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)
# ==============================================================
# SECTION 2 — AI PREDICTION DASHBOARD  (above upload)
# ==============================================================

def render_prediction_dashboard():
    section_header(
        "🤖", "AI Prediction Dashboard",
        "Fill sidebar inputs and click any button to run a prediction",
    )

    r1c1, r1c2 = st.columns(2, gap="large")
    r2c1, r2c2 = st.columns(2, gap="large")

    card_defs = [
        (r1c1, "⚠️", "Risk Prediction",
         "Random Forest Classifier",
         "Classify policyholder into Low, Medium or High risk.",
         "#2563EB", "home_risk_btn"),
        (r1c2, "💰", "Premium Prediction",
         "XGBoost Regressor",
         "Predict annual insurance premium in Indian Rupees.",
         "#7C3AED", "home_prem_btn"),
        (r2c1, "🚨", "Fraud Detection",
         "XGBoost Classifier",
         "Detect potentially fraudulent insurance claims.",
         "#EF4444", "home_fraud_btn"),
        (r2c2, "📄", "Claim Prediction",
         "XGBoost Regressor",
         "Estimate expected claim amount for loss reserving.",
         "#F59E0B", "home_claim_btn"),
    ]

    clicked = {}
    for col, icon, title, model, desc, color, key in card_defs:
        with col:
            st.markdown(
                f"""
                <div style="background:white; border-radius:20px;
                            padding:24px 24px 10px;
                            box-shadow:0 4px 20px rgba(0,0,0,0.07);
                            border:1px solid #F1F5F9;
                            border-top:4px solid {color};
                            margin-bottom:4px;
                            transition:box-shadow 0.2s;">
                    <div style="display:flex; align-items:center;
                                gap:14px; margin-bottom:12px;">
                        <div style="background:{color}18; width:48px;
                                    height:48px; border-radius:14px;
                                    display:flex; align-items:center;
                                    justify-content:center;
                                    font-size:26px;
                                    box-shadow:0 2px 8px {color}30;">
                            {icon}
                        </div>
                        <div>
                            <div style="font-size:16px; font-weight:800;
                                        color:#0F172A; line-height:1.2;">
                                {title}
                            </div>
                            <div style="font-size:11px; color:#94A3B8;
                                        margin-top:3px; font-weight:500;">
                                {model}
                            </div>
                        </div>
                    </div>
                    <div style="font-size:13px; color:#64748B;
                                line-height:1.55; margin-bottom:16px;">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            clicked[key] = st.button(
                f"▶  Run {title}",
                type="primary",
                use_container_width=True,
                key=key,
            )

    # Record which button was clicked
    for key, was_clicked in clicked.items():
        if was_clicked:
            mapping = {
                "home_risk_btn":  "risk",
                "home_prem_btn":  "premium",
                "home_fraud_btn": "fraud",
                "home_claim_btn": "claim",
            }
            st.session_state["home_predict"] = mapping[key]

    # Show result if any button has been clicked
    if "home_predict" in st.session_state:
        _show_prediction_result(st.session_state["home_predict"])


def _show_prediction_result(ptype: str):
    """Run prediction from sidebar inputs and show result card + gauge."""
    ui = st.session_state.get("current_user_input")
    if ui is None:
        st.info("ℹ️ Fill in the sidebar inputs first.")
        return

    errs = validate_input(ui)
    if errs:
        for e in errs:
            st.error(f"❌ {e}")
        return

    with st.spinner("🤖 Running AI prediction..."):
        try:
            r = run_predictions(ui)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0F172A,#1E3A8A);
                    border-radius:16px; padding:13px 22px; margin:20px 0 14px;">
            <span style="color:white; font-size:15px; font-weight:700;">
                🎯 Prediction Result
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    g1, g2 = st.columns(2, gap="large")

    if ptype == "risk":
        color = risk_color(r["risk_label"])
        score = {"Low": 18, "Medium": 52, "High": 86}.get(r["risk_label"], 50)
        icon  = {"Low": "✅", "Medium": "⚠️", "High": "🚨"}.get(r["risk_label"], "❓")
        fig   = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Risk Score", "font": {"size": 13, "color": "#64748B"}},
            number={"suffix": " / 100", "font": {"size": 20, "color": color}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": color, "thickness": 0.28},
                "steps": [
                    {"range": [0, 35],   "color": "#D1FAE5"},
                    {"range": [35, 65],  "color": "#FEF3C7"},
                    {"range": [65, 100], "color": "#FEE2E2"},
                ],
            },
        ))
        fig.update_layout(height=230,
                          margin=dict(t=36, b=0, l=16, r=16),
                          paper_bgcolor="white")
        with g1:
            st.markdown(
                f"""
                <div style="background:white; border-radius:20px; padding:28px;
                            box-shadow:0 4px 24px rgba(0,0,0,0.08);
                            border-top:5px solid {color}; text-align:center;">
                    <div style="font-size:46px;">{icon}</div>
                    <div style="font-size:24px; font-weight:900; color:{color};
                                margin-top:8px;">{r["risk_label"]} Risk</div>
                    <div style="font-size:13px; color:#64748B; margin-top:8px;">
                        Confidence: <strong>{r["risk_conf"]:.1%}</strong>
                        &nbsp;|&nbsp; Score: <strong>{score} / 100</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with g2:
            st.plotly_chart(fig, use_container_width=True)
        st.progress(score / 100)

    elif ptype == "premium":
        monthly = r["prem_inr"] / 12
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=min(r["prem_inr"], 500_000),
            title={"text": "Annual Premium", "font": {"size": 13}},
            number={"prefix": "Rs ", "valueformat": ",.0f"},
            gauge={
                "axis": {"range": [0, 500_000]},
                "bar":  {"color": "#7C3AED", "thickness": 0.28},
                "steps": [
                    {"range": [0,       40_000],  "color": "#D1FAE5"},
                    {"range": [40_000,  150_000], "color": "#FEF3C7"},
                    {"range": [150_000, 500_000], "color": "#FEE2E2"},
                ],
            },
        ))
        fig.update_layout(height=230,
                          margin=dict(t=36, b=0, l=16, r=16),
                          paper_bgcolor="white")
        with g1:
            st.markdown(
                f"""
                <div style="background:white; border-radius:20px; padding:28px;
                            box-shadow:0 4px 24px rgba(0,0,0,0.08);
                            border-top:5px solid #7C3AED; text-align:center;">
                    <div style="font-size:46px;">💰</div>
                    <div style="font-size:24px; font-weight:900; color:#7C3AED;
                                margin-top:8px;">{fmt_inr_plain(r["prem_inr"])}</div>
                    <div style="font-size:13px; color:#64748B; margin-top:8px;">
                        Annual Premium &nbsp;|&nbsp;
                        Monthly: <strong>{fmt_inr_plain(monthly)}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with g2:
            st.plotly_chart(fig, use_container_width=True)

    elif ptype == "fraud":
        fc   = "#EF4444" if r["is_fraud"] else "#10B981"
        fst  = "Potential Fraud" if r["is_fraud"] else "Likely Genuine"
        icon = "🚨" if r["is_fraud"] else "✅"
        fig  = go.Figure(go.Indicator(
            mode="gauge+number",
            value=r["fraud_prob"] * 100,
            title={"text": "Fraud Probability", "font": {"size": 13}},
            number={"suffix": "%", "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": fc, "thickness": 0.28},
                "steps": [
                    {"range": [0, 30],   "color": "#D1FAE5"},
                    {"range": [30, 60],  "color": "#FEF3C7"},
                    {"range": [60, 100], "color": "#FEE2E2"},
                ],
            },
        ))
        fig.update_layout(height=230,
                          margin=dict(t=36, b=0, l=16, r=16),
                          paper_bgcolor="white")
        with g1:
            st.markdown(
                f"""
                <div style="background:white; border-radius:20px; padding:28px;
                            box-shadow:0 4px 24px rgba(0,0,0,0.08);
                            border-top:5px solid {fc}; text-align:center;">
                    <div style="font-size:46px;">{icon}</div>
                    <div style="font-size:24px; font-weight:900; color:{fc};
                                margin-top:8px;">{fst}</div>
                    <div style="font-size:13px; color:#64748B; margin-top:8px;">
                        Fraud Probability:
                        <strong>{r["fraud_prob"]:.1%}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with g2:
            st.plotly_chart(fig, use_container_width=True)
        st.progress(r["fraud_prob"])

    elif ptype == "claim":
        sev = (
            "Low"    if r["claim_inr"] < 30_000   else
            "Medium" if r["claim_inr"] < 1_00_000 else
            "High"
        )
        cc      = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}[sev]
        icon    = {"Low": "🟢",      "Medium": "🟡",      "High": "🔴"}[sev]
        reserve = r["claim_inr"] * 1.25
        fig     = go.Figure(go.Indicator(
            mode="gauge+number",
            value=min(r["claim_inr"], 1_000_000),
            title={"text": "Expected Claim", "font": {"size": 13}},
            number={"prefix": "Rs ", "valueformat": ",.0f"},
            gauge={
                "axis": {"range": [0, 1_000_000]},
                "bar":  {"color": cc, "thickness": 0.28},
                "steps": [
                    {"range": [0,         1_00_000],  "color": "#D1FAE5"},
                    {"range": [1_00_000,  3_00_000],  "color": "#FEF3C7"},
                    {"range": [3_00_000, 1_000_000],  "color": "#FEE2E2"},
                ],
            },
        ))
        fig.update_layout(height=230,
                          margin=dict(t=36, b=0, l=16, r=16),
                          paper_bgcolor="white")
        with g1:
            st.markdown(
                f"""
                <div style="background:white; border-radius:20px; padding:28px;
                            box-shadow:0 4px 24px rgba(0,0,0,0.08);
                            border-top:5px solid {cc}; text-align:center;">
                    <div style="font-size:46px;">{icon}</div>
                    <div style="font-size:24px; font-weight:900; color:{cc};
                                margin-top:8px;">{fmt_inr_plain(r["claim_inr"])}</div>
                    <div style="font-size:13px; color:#64748B; margin-top:8px;">
                        Severity: <strong>{sev}</strong> &nbsp;|&nbsp;
                        Reserve: <strong>{fmt_inr_plain(reserve)}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with g2:
            st.plotly_chart(fig, use_container_width=True)


# ==============================================================
# SECTION 3 — DATASET UPLOAD  (full width, no extras)
# ==============================================================

def render_upload_section() -> Optional[pd.DataFrame]:
    section_header(
        "📂", "Dataset Upload",
        "Upload your insurance CSV for batch predictions and analytics",
    )

    # Full-width styled drop-zone card
    st.markdown(
        """
        <div style="background:white; border-radius:20px;
                    border:2px dashed #2563EB; padding:30px 36px 18px;
                    box-shadow:0 4px 20px rgba(37,99,235,0.07);
                    text-align:center;">
            <div style="font-size:44px; margin-bottom:10px;">📊</div>
            <div style="font-size:16px; font-weight:700; color:#1E293B;
                        margin-bottom:5px;">
                Drag and Drop Your CSV File
            </div>
            <div style="font-size:13px; color:#94A3B8; margin-bottom:6px;">
                Supports .csv files &nbsp;•&nbsp; Max 200 MB
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="home_csv_upload",
        label_visibility="collapsed",
    )

    df_raw = None

    if uploaded is not None:
        try:
            df_raw  = pd.read_csv(uploaded)
            size_kb = uploaded.size / 1024
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#F0FDF4,#DCFCE7);
                            border-radius:14px; padding:16px 24px;
                            border:1px solid #BBF7D0; margin-top:14px;
                            display:flex; gap:36px; flex-wrap:wrap;
                            align-items:center;">
                    <div>
                        <div style="font-size:9px; color:#16A34A; font-weight:800;
                                    text-transform:uppercase;
                                    letter-spacing:1px;">File</div>
                        <div style="font-size:14px; font-weight:700;
                                    color:#166534;">{uploaded.name}</div>
                    </div>
                    <div>
                        <div style="font-size:9px; color:#16A34A; font-weight:800;
                                    text-transform:uppercase;
                                    letter-spacing:1px;">Rows</div>
                        <div style="font-size:14px; font-weight:700;
                                    color:#166534;">{df_raw.shape[0]:,}</div>
                    </div>
                    <div>
                        <div style="font-size:9px; color:#16A34A; font-weight:800;
                                    text-transform:uppercase;
                                    letter-spacing:1px;">Columns</div>
                        <div style="font-size:14px; font-weight:700;
                                    color:#166534;">{df_raw.shape[1]}</div>
                    </div>
                    <div>
                        <div style="font-size:9px; color:#16A34A; font-weight:800;
                                    text-transform:uppercase;
                                    letter-spacing:1px;">Size</div>
                        <div style="font-size:14px; font-weight:700;
                                    color:#166534;">{size_kb:.1f} KB</div>
                    </div>
                    <div>
                        <div style="font-size:9px; color:#16A34A; font-weight:800;
                                    text-transform:uppercase;
                                    letter-spacing:1px;">Status</div>
                        <div style="font-size:14px; font-weight:700;
                                    color:#166534;">✅ Ready</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"❌ Could not read file: {e}")

    return df_raw


# ==============================================================
# SECTION 4 — PREDICTION RESULT CARDS  (from dataset first row)
# ==============================================================

def render_prediction_cards(results: dict):
    section_header(
        "🎯", "AI Prediction Results",
        "Predictions generated from the first record of your dataset",
    )

    c1, c2, c3, c4 = st.columns(4, gap="large")
    rc  = risk_color(results["risk_label"])
    rem = {"Low": "✅", "Medium": "⚠️", "High": "🚨"}.get(results["risk_label"], "❓")

    clm_sev = (
        "Low"    if results["claim_inr"] < 30_000   else
        "Medium" if results["claim_inr"] < 1_00_000 else
        "High"
    )
    clm_col = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}[clm_sev]

    cards = [
        (c1, rem, "Risk Prediction",
         "Random Forest",
         f"{results['risk_label']} Risk",
         f"Confidence: {results['risk_conf']:.1%}",
         rc),
        (c2, "💰", "Premium Prediction",
         "XGBoost Regressor",
         fmt_inr_plain(results["prem_inr"]),
         f"Monthly: {fmt_inr_plain(results['prem_inr']/12)}",
         "#7C3AED"),
        (c3,
         "🚨" if results["is_fraud"] else "✅",
         "Fraud Detection",
         "XGBoost Classifier",
         "Potential Fraud" if results["is_fraud"] else "Likely Genuine",
         f"Probability: {results['fraud_prob']:.1%}",
         "#EF4444" if results["is_fraud"] else "#10B981"),
        (c4, "📄", "Claim Prediction",
         "XGBoost Regressor",
         fmt_inr_plain(results["claim_inr"]),
         f"Severity: {clm_sev}",
         clm_col),
    ]

    for col, icon, title, model, value, sub, color in cards:
        with col:
            st.markdown(
                f"""
                <div style="background:white; border-radius:20px;
                            padding:22px 20px;
                            box-shadow:0 6px 24px rgba(0,0,0,0.08);
                            border:1px solid #F1F5F9;
                            border-top:4px solid {color};
                            min-height:180px;">
                    <div style="display:flex; justify-content:space-between;
                                align-items:flex-start; margin-bottom:12px;">
                        <div style="font-size:10px; font-weight:800;
                                    color:#94A3B8; text-transform:uppercase;
                                    letter-spacing:0.9px;">{title}</div>
                        <div style="font-size:26px;">{icon}</div>
                    </div>
                    <div style="font-size:10px; color:#CBD5E1;
                                margin-bottom:4px;">{model}</div>
                    <div style="font-size:20px; font-weight:800;
                                color:{color}; margin-bottom:6px;">{value}</div>
                    <div style="font-size:12px; color:#64748B;">{sub}</div>
                    <div style="background:{color}18; border-radius:99px;
                                height:4px; margin-top:16px; overflow:hidden;">
                        <div style="background:{color}; height:100%;
                                    width:76%; border-radius:99px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================
# SECTION 5 — INTERACTIVE DASHBOARD CHARTS
# ==============================================================

def render_dashboard_charts(df_results: pd.DataFrame):
    section_header(
        "📊", "Interactive Dashboard Analytics",
        "Visual analysis of all batch prediction results",
    )

    t1, t2, t3, t4 = st.tabs([
        "📊 Risk Analysis",
        "💰 Premium Analysis",
        "🚨 Fraud Analysis",
        "📄 Claim Analysis",
    ])

    chart_layout = dict(
        paper_bgcolor="white",
        plot_bgcolor="#FAFBFF",
        margin=dict(t=42, b=22, l=22, r=22),
    )

    with t1:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            if "Predicted_Risk" in df_results.columns:
                rc = df_results["Predicted_Risk"].value_counts().reset_index()
                rc.columns = ["Risk", "Count"]
                fig = px.pie(rc, names="Risk", values="Count",
                             color="Risk", color_discrete_map=RISK_COLORS,
                             title="Risk Level Distribution", hole=0.5)
                fig.update_layout(height=330, **chart_layout)
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "Predicted_Risk" in df_results.columns:
                rc2 = df_results["Predicted_Risk"].value_counts().reset_index()
                rc2.columns = ["Risk Level", "Count"]
                fig2 = px.bar(rc2, x="Risk Level", y="Count",
                              color="Risk Level",
                              color_discrete_map=RISK_COLORS,
                              title="Risk Count by Level", text="Count")
                fig2.update_traces(textposition="outside")
                fig2.update_layout(height=330, showlegend=False,
                                   **chart_layout)
                st.plotly_chart(fig2, use_container_width=True)

        if "Risk_Confidence" in df_results.columns:
            fig3 = px.histogram(
                df_results, x="Risk_Confidence", nbins=30,
                title="Risk Confidence Distribution (%)",
                color_discrete_sequence=["#2563EB"],
            )
            fig3.update_layout(height=290, **chart_layout)
            st.plotly_chart(fig3, use_container_width=True)

    with t2:
        if "Premium_INR" in df_results.columns:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                fig = px.histogram(
                    df_results, x="Premium_INR", nbins=35,
                    title="Premium Distribution (Rs)",
                    color_discrete_sequence=["#7C3AED"],
                )
                fig.update_layout(height=310, **chart_layout)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                if "Predicted_Risk" in df_results.columns:
                    fig2 = px.box(
                        df_results, x="Predicted_Risk", y="Premium_INR",
                        color="Predicted_Risk",
                        color_discrete_map=RISK_COLORS,
                        title="Premium by Risk Level (Rs)",
                    )
                    fig2.update_layout(height=310, showlegend=False,
                                       **chart_layout)
                    st.plotly_chart(fig2, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Premium",
                      fmt_inr_plain(df_results["Premium_INR"].mean()))
            m2.metric("Min Premium",
                      fmt_inr_plain(df_results["Premium_INR"].min()))
            m3.metric("Max Premium",
                      fmt_inr_plain(df_results["Premium_INR"].max()))

    with t3:
        if "Fraud_Probability_Pct" in df_results.columns:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                fig = px.histogram(
                    df_results, x="Fraud_Probability_Pct", nbins=30,
                    title="Fraud Probability Distribution (%)",
                    color_discrete_sequence=["#EF4444"],
                )
                fig.update_layout(height=310, **chart_layout)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fraud_cats = pd.cut(
                    df_results["Fraud_Probability_Pct"],
                    bins=[0, 30, 60, 100],
                    labels=["Low", "Medium", "High"],
                )
                fc_counts = fraud_cats.value_counts().reset_index()
                fc_counts.columns = ["Category", "Count"]
                fig2 = px.pie(
                    fc_counts, names="Category", values="Count",
                    color="Category",
                    color_discrete_map={
                        "Low": "#10B981",
                        "Medium": "#F59E0B",
                        "High": "#EF4444",
                    },
                    title="Fraud Risk Categories", hole=0.5,
                )
                fig2.update_layout(height=310, **chart_layout)
                st.plotly_chart(fig2, use_container_width=True)

    with t4:
        if "Claim_INR" in df_results.columns:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                fig = px.histogram(
                    df_results[df_results["Claim_INR"] > 0],
                    x="Claim_INR", nbins=35,
                    title="Claim Amount Distribution (Rs)",
                    color_discrete_sequence=["#F59E0B"],
                )
                fig.update_layout(height=310, **chart_layout)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                if "Premium_INR" in df_results.columns:
                    sample = df_results.sample(
                        min(500, len(df_results)), random_state=42
                    )
                    fig2 = px.scatter(
                        sample, x="Premium_INR", y="Claim_INR",
                        color=("Predicted_Risk"
                               if "Predicted_Risk" in sample.columns
                               else None),
                        color_discrete_map=RISK_COLORS,
                        title="Premium vs Claim (Rs)",
                        opacity=0.55,
                    )
                    fig2.update_layout(height=310, **chart_layout)
                    st.plotly_chart(fig2, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Claim",
                      fmt_inr_plain(df_results["Claim_INR"].mean()))
            m2.metric("Zero Claims",
                      f"{(df_results['Claim_INR'] == 0).sum():,}")
            m3.metric("Max Claim",
                      fmt_inr_plain(df_results["Claim_INR"].max()))

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="📥 Download Complete Prediction Results (CSV)",
        data=df_results.to_csv(index=False).encode("utf-8"),
        file_name="medipredict_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ==============================================================
# FOOTER
# ==============================================================

def render_footer():
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, fc, _ = st.columns([1, 3, 1])
    with fc:
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#0F172A,#1E3A8A);
                        border-radius:20px; padding:30px; text-align:center;
                        box-shadow:0 8px 32px rgba(15,23,42,0.28);">
                <div style="width:48px; height:48px; margin:0 auto 12px;">
                    {get_logo_svg(48)}
                </div>
                <div style="font-weight:900; color:#E2E8F0; font-size:16px;
                            letter-spacing:3px; text-transform:uppercase;">
                    MEDIPREDICT AI
                </div>
                <div style="font-size:12px; color:#60A5FA; margin-top:5px;">
                    AI-Powered Healthcare Insurance Analytics Platform
                </div>
                <div style="font-size:11px; color:#475569; margin-top:5px;">
                    Developed using Python, Streamlit and Machine Learning
                </div>
                <hr style="border:1px solid rgba(255,255,255,0.08);
                           margin:14px 0;">
                <div style="font-size:10px; color:#334155;">
                    2026 MEDIPREDICT AI — All Rights Reserved
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================
# MAIN SHOW  —  fixed layout order
# ==============================================================

def show():
    # ── 1. Compact Hero + Platform Status (same row) ───────────
    render_hero()

    # ── 2. AI Prediction Dashboard (above upload) ──────────────
    render_prediction_dashboard()

    # ── 3. Dataset Upload (full width) ─────────────────────────
    df_raw = render_upload_section()

    # ── 4–6. Only shown after upload ───────────────────────────
    if df_raw is not None:
        with st.spinner("Preparing dataset..."):
            df_clean = clean_df(df_raw)

        # 4. Prediction result cards from first row
        try:
            with st.spinner("Running AI predictions..."):
                ui      = get_first_row_input(df_clean)
                results = run_predictions(ui)
            render_prediction_cards(results)
        except Exception as e:
            st.error(f"Prediction error: {e}\n\n"
                     "Run: `python notebooks/eda_and_training.py`")
            render_footer()
            return

        # 5. Batch predictions + dashboard charts
        has_required = all(c in df_clean.columns for c in FEATURE_COLS)
        if has_required:
            try:
                with st.spinner("Running batch predictions for charts..."):
                    df_results = run_batch(df_clean)
                render_dashboard_charts(df_results)
            except Exception as e:
                st.warning(f"Batch prediction skipped: {e}")

    # ── 6. Footer (always) ─────────────────────────────────────
    render_footer()