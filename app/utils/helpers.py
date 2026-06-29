# ============================================================
# app/utils/helpers.py — MEDIPREDICT AI — COMPLETE FILE
# Contains: models, transforms, formatting, validation,
#           explanations, section_header, white_card, sidebar
# ============================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, APP_DIR)

MODELS_DIR = os.path.join(ROOT_DIR, "models")
USD_TO_INR = 83.0

FEATURE_COLS = [
    "age", "gender", "region", "marital_status", "num_dependents",
    "bmi", "smoker", "chronic_disease", "prev_hospitalizations",
    "num_medications", "exercise_frequency", "coverage_type",
    "policy_duration", "deductible", "num_prev_claims",
    "annual_income", "credit_score",
]

CAT_FEATURE_COLS = [
    "gender", "region", "marital_status", "coverage_type", "smoker", "exercise_frequency",
]

SCALE_COLS = [
    "age",
    "num_dependents",
    "bmi",
    "chronic_disease",
    "prev_hospitalizations",
    "num_medications",
    "policy_duration",
    "deductible",
    "num_prev_claims",
    "annual_income",
    "credit_score",
]


# ==============================================================
# LOGO SVG
# ==============================================================

def get_logo_svg(size: int = 56) -> str:
    """Return a self-contained inline SVG medical AI logo."""
    s = str(size)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="' + s + '" height="' + s + '" viewBox="0 0 100 100">'
        '<defs>'
        '<linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#1E3A8A"/>'
        '<stop offset="100%" stop-color="#2563EB"/>'
        '</linearGradient>'
        '<filter id="shadow">'
        '<feDropShadow dx="0" dy="2" stdDeviation="3" '
        'flood-color="#2563EB" flood-opacity="0.4"/>'
        '</filter>'
        '</defs>'
        '<circle cx="50" cy="50" r="48" fill="url(#g1)" '
        'filter="url(#shadow)"/>'
        '<circle cx="50" cy="50" r="44" fill="none" '
        'stroke="rgba(255,255,255,0.15)" stroke-width="1"/>'
        '<rect x="44" y="24" width="12" height="52" rx="5" '
        'fill="white" opacity="0.95"/>'
        '<rect x="24" y="44" width="52" height="12" rx="5" '
        'fill="white" opacity="0.95"/>'
        '<polyline '
        'points="14,54 24,54 30,40 38,66 46,54 54,54 60,40 68,66 76,54 86,54" '
        'fill="none" stroke="#93C5FD" stroke-width="2.5" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg>'
    )


# ==============================================================
# UI HELPERS — section_header and white_card
# ==============================================================

def section_header(icon: str, title: str, subtitle: str = "") -> None:
    """Render a styled section heading with gradient icon box."""
    sub_html = (
        f'<div style="font-size:13px; color:#64748B; margin-top:3px;">'
        f'{subtitle}</div>'
    ) if subtitle else ""

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:14px;
                    margin:32px 0 18px; padding-bottom:14px;
                    border-bottom:2px solid #E2E8F0;">
            <div style="background:linear-gradient(135deg,#2563EB,#7C3AED);
                        width:42px; height:42px; border-radius:12px;
                        display:flex; align-items:center;
                        justify-content:center; font-size:20px;
                        box-shadow:0 4px 12px rgba(37,99,235,0.3);
                        flex-shrink:0;">{icon}</div>
            <div>
                <div style="font-size:20px; font-weight:800;
                            color:#0F172A; line-height:1.2;">{title}</div>
                {sub_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def white_card(content_html: str, padding: str = "24px",
               extra_style: str = "") -> str:
    """Return HTML string for a white rounded shadow card."""
    return (
        f'<div style="background:white; border-radius:20px; '
        f'padding:{padding}; '
        f'box-shadow:0 4px 24px rgba(0,0,0,0.06); '
        f'border:1px solid #F1F5F9; {extra_style}">'
        f'{content_html}'
        f'</div>'
    )


# ==============================================================
# MODEL LOADING
# ==============================================================

@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    """Load all trained ML models and encoders from disk."""
    required = {
        "risk_model":    "risk_model.joblib",
        "premium_model": "premium_model.joblib",
        "fraud_model":   "fraud_model.joblib",
        "claim_model":   "claim_model.joblib",
        "encoders":      "encoders.joblib",
    }
    loaded, missing = {}, []
    for key, fname in required.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            loaded[key] = joblib.load(path)
        else:
            missing.append(fname)

    if missing:
        st.error(
            f"Missing model files: {missing}\n\n"
            "Run: `python notebooks/eda_and_training.py`"
        )
        st.stop()
    return loaded


# ==============================================================
# INPUT TRANSFORMATION
# ==============================================================

def transform_input(user_input: dict, encoders: dict) -> pd.DataFrame:
    """
    Convert sidebar user_input dict into a single-row DataFrame
    ready for model prediction (encoded + scaled).
    """
    row = {col: [user_input[col]] for col in FEATURE_COLS}
    df  = pd.DataFrame(row)
    df[CAT_FEATURE_COLS] = encoders["ordinal_encoder"].transform(
        df[CAT_FEATURE_COLS]
    )
    df[SCALE_COLS] = encoders["scaler"].transform(df[SCALE_COLS])
    return df


def transform_batch(df_input: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Transform a full DataFrame of records for batch prediction.
    Missing columns are filled with defaults.
    """
    df = df_input.copy()

    for col in CAT_FEATURE_COLS:
        if col not in df.columns:
            df[col] = "No"
    for col in SCALE_COLS:
        if col not in df.columns:
            df[col] = 0

    df[CAT_FEATURE_COLS] = encoders["ordinal_encoder"].transform(
        df[CAT_FEATURE_COLS]
    )
    df[SCALE_COLS] = encoders["scaler"].transform(df[SCALE_COLS])
    return df[FEATURE_COLS]


# ==============================================================
# FORMATTING
# ==============================================================

def fmt_inr(value: float) -> str:
    """Format as Indian Rupees using HTML entity for rupee symbol."""
    value = max(0.0, float(value))
    if value >= 1_00_00_000:
        return f"&#8377;{value / 1_00_00_000:.2f} Cr"
    if value >= 1_00_000:
        return f"&#8377;{value / 1_00_000:.2f} L"
    return f"&#8377;{int(value):,}"


def fmt_inr_plain(value: float) -> str:
    """Format as Indian Rupees using plain ₹ symbol (for st.metric etc.)."""
    value = max(0.0, float(value))
    if value >= 1_00_00_000:
        return f"₹{value / 1_00_00_000:.2f} Cr"
    if value >= 1_00_000:
        return f"₹{value / 1_00_000:.2f} L"
    return f"₹{int(value):,}"


def usd_to_inr(usd: float) -> float:
    """Convert USD amount to INR."""
    return float(usd) * USD_TO_INR


def fmt_percent(p: float) -> str:
    """Format a 0–1 float as a percentage string."""
    return f"{p * 100:.1f}%"


def risk_color(label: str) -> str:
    """Return hex color for a risk label."""
    return {
        "Low":    "#10B981",
        "Medium": "#F59E0B",
        "High":   "#EF4444",
    }.get(label, "#6B7280")


def risk_badge(label: str) -> str:
    """Return an HTML badge span for a risk label."""
    mapping = {
        "Low":    ("D1FAE5", "065F46"),
        "Medium": ("FEF3C7", "92400E"),
        "High":   ("FEE2E2", "991B1B"),
    }
    bg, txt = mapping.get(label, ("F3F4F6", "374151"))
    return (
        f"<span style='background:#{bg}; color:#{txt}; "
        f"padding:4px 14px; border-radius:20px; "
        f"font-weight:700; font-size:13px;'>{label} Risk</span>"
    )


def fraud_badge(is_fraud: bool) -> str:
    """Return an HTML badge span for fraud status."""
    if is_fraud:
        return (
            "<span style='background:#FEE2E2; color:#991B1B; "
            "padding:4px 14px; border-radius:20px; font-weight:700;'>"
            "&#9888; Potential Fraud</span>"
        )
    return (
        "<span style='background:#D1FAE5; color:#065F46; "
        "padding:4px 14px; border-radius:20px; font-weight:700;'>"
        "&#10003; Likely Genuine</span>"
    )


# ==============================================================
# INPUT VALIDATION
# ==============================================================

def validate_input(ui: dict) -> list:
    """
    Validate sidebar inputs.
    Returns a list of error strings. Empty list means all valid.
    """
    errs = []
    if not (18 <= ui["age"] <= 75):
        errs.append("Age must be between 18 and 75.")
    if not (15.0 <= ui["bmi"] <= 55.0):
        errs.append("BMI must be between 15.0 and 55.0.")
    if not (300 <= ui["credit_score"] <= 850):
        errs.append("Credit Score must be between 300 and 850.")
    if ui["annual_income"] < (1_00_000 / USD_TO_INR):
        errs.append("Annual Income must be at least Rs 1,00,000.")
    if ui["num_dependents"] < 0:
        errs.append("Dependents cannot be negative.")
    if ui["num_prev_claims"] < 0:
        errs.append("Previous Claims cannot be negative.")
    return errs


# ==============================================================
# AI EXPLANATION GENERATORS
# ==============================================================

def generate_risk_explanation(ui: dict, risk_label: str) -> dict:
    """Generate human-readable risk explanation from user inputs."""
    positive, negative = [], []

    # Smoker
    if ui["smoker"] == "Yes":
        negative.append("Smoking significantly increases health risk")
    else:
        positive.append("Non-smoker — lower health risk")

    # BMI
    bmi = ui["bmi"]
    if bmi < 18.5:
        negative.append("Underweight BMI — potential health concern")
    elif bmi <= 24.9:
        positive.append("Healthy BMI — good overall fitness")
    elif bmi <= 29.9:
        negative.append("Overweight BMI — moderate health risk")
    else:
        negative.append("Obese BMI — high health risk factor")

    # Chronic disease
    if ui["chronic_disease"]:
        negative.append("Chronic disease — elevated medical risk")
    else:
        positive.append("No chronic disease — healthy baseline")

    # Hospitalisations
    h = ui["prev_hospitalizations"]
    if h == 0:
        positive.append("No previous hospitalisations")
    elif h <= 2:
        negative.append(f"{h} prior hospitalisation(s)")
    else:
        negative.append(f"{h} hospitalisations — high medical history")

    # Exercise
    ex = ui["exercise_frequency"]

    if ex in ["Daily", "Often"]:
        positive.append("Regular exercise — excellent lifestyle")

    elif ex == "Sometimes":
        positive.append("Moderate exercise frequency")

    else:   # Never or Rarely
        negative.append("Low exercise frequency — lifestyle risk")




    # Credit score
    cs = ui["credit_score"]
    if cs >= 750:
        positive.append("Excellent credit score")
    elif cs >= 650:
        positive.append("Good credit score")
    else:
        negative.append("Low credit score — financial risk indicator")

    # Previous claims
    nc = ui["num_prev_claims"]
    if nc == 0:
        positive.append("No previous claims history")
    elif nc <= 2:
        negative.append(f"{nc} previous claim(s)")
    else:
        negative.append(f"{nc} previous claims — high utilisation")

    recommendations = {
        "Low": [
            "Maintain your current healthy lifestyle",
            "Schedule annual preventive health check-ups",
            "You may be eligible for a premium discount",
            "Consider increasing coverage for better protection",
        ],
        "Medium": [
            "Improve exercise to at least 3-4 days per week",
            "Monitor BMI and maintain a balanced diet",
            "Schedule bi-annual health check-ups",
            "Review and optimise your current coverage plan",
        ],
        "High": [
            "Consult a doctor for a comprehensive health evaluation",
            "Consider lifestyle changes: diet, exercise, quit smoking",
            "Higher premium may apply — explore suitable plans",
            "Enrol in a chronic disease management programme",
        ],
    }

    return {
        "positive":        positive,
        "negative":        negative,
        "recommendations": recommendations.get(risk_label, []),
        "summary": (
            f"Risk is {risk_label} based on "
            f"{len(positive)} positive and "
            f"{len(negative)} risk indicators."
        ),
    }


def generate_premium_explanation(ui: dict, premium_inr: float) -> dict:
    """Explain why the predicted premium is at this level."""
    reasons = []

    if ui["age"] > 50:
        reasons.append(f"Age {ui['age']} — premiums rise significantly after 50")
    elif ui["age"] > 35:
        reasons.append(f"Age {ui['age']} — moderate age loading applied")

    if ui["smoker"] == "Yes":
        reasons.append("Smoker surcharge — typically +40% to +60%")
    if ui["chronic_disease"]:
        reasons.append("Chronic disease loading — higher medical costs")
    if ui["bmi"] > 30:
        reasons.append(f"High BMI ({ui['bmi']:.1f}) — obesity cost loading")
    if ui["num_prev_claims"] > 2:
        reasons.append(f"{ui['num_prev_claims']} previous claims — history surcharge")
    if ui["coverage_type"] in ["Premium", "Comprehensive"]:
        reasons.append(f"{ui['coverage_type']} coverage — broader protection costs more")
    if ui["prev_hospitalizations"] > 1:
        reasons.append(f"{ui['prev_hospitalizations']} hospitalisations — utilisation loading")

    if not reasons:
        reasons.append("Healthy profile — no major risk loadings applied")
        reasons.append("Competitive base premium applies")

    return {
        "reasons": reasons,
        "recommendations": [
            "Maintaining a healthy BMI may reduce future premiums",
            "Quitting smoking can reduce your premium by up to 30%",
            "Regular exercise qualifies you for wellness discounts",
            "Annual check-ups help maintain low-risk status",
        ],
    }


def generate_fraud_explanation(ui: dict, is_fraud: bool, prob: float) -> dict:
    """Generate fraud detection explanation from inputs."""
    flags, clear = [], []

    if ui["num_prev_claims"] >= 4:
        flags.append(f"High claim frequency ({ui['num_prev_claims']} claims)")
    else:
        clear.append("Claim frequency is within normal range")

    if ui["policy_duration"] <= 2 and ui["num_prev_claims"] > 1:
        flags.append("Multiple claims on a new policy — suspicious pattern")
    else:
        clear.append("Policy duration and claim history are consistent")

    if ui["prev_hospitalizations"] >= 3:
        flags.append(f"Frequent hospitalisations ({ui['prev_hospitalizations']})")
    else:
        clear.append("Hospitalisation history appears normal")

    if ui["credit_score"] < 550:
        flags.append("Low credit score — financial motivation indicator")
    else:
        clear.append("Financial profile appears stable")

    if ui["num_medications"] >= 8:
        flags.append(f"High medication count ({ui['num_medications']}) — verify")
    else:
        clear.append("Medication count within expected range")

    if is_fraud:
        action = "Investigation Recommended — Forward to Special Investigation Unit (SIU)"
    elif prob > 0.35:
        action = "Manual Review Required — Verify claim documents before approval"
    else:
        action = "Approve — Standard claim processing recommended"

    return {
        "flags": flags,
        "clear": clear,
        "action": action,
    }


def generate_claim_explanation(ui: dict, claim_inr: float) -> dict:
    """Explain the predicted claim amount."""
    drivers = []

    if ui["age"] > 50:
        drivers.append(f"Age {ui['age']} — older policyholders tend to have higher claims")
    if ui["chronic_disease"]:
        drivers.append("Chronic disease — ongoing medical expense contributor")
    if ui["prev_hospitalizations"] > 0:
        drivers.append(f"{ui['prev_hospitalizations']} hospitalisation(s) — utilisation history")
    if ui["num_prev_claims"] > 0:
        drivers.append(f"{ui['num_prev_claims']} previous claim(s) — historical pattern")
    if ui["smoker"] == "Yes":
        drivers.append("Smoking — increased long-term health costs")
    if ui["bmi"] > 30:
        drivers.append(f"BMI {ui['bmi']:.1f} — obesity-related medical costs")
    if not drivers:
        drivers.append("Healthy profile — minimal medical cost drivers identified")

    reserve = claim_inr * 1.25
    if claim_inr < 50_000:
        processing = "3-5 business days"
    elif claim_inr < 2_00_000:
        processing = "7-14 business days"
    else:
        processing = "14-30 business days"

    return {
        "drivers": drivers,
        "recommendations": [
            "Preventive healthcare can significantly reduce future claims",
            f"Recommended reserve: {fmt_inr_plain(reserve)} (25% safety margin)",
            "Regular health check-ups help catch issues early",
            "Healthy lifestyle reduces expected claim frequency",
        ],
        "reserve":         reserve,
        "processing_time": processing,
    }


# ==============================================================
# SIDEBAR INPUTS + NAVIGATION
# ==============================================================

def render_sidebar_inputs():
    """
    Render the complete sidebar:
      - Logo + Brand header
      - Navigation radio (no label, no text box)
      - Demographics, Health, Policy, Financial inputs
      - Footer

    Returns:
      user_input (dict), selected_page (str)
    """

    # ── Brand Header ───────────────────────────────────────────
    st.sidebar.markdown(
        f"""
        <div style="padding:28px 20px 20px; text-align:center;
                    border-bottom:1px solid rgba(255,255,255,0.08);">
            <div style="width:68px; height:68px; margin:0 auto 12px;
                        filter:drop-shadow(0 4px 14px rgba(37,99,235,0.55));">
                {get_logo_svg(68)}
            </div>
            <div style="font-size:15px; font-weight:900; color:#E2E8F0;
                        letter-spacing:2.5px; text-transform:uppercase;">
                MEDIPREDICT AI
            </div>
            <div style="font-size:10px; color:#475569; margin-top:4px;
                        letter-spacing:0.5px;">
                Healthcare Insurance Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Navigation ─────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='padding:12px 0 4px;'></div>",
        unsafe_allow_html=True,
    )

    pages = {
        "🏠  Home":               "Home",
        "⚠️  Risk Prediction":    "Risk",
        "💰  Premium Prediction": "Premium",
        "🚨  Fraud Detection":    "Fraud",
        "📄  Claim Prediction":   "Claim",
    }

    if "page" not in st.session_state:
        st.session_state.page = "🏠  Home"

    # label_visibility="hidden" removes label text AND its space
    page_sel = st.sidebar.radio(
        label="__nav__",
        options=list(pages.keys()),
        index=list(pages.keys()).index(st.session_state.page),
        label_visibility="hidden",
        key="sidebar_nav",
    )
    st.session_state.page = page_sel

    # ── Divider ────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='height:1px; background:rgba(255,255,255,0.08);"
        " margin:10px 16px 4px;'></div>",
        unsafe_allow_html=True,
    )

    # ── Section label helper ───────────────────────────────────
    def sb_label(icon: str, text: str) -> None:
        st.sidebar.markdown(
            f"<div style='font-size:9px; font-weight:700; color:#475569;"
            f" letter-spacing:1.5px; text-transform:uppercase;"
            f" padding:10px 16px 4px;'>{icon} {text}</div>",
            unsafe_allow_html=True,
        )

    def sb_divider() -> None:
        st.sidebar.markdown(
            "<div style='height:1px; background:rgba(255,255,255,0.06);"
            " margin:8px 16px;'></div>",
            unsafe_allow_html=True,
        )

    # ── Demographics ───────────────────────────────────────────
    sb_label("👤", "Demographics")
    age            = st.sidebar.slider("Age (years)", 18, 75, 35)
    gender         = st.sidebar.selectbox("Gender", ["Male", "Female"])
    region         = st.sidebar.selectbox(
        "Region", ["Northeast", "Southeast", "Midwest", "West"]
    )
    marital_status = st.sidebar.selectbox(
        "Marital Status", ["Single", "Married", "Divorced", "Widowed"]
    )
    num_dependents = st.sidebar.number_input(
        "Dependents", min_value=0, max_value=5, value=0, step=1
    )
    sb_divider()

    # ── Health ─────────────────────────────────────────────────
    sb_label("🏥", "Health")
    bmi = st.sidebar.slider(
        "BMI", 15.0, 55.0, 25.0, 0.1,
        help="18.5-24.9 Normal | 25-29.9 Overweight | 30+ Obese",
    )
    smoker                = st.sidebar.selectbox("Smoker", ["No", "Yes"])
    chronic_disease       = st.sidebar.selectbox(
        "Chronic Disease", [0, 1],
        format_func=lambda x: "Yes" if x else "No",
    )
    prev_hospitalizations = st.sidebar.slider("Hospitalisations", 0, 5, 0)
    num_medications       = st.sidebar.slider("Medications", 0, 10, 0)
    exercise_frequency = st.sidebar.selectbox(
    "Exercise Frequency",
    [
        "Never",
        "Rarely",
        "Sometimes",
        "Often",
        "Daily",
    ],
    index=2,
)
    sb_divider()

    # ── Policy ─────────────────────────────────────────────────
    sb_label("📄", "Policy")
    coverage_type   = st.sidebar.selectbox(
        "Coverage", ["Basic", "Standard", "Premium"],
        index=1,
    )
    policy_duration = st.sidebar.slider("Duration (yrs)", 1, 20, 5)
    deductible      = st.sidebar.selectbox(
        "Deductible", [500, 1000, 1500, 2000, 3000, 5000], index=1
    )
    num_prev_claims = st.sidebar.slider("Prior Claims", 0, 7, 0)
    sb_divider()

    # ── Financial ──────────────────────────────────────────────
    sb_label("💰", "Financial")
    annual_income_inr = st.sidebar.number_input(
        "Annual Income (Rs)",
        min_value=100_000,
        max_value=50_000_000,
        value=600_000,
        step=10_000,
    )
    credit_score      = st.sidebar.slider("Credit Score", 300, 850, 680)

    # Convert INR to USD for model (trained on USD values)
    annual_income_usd = annual_income_inr / USD_TO_INR

    # ── Footer ─────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='height:1px; background:rgba(255,255,255,0.06);"
        " margin:12px 16px 0;'></div>"
        "<div style='text-align:center; font-size:9px; color:#1E293B;"
        " padding:10px 0 14px; letter-spacing:0.5px;'>"
        "2026 MEDIPREDICT AI</div>",
        unsafe_allow_html=True,
    )

    # ── Build user_input dict ──────────────────────────────────
    user_input = {
        "age":                   age,
        "gender":                gender,
        "region":                region,
        "marital_status":        marital_status,
        "num_dependents":        num_dependents,
        "bmi":                   bmi,
        "smoker":                smoker,
        "chronic_disease":       chronic_disease,
        "prev_hospitalizations": prev_hospitalizations,
        "num_medications":       num_medications,
        "exercise_frequency":    exercise_frequency,
        "coverage_type":         coverage_type,
        "policy_duration":       policy_duration,
        "deductible":            deductible,
        "num_prev_claims":       num_prev_claims,
        "annual_income":         annual_income_usd,
        "annual_income_inr":     annual_income_inr,
        "credit_score":          credit_score,
    }

    return user_input, pages[page_sel]
