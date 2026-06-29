# ============================================================
# app/app.py — MEDIPREDICT AI
# Run: python -m streamlit run app/app.py
# ============================================================

import os, sys

APP_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, APP_DIR)

import streamlit as st
from pages import home, risk_prediction, premium_prediction
from pages import fraud_detection, claim_prediction
from utils.helpers import render_sidebar_inputs

st.set_page_config(
    page_title="MEDIPREDICT AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Background */
.main { background: #F0F4FF !important; }
.main .block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1280px !important;
}

/* Hide Streamlit chrome */
[data-testid="stSidebarNav"] { display: none !important; }
#MainMenu, footer, header    { visibility: hidden !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0F172A 0%,#1E293B 60%,#0F172A 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }

[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #94A3B8 !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
}

/* Nav radio */
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px !important;
    padding: 0 14px !important;
}
[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 11px 16px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    display: block !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(37,99,235,0.18) !important;
    border-color: rgba(37,99,235,0.35) !important;
    transform: translateX(3px) !important;
}
/* Hide radio label text */
[data-testid="stSidebar"] .stRadio > div:first-child { display:none !important; }
[data-testid="stSidebar"] .stRadio > label          { display:none !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: white !important;
    border-radius: 14px !important;
    padding: 16px !important;
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}
[data-testid="stMetricLabel"] { color:#64748B !important; font-size:12px !important; }
[data-testid="stMetricValue"] { color:#1E293B !important; font-size:22px !important; font-weight:800 !important; }

/* ── Primary buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#2563EB 0%,#7C3AED 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 13px 28px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.35) !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(37,99,235,0.45) !important;
}
.stButton > button[kind="secondary"] {
    background: white !important;
    color: #2563EB !important;
    border: 2px solid #2563EB !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

/* ── Download ── */
.stDownloadButton > button {
    background: white !important;
    color: #2563EB !important;
    border: 2px solid #2563EB !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    width: 100% !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: white !important;
    border-radius: 14px !important;
    padding: 5px !important;
    border: 1px solid #E2E8F0 !important;
    gap: 4px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    color: #64748B !important;
    padding: 9px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#2563EB,#7C3AED) !important;
    color: white !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #2563EB !important;
    border-radius: 16px !important;
    background: rgba(37,99,235,0.02) !important;
    padding: 12px !important;
}

/* ── Progress bars ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg,#2563EB,#7C3AED) !important;
    border-radius: 99px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F0F4FF; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 99px; }

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(14px); }
    to   { opacity:1; transform:translateY(0); }
}
.fade-in { animation: fadeInUp 0.45s ease both; }
</style>
""", unsafe_allow_html=True)


def main():
    user_input, page = render_sidebar_inputs()
    st.session_state["current_user_input"] = user_input

    if   page == "Home":    home.show()
    elif page == "Risk":    risk_prediction.show(user_input)
    elif page == "Premium": premium_prediction.show(user_input)
    elif page == "Fraud":   fraud_detection.show(user_input)
    elif page == "Claim":   claim_prediction.show(user_input)


if __name__ == "__main__":
    main()