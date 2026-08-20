import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import re
import random
import string
import smtplib
from email.message import EmailMessage
from datetime import datetime
from io import BytesIO

import torch
import torch.nn as nn
import joblib

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt

import base64
import hashlib
import html

from who_standards import (
    calc_hfa_percentile,
    calc_bfa_percentile,
    classify_pediatric_status,
    CLASS_LABELS
)
from meal_planner import (
    generate_custom_meal_plan,
    create_meal_plan_pdf,
    HAND_PORTION_GUIDE,
    GROWTH_STATUS_ADVICE,
    RDA_TABLE
)

# ------------------- SECURITY HELPERS -------------------
def hash_password(password: str) -> str:
    """Hash password using SHA-256 with a unique application salt."""
    if not password:
        return ""
    salt = "ai_growth_child_advisor_salt_2026"
    return hashlib.sha256((str(password) + salt).encode('utf-8')).hexdigest()

def verify_password(stored_password: str, input_password: str) -> bool:
    """Verify password supporting both SHA-256 hashed and legacy plaintext values."""
    if not stored_password or not input_password:
        return False
    if stored_password == hash_password(input_password):
        return True
    if stored_password == input_password:
        return True
    return False

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="AI Child Growth Advisor — Official Portal",
    page_icon="logo_icon.png" if os.path.exists("logo_icon.png") else "🧒",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_base64_image(image_path):
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            return ""
    return ""

LOGO_B64 = get_base64_image("logo_icon.png") or get_base64_image("logo.png")
FULL_LOGO_B64 = get_base64_image("logo.png")

# ------------------- CONSTANTS & FILE PATHS -------------------
HFA_BOYS_FILE = "tab_hfa_boys_p_0_5.xlsx"
HFA_GIRLS_FILE = "tab_hfa_girls_p_0_5.xlsx"
WFA_BOYS_FILE = "tab_wfa_boys_p_0_5.xlsx"
WFA_GIRLS_FILE = "tab_wfa_girls_p_0_5.xlsx"
WFH_BOYS_FILE = "tab_wfh_boys_p_0_5.xlsx"
WFH_GIRLS_FILE = "tab_wfh_girls_p_0_5.xlsx"
MODEL_PATH = "growth_model.pth"
SCALER_PATH = "scaler.joblib"
PARAMS_PATH = "best_params.json"
DAYS_PER_MONTH = 30.4375

ATTENDANCE_FILE = "Child_Attendance_Data(1).xlsx"
FOOD_RECOMMENDATIONS_FILE = "Food+Recommendations.csv"
MEAL_SCHEDULE_FILE = "MealSchedule.csv"
ATTENDANCE_JSON = "attendance_records.json"
MESSAGES_JSON = "messages.json"
USERS_JSON = "users.json"
REPORTS_JSON = "reports.json"
TEMP_REPORTS_DIR = "temp_reports"
VERIFICATION_CODES_JSON = "verification_codes.json"
PENDING_TEACHER_JSON = "pending_teacher_requests.json"

# ------------------- ADMIN EMAIL & SECRETS CONFIG -------------------
# Securely load credentials from Streamlit Secrets or Environment Variables (safe against live leaks)
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "mystuntman009@gmail.com")
ADMIN_EMAIL_PASSWORD = os.environ.get("ADMIN_EMAIL_PASSWORD", "")
try:
    if hasattr(st, "secrets"):
        if "ADMIN_EMAIL" in st.secrets:
            ADMIN_EMAIL = st.secrets["ADMIN_EMAIL"]
        if "ADMIN_EMAIL_PASSWORD" in st.secrets:
            ADMIN_EMAIL_PASSWORD = st.secrets["ADMIN_EMAIL_PASSWORD"]
except Exception:
    pass

if not os.path.exists(TEMP_REPORTS_DIR):
    os.makedirs(TEMP_REPORTS_DIR, exist_ok=True)

# ------------------- THEME STATE & DYNAMIC CSS STYLING -------------------
if "app_theme" not in st.session_state:
    st.session_state.app_theme = "dark"

def get_theme_css(theme: str) -> str:
    if theme == "light":
        return """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: linear-gradient(180deg, #fcfaf4 0%, #f4eee2 100%) !important;
        color: #38240D !important;
    }

    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ---- Universal Text & Headings (Light) ---- */
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #38240D !important;
        font-weight: 800 !important;
    }
    .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown div {
        color: #38240D !important;
    }

    /* ---- Form & Input Labels (Light) ---- */
    label, [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
        color: #38240D !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }

    /* ---- Streamlit Metrics (Light) ---- */
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        color: #C05800 !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        text-shadow: none !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        color: #713600 !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* ---- Streamlit Tabs (Light) ---- */
    [data-testid="stTabs"] button[role="tab"] {
        color: #713600 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #C05800 !important;
        border-bottom: 3px solid #C05800 !important;
        font-weight: 800 !important;
    }
    [data-testid="stTabs"] button[role="tab"] p {
        color: inherit !important;
    }

    /* ---- Form Inputs & Selectboxes (Light) ---- */
    input, textarea {
        background-color: #ffffff !important;
        color: #38240D !important;
        border: 1.5px solid rgba(192, 88, 0, 0.35) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border: 1.5px solid rgba(192, 88, 0, 0.35) !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"] input {
        color: #38240D !important;
        background-color: #ffffff !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #38240D !important;
        border: 1.5px solid rgba(192, 88, 0, 0.35) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] span {
        color: #38240D !important;
    }
    div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li {
        background-color: #ffffff !important;
        color: #38240D !important;
    }
    div[role="radiogroup"] label, div[data-testid="stCheckbox"] label {
        color: #38240D !important;
        font-weight: 600 !important;
    }
    div[role="radiogroup"] label p, div[data-testid="stCheckbox"] label p {
        color: #38240D !important;
        font-weight: 600 !important;
    }

    /* ---- Hero Banner (Light) ---- */
    .hero-banner {
        background: linear-gradient(135deg, #713600 0%, #a34b00 50%, #C05800 100%) !important;
        color: #FDFBD4 !important;
        padding: 2.2rem 2rem;
        border-radius: 18px;
        box-shadow: 0 15px 28px -5px rgba(113, 54, 0, 0.25);
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(253, 251, 212, 0.35);
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: rgba(253, 251, 212, 0.12);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #FDFBD4 !important;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #fef8dd !important;
        max-width: 750px;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }

    /* ---- Metric Cards (Light) ---- */
    .metric-card {
        background: #ffffff !important;
        border: 1.5px solid rgba(192, 88, 0, 0.22) !important;
        box-shadow: 0 4px 14px rgba(113, 54, 0, 0.08) !important;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #C05800 !important;
        box-shadow: 0 8px 20px rgba(113, 54, 0, 0.15) !important;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 800;
        color: #C05800 !important;
    }
    .metric-lbl {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #713600 !important;
        font-weight: 700;
    }

    /* ---- Custom Cards (Light) ---- */
    .custom-card {
        background: #ffffff !important;
        border: 1.5px solid rgba(192, 88, 0, 0.22) !important;
        border-radius: 16px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 16px rgba(113, 54, 0, 0.08) !important;
        color: #38240D !important;
    }
    .custom-card:hover {
        border-color: #C05800 !important;
    }
    .custom-card * {
        color: #38240D !important;
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #38240D !important;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ---- Chat Bubbles (Light) ---- */
    .chat-bubble-teacher {
        background: #fdf6ec !important;
        border: 1px solid #713600 !important;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        color: #38240D !important;
    }
    .chat-bubble-parent {
        background: #fef9f0 !important;
        border: 1px solid #C05800 !important;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        color: #38240D !important;
    }

    /* ---- Badges (Light) ---- */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-healthy { background-color: #e6f4ea !important; color: #137333 !important; border: 1px solid #34a853 !important; }
    .badge-underweight { background-color: #fef7e0 !important; color: #b06000 !important; border: 1px solid #f9ab00 !important; }
    .badge-overweight { background-color: #fef0e6 !important; color: #c05800 !important; border: 1px solid #c05800 !important; }
    .badge-obese { background-color: #fce8e6 !important; color: #c5221f !important; border: 1px solid #ea4335 !important; }
    .badge-stunted { background-color: #f3e8fd !important; color: #7627bb !important; border: 1px solid #9333ea !important; }
    .badge-info { background-color: #38240D !important; color: #FDFBD4 !important; }

    .section-header {
        font-size: 1.5rem;
        font-weight: 800;
        color: #38240D !important;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
        border-left: 5px solid #C05800;
        padding-left: 0.85rem;
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(192, 88, 0, 0.2);
        background: #ffffff !important;
    }

    /* ---- Buttons (Light) ---- */
    button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #38240D !important;
        border: 1.5px solid rgba(192, 88, 0, 0.35) !important;
        font-weight: 600 !important;
    }
    button[kind="secondary"] p, button[kind="secondary"] div {
        color: #38240D !important;
        font-weight: 600 !important;
    }
    button[kind="secondary"]:hover {
        background-color: rgba(192, 88, 0, 0.1) !important;
        border-color: #C05800 !important;
        color: #C05800 !important;
    }
    button[kind="primary"], button[kind="primaryFormSubmit"] {
        background: linear-gradient(135deg, #713600 0%, #C05800 100%) !important;
        color: #FDFBD4 !important;
        border: 1px solid #713600 !important;
        font-weight: 700 !important;
    }
    button[kind="primary"] p, button[kind="primaryFormSubmit"] p {
        color: #FDFBD4 !important;
        font-weight: 700 !important;
    }

    /* ---- Sidebar Navigation Alignment (Light) ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fdfbf7 0%, #f6eee0 100%) !important;
        border-right: 1px solid rgba(192, 88, 0, 0.2) !important;
    }
    section[data-testid="stSidebar"] div.stButton > button {
        width: 100% !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding: 0.62rem 0.95rem !important;
        border-radius: 10px !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        color: #38240D !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        transition: all 0.2s cubic-bezier(.4,0,.2,1) !important;
        margin: 2px 0 !important;
        min-height: 42px !important;
        box-sizing: border-box !important;
    }
    section[data-testid="stSidebar"] div.stButton > button p,
    section[data-testid="stSidebar"] div.stButton > button div {
        text-align: left !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        font-size: 0.92rem !important;
        color: #38240D !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background: rgba(192, 88, 0, 0.12) !important;
        border-color: rgba(192, 88, 0, 0.3) !important;
        color: #713600 !important;
        transform: translateX(3px) !important;
    }
    /* Active Nav Button */
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #713600 0%, #C05800 100%) !important;
        border: 1px solid #713600 !important;
        color: #FDFBD4 !important;
        box-shadow: 0 4px 14px rgba(113, 54, 0, 0.25) !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] p,
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] div {
        color: #FDFBD4 !important;
        font-weight: 700 !important;
    }
    .sidebar-nav-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #713600 !important;
        padding: 0.6rem 0.4rem 0.35rem;
    }

    .logout-btn > button {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        color: #dc2626 !important;
    }
    .logout-btn > button:hover {
        background: rgba(239, 68, 68, 0.2) !important;
        border-color: #dc2626 !important;
        color: #991b1b !important;
    }

    .att-progress-bg {
        background: #f0e6d6;
        border: 1px solid rgba(192, 88, 0, 0.2);
        border-radius: 9999px;
        height: 8px;
        width: 100%;
        overflow: hidden;
        margin-top: 6px;
    }
    .att-progress-bar {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.4s ease;
    }
</style>
        """
    else:  # DARK MODE (CHOCOLATE TRUFFLE)
        return """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: radial-gradient(circle at 50% 0%, #2b1a0d 0%, #170e06 100%) !important;
        color: #FDFBD4 !important;
    }

    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ---- Universal Text & Headings (Dark) ---- */
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #FDFBD4 !important;
        font-weight: 800 !important;
    }
    .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown div {
        color: #FDFBD4 !important;
    }

    /* ---- Form & Input Labels (Dark) ---- */
    label, [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
        color: #FDFBD4 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }

    /* ---- Streamlit Metrics (Dark) ---- */
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        color: #C05800 !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        text-shadow: 0 0 12px rgba(192, 88, 0, 0.3) !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        color: #e5d8b8 !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* ---- Streamlit Tabs (Dark) ---- */
    [data-testid="stTabs"] button[role="tab"] {
        color: #e5d8b8 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #C05800 !important;
        border-bottom: 3px solid #C05800 !important;
        font-weight: 800 !important;
    }
    [data-testid="stTabs"] button[role="tab"] p {
        color: inherit !important;
    }

    /* ---- Form Inputs & Selectboxes (Dark) ---- */
    input, textarea {
        background-color: #2b1b0d !important;
        color: #FDFBD4 !important;
        border: 1.5px solid rgba(192, 88, 0, 0.35) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="input"] {
        background-color: #2b1b0d !important;
        border: 1.5px solid rgba(192, 88, 0, 0.35) !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"] input {
        color: #FDFBD4 !important;
        background-color: #2b1b0d !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #2b1b0d !important;
        color: #FDFBD4 !important;
        border: 1.5px solid rgba(192, 88, 0, 0.35) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] span {
        color: #FDFBD4 !important;
    }
    div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li {
        background-color: #2b1b0d !important;
        color: #FDFBD4 !important;
    }
    div[role="radiogroup"] label, div[data-testid="stCheckbox"] label {
        color: #FDFBD4 !important;
        font-weight: 600 !important;
    }
    div[role="radiogroup"] label p, div[data-testid="stCheckbox"] label p {
        color: #FDFBD4 !important;
        font-weight: 600 !important;
    }

    /* ---- Hero Banner ---- */
    .hero-banner {
        background: linear-gradient(135deg, #38240D 0%, #713600 50%, #C05800 100%) !important;
        color: #FDFBD4 !important;
        padding: 2.2rem 2rem;
        border-radius: 18px;
        box-shadow: 0 20px 30px -5px rgba(56, 36, 13, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(253, 251, 212, 0.25);
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: rgba(253, 251, 212, 0.08);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #FDFBD4 !important;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.4);
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #f7eed0 !important;
        max-width: 750px;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }

    /* ---- Metric Cards ---- */
    .metric-card {
        background: rgba(56, 36, 13, 0.75) !important;
        border: 1.5px solid rgba(192, 88, 0, 0.35) !important;
        backdrop-filter: blur(10px);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #C05800 !important;
        box-shadow: 0 8px 20px rgba(113, 54, 0, 0.3);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 800;
        color: #C05800 !important;
        text-shadow: 0 0 12px rgba(192, 88, 0, 0.3);
    }
    .metric-lbl {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #e5d8b8 !important;
        font-weight: 600;
    }

    /* ---- Custom Cards ---- */
    .custom-card {
        background: #2b1b0d !important;
        border: 1.5px solid rgba(192, 88, 0, 0.28) !important;
        border-radius: 16px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
        color: #FDFBD4 !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .custom-card:hover {
        border-color: #C05800 !important;
        box-shadow: 0 8px 24px rgba(113, 54, 0, 0.25);
    }
    .custom-card * {
        color: #FDFBD4 !important;
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #FDFBD4 !important;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ---- Chat Bubbles ---- */
    .chat-bubble-teacher {
        background: #38240D !important;
        border: 1px solid #713600 !important;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .chat-bubble-parent {
        background: #2b1709 !important;
        border: 1px solid #C05800 !important;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    /* ---- Badges ---- */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-underweight { background-color: rgba(253, 251, 212, 0.18) !important; color: #FDFBD4 !important; border: 1px solid rgba(253, 251, 212, 0.4) !important; }
    .badge-healthy { background-color: rgba(192, 88, 0, 0.3) !important; color: #FDFBD4 !important; border: 1px solid #C05800 !important; }
    .badge-overweight { background-color: rgba(224, 110, 20, 0.35) !important; color: #FDFBD4 !important; border: 1px solid #e06d06 !important; }
    .badge-obese { background-color: rgba(239, 68, 68, 0.25) !important; color: #fecaca !important; border: 1px solid #ef4444 !important; }
    .badge-stunted { background-color: rgba(113, 54, 0, 0.5) !important; color: #FDFBD4 !important; border: 1px solid #713600 !important; }
    .badge-info { background-color: #FDFBD4 !important; color: #38240D !important; font-weight: 800 !important; }

    .section-header {
        font-size: 1.5rem;
        font-weight: 800;
        color: #FDFBD4 !important;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
        border-left: 5px solid #C05800;
        padding-left: 0.85rem;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(192, 88, 0, 0.25);
    }

    /* ---- Buttons (Dark) ---- */
    button[kind="secondary"] {
        background-color: rgba(56, 36, 13, 0.6) !important;
        color: #FDFBD4 !important;
        border: 1.5px solid rgba(192, 88, 0, 0.35) !important;
        font-weight: 600 !important;
    }
    button[kind="secondary"] p, button[kind="secondary"] div {
        color: #FDFBD4 !important;
        font-weight: 600 !important;
    }
    button[kind="secondary"]:hover {
        background-color: rgba(192, 88, 0, 0.25) !important;
        border-color: #C05800 !important;
        color: #ffffff !important;
    }
    button[kind="primary"], button[kind="primaryFormSubmit"] {
        background: linear-gradient(135deg, #713600 0%, #C05800 100%) !important;
        border: 1px solid #FDFBD4 !important;
        color: #FDFBD4 !important;
        box-shadow: 0 4px 18px rgba(192, 88, 0, 0.5) !important;
        font-weight: 700 !important;
    }
    button[kind="primary"] p, button[kind="primaryFormSubmit"] p {
        color: #FDFBD4 !important;
        font-weight: 700 !important;
    }

    /* ---- Sidebar Navigation Alignment (Dark) ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #180e05 0%, #261509 50%, #38240D 100%) !important;
        border-right: 1px solid rgba(192, 88, 0, 0.25) !important;
    }
    section[data-testid="stSidebar"] div.stButton > button {
        width: 100% !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding: 0.62rem 0.95rem !important;
        border-radius: 10px !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        color: #f7eed0 !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        transition: all 0.2s cubic-bezier(.4,0,.2,1) !important;
        margin: 2px 0 !important;
        min-height: 42px !important;
        box-sizing: border-box !important;
    }
    section[data-testid="stSidebar"] div.stButton > button p,
    section[data-testid="stSidebar"] div.stButton > button div {
        text-align: left !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        font-size: 0.92rem !important;
        color: #f7eed0 !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background: rgba(192, 88, 0, 0.2) !important;
        border-color: rgba(192, 88, 0, 0.45) !important;
        color: #FDFBD4 !important;
        transform: translateX(3px) !important;
    }
    /* Active Nav Button */
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #713600 0%, #C05800 100%) !important;
        border: 1px solid #FDFBD4 !important;
        color: #FDFBD4 !important;
        box-shadow: 0 4px 18px rgba(192, 88, 0, 0.5) !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] p,
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] div {
        color: #FDFBD4 !important;
        font-weight: 700 !important;
    }

    .sidebar-nav-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #bfa882 !important;
        padding: 0.6rem 0.4rem 0.35rem;
    }

    .logout-btn > button {
        background: rgba(239, 68, 68, 0.12) !important;
        border: 1px solid rgba(239, 68, 68, 0.35) !important;
        color: #fca5a5 !important;
    }
    .logout-btn > button:hover {
        background: rgba(239, 68, 68, 0.3) !important;
        border-color: #ef4444 !important;
        color: #ffffff !important;
    }

    .att-progress-bg {
        background: #1a0f06;
        border: 1px solid rgba(192, 88, 0, 0.2);
        border-radius: 9999px;
        height: 8px;
        width: 100%;
        overflow: hidden;
        margin-top: 6px;
    }
    .att-progress-bar {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.4s ease;
    }
</style>
        """

st.markdown(get_theme_css(st.session_state.app_theme), unsafe_allow_html=True)

# ------------------- AI MODEL & WHO BACKEND ENGINE -------------------
class GrowthNet(nn.Module):
    def __init__(self, in_features=9, n_layers=4, n_units=134, dropout_rate=0.16):
        super().__init__()
        layers = []
        current_in = in_features
        for _ in range(n_layers):
            layers.append(nn.Linear(current_in, n_units))
            layers.append(nn.BatchNorm1d(n_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_in = n_units
        layers.append(nn.Linear(current_in, len(CLASS_LABELS)))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model_and_scaler(model_path=MODEL_PATH, scaler_path=SCALER_PATH, params_path=PARAMS_PATH):
    try:
        if os.path.exists(params_path) and os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(params_path, 'r') as f:
                best_params = json.load(f)
            model = GrowthNet(
                in_features=9,
                n_layers=best_params.get('n_layers', 4),
                n_units=best_params.get('n_units', 134),
                dropout_rate=best_params.get('dropout_rate', 0.16)
            )
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            scaler = joblib.load(scaler_path)
            return model, scaler
    except Exception as e:
        st.warning(f"AI Model load warning: {e}")
    return None, None

@st.cache_data
def load_ref(path: str, primary_col_regex: str):
    try:
        if os.path.exists(path):
            df = pd.read_excel(path)
            primary_col = next((c for c in df.columns if re.search(primary_col_regex, str(c), re.I)), None)
            if primary_col:
                pcols = [c for c in df.columns if re.match(r"P\d+", str(c))]
                df = df[[primary_col] + pcols].copy()
                df.columns = ["primary"] + pcols
                return df, pcols
    except Exception:
        pass
    return None, None

def parse_pcol(c: str) -> float:
    c_str = str(c).strip().upper()
    if c_str == "P999":
        return 99.9
    if c_str == "P01":
        return 0.1
    nums = re.findall(r"\d+", c_str)
    return float(nums[0]) if nums else 50.0

def interp_curve(ref_df, pcols, val):
    values = ref_df.iloc[:, 0].values.astype(float)
    if val <= values.min():
        row = ref_df.iloc[0]
    elif val >= values.max():
        row = ref_df.iloc[-1]
    else:
        idx = np.searchsorted(values, val, side="right")
        v0, v1 = values[idx - 1], values[idx]
        frac = (val - v0) / (v1 - v0)
        row0, row1 = ref_df.iloc[idx - 1], ref_df.iloc[idx]
        return {parse_pcol(c): float(row0[c] + frac * (row1[c] - row0[c])) for c in pcols}
    return {parse_pcol(c): float(row[c]) for c in pcols}

def est_percentile(value, curve):
    pts = sorted(curve.items(), key=lambda item: item[1])
    values = [v for p, v in pts]
    percs = [p for p, v in pts]
    if value <= values[0]:
        return min(percs[0], 0.1)
    if value >= values[-1]:
        return max(percs[-1], 99.9)
    j = np.searchsorted(values, value, side="right")
    v0, v1, p0, p1 = values[j - 1], values[j], percs[j - 1], percs[j]
    p_est = p0 + (value - v0) / (v1 - v0) * (p1 - p0)
    return max(0.1, min(99.9, p_est))

def ai_predict(model, scaler, age_m, ht, wt, sex, wfh_p, hfa_p, wfa_p=50.0, bfa_p=50.0):
    bmi = wt / ((ht / 100.0) ** 2)
    confidence_score = 0.96
    status = "Healthy"
    sex_num = 1 if sex in ["M", "Male", 1] else 0

    if model is not None and scaler is not None:
        try:
            input_data = np.array([[age_m, ht, wt, sex_num, bmi, hfa_p, wfa_p, wfh_p, bfa_p]])
            input_scaled = scaler.transform(input_data)
            x = torch.tensor(input_scaled, dtype=torch.float32)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx_tensor = torch.max(probs, dim=1)
                pred_idx = int(pred_idx_tensor.item())
                confidence_score = float(confidence.item())
                status = CLASS_LABELS.get(pred_idx, "Healthy")
        except Exception:
            pass

    # WHO pediatric clinical verification fallback
    rule_status_id = classify_pediatric_status(hfa_p, wfa_p, wfh_p, bfa_p, bmi)
    rule_status = CLASS_LABELS.get(rule_status_id, status)
    if status not in CLASS_LABELS.values() or confidence_score < 0.55:
        status = rule_status

    return status, confidence_score

def build_age_meal_ideas(age_m):
    if age_m < 24:
        return [
            "3 soft family meals + 2 healthy snacks with toddler-friendly portions.",
            "Include protein at each meal (egg yolk/boiled egg, soft paneer, dal, fish).",
            "Limit milk to 400-500ml/day to prevent iron deficiency and encourage solid food intake."
        ]
    elif age_m < 36:
        return [
            "3 structured meals + 2 planned nutritious snacks following a fixed schedule.",
            "Balanced plate: 1/2 vegetables and seasonal fruits, 1/4 protein, 1/4 whole grains.",
            "Foster self-feeding habits with colorful cut fruits and finger-food parathas."
        ]
    return [
        "3 wholesome meals + 2 active-play snacks (ICMR-NIN 2024 Guidelines).",
        "Offer wholesome dairy, nuts powder, pulses, green leafy veggies, and seasonal fruits.",
        "Ensure 60+ minutes of active physical outdoor play daily."
    ]

def get_ai_recommendations(status, age_m, wfh_p, hfa_p, bmi):
    advice = GROWTH_STATUS_ADVICE.get(status, GROWTH_STATUS_ADVICE["Healthy"])
    recs = {
        "summary": f"Diagnostic Status: {status} (BMI: {bmi:.1f} | Wt-for-Ht: P{wfh_p:.1f} | Ht-for-Age: P{hfa_p:.1f})",
        "what_to_eat": advice["tips"][:3],
        "how_to_eat": [
            "Follow a consistent 5-meal daily schedule (Breakfast, Snack, Lunch, Snack, Dinner).",
            "Use hand-size portion guides: Fist for veggies/fruits, Palm for protein, Cupped hand for grains.",
            "Avoid screen-time eating to build healthy intuitive hunger cues."
        ],
        "meal_ideas": build_age_meal_ideas(age_m),
    }
    return recs

def flatten_recommendation_plan(plan):
    flat = [plan["summary"], "What to eat:"]
    flat.extend([f"- {item}" for item in plan["what_to_eat"]])
    flat.append("How to eat:")
    flat.extend([f"- {item}" for item in plan["how_to_eat"]])
    flat.append("Meal and snack ideas:")
    flat.extend([f"- {item}" for item in plan["meal_ideas"]])
    return flat

def generate_report(age_m, ht, wt, sex, model, scaler):
    hfa_ref, hfa_pcols = load_ref(HFA_BOYS_FILE if sex in ["M", "Male", 1] else HFA_GIRLS_FILE, r"age|day|month")
    wfa_ref, wfa_pcols = load_ref(WFA_BOYS_FILE if sex in ["M", "Male", 1] else WFA_GIRLS_FILE, r"age|day|month")
    wfh_ref, wfh_pcols = load_ref(WFH_BOYS_FILE if sex in ["M", "Male", 1] else WFH_GIRLS_FILE, r"height|length")
    
    age_d = age_m * DAYS_PER_MONTH
    
    # HFA Percentile
    hfa_p = calc_hfa_percentile(age_m, ht, sex)
    if hfa_ref is not None:
        table_val = age_d if float(hfa_ref.iloc[:, 0].max()) > 120 else age_m
        hfa_curve = interp_curve(hfa_ref, hfa_pcols, table_val)
    else:
        hfa_curve = {50: ht}

    # WFA Percentile
    if wfa_ref is not None:
        wfa_curve = interp_curve(wfa_ref, wfa_pcols, age_m)
        wfa_p = est_percentile(wt, wfa_curve)
    else:
        wfa_curve = {50: wt}
        wfa_p = 50.0

    # WFH Percentile
    if wfh_ref is not None:
        wfh_curve = interp_curve(wfh_ref, wfh_pcols, ht)
        wfh_p = est_percentile(wt, wfh_curve)
    else:
        wfh_curve = {50: wt}
        wfh_p = 50.0

    bmi = wt / ((ht / 100.0) ** 2)
    bfa_p = calc_bfa_percentile(age_m, bmi, sex)

    ai_status, confidence = ai_predict(model, scaler, age_m, ht, wt, sex, wfh_p, hfa_p, wfa_p, bfa_p)

    who_msgs = []
    if wfh_p < 3 or bfa_p < 5:
        who_msgs.append((f"Wasting / Underweight risk (WFH: P{wfh_p:.1f} | BMI: P{bfa_p:.1f})", colors.red))
    elif wfh_p > 97 or bfa_p > 97:
        who_msgs.append((f"Obesity risk (WFH: P{wfh_p:.1f} | BMI: P{bfa_p:.1f})", colors.red))
    elif wfh_p > 85 or bfa_p > 85:
        who_msgs.append((f"Overweight risk (WFH: P{wfh_p:.1f} | BMI: P{bfa_p:.1f})", colors.HexColor("#f59e0b")))
    else:
        who_msgs.append(("Weight-for-height healthy.", colors.green))

    if hfa_p < 3:
        who_msgs.append((f"Stunting risk (Ht-for-Age: P{hfa_p:.1f})", colors.red))
    else:
        who_msgs.append(("Height-for-age healthy.", colors.green))

    rec_plan = get_ai_recommendations(ai_status, age_m, wfh_p, hfa_p, bmi)
    recommendations = flatten_recommendation_plan(rec_plan)

    return {
        "wfh_p": wfh_p,
        "hfa_p": hfa_p,
        "wfa_p": wfa_p,
        "bfa_p": bfa_p,
        "bmi": bmi,
        "who_msgs": who_msgs,
        "recommendations": recommendations,
        "ai_status": ai_status,
        "confidence": confidence,
        "hfa_curve": hfa_curve,
        "wfh_curve": wfh_curve,
        "age_d": age_d,
        "ht": ht,
        "wt": wt,
    }

def create_pdf_report(child_name, age_months, report):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin_left = 1.5 * cm
    content_width = width - 3.0 * cm

    c.setFillColor(colors.HexColor("#1f77b4"))
    c.rect(0, height - 1.2 * cm, width, 1.2 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin_left, height - 0.8 * cm, "Child Growth Report")
    c.setFillColor(colors.black)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin_left, height - 2 * cm, f"Name: {child_name}")
    c.setFont("Helvetica", 10)
    c.drawString(margin_left, height - 2.5 * cm, f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}")

    c.setFont("Helvetica-Bold", 11)
    y = height - 3.2 * cm
    metrics = [
        f"Age: {int(age_months)//12}y {int(age_months)%12}m",
        f"Height: {report['ht']:.1f} cm",
        f"Weight: {report['wt']:.1f} kg",
        f"BMI: {report['bmi']:.1f}",
        f"Height Percentile: P{report['hfa_p']:.1f}",
        f"Weight-for-Height: P{report['wfh_p']:.1f}",
    ]
    
    col1_x = margin_left
    col2_x = margin_left + content_width / 2
    metric_idx = 0
    for i in range(3):
        if metric_idx < len(metrics):
            c.drawString(col1_x, y, metrics[metric_idx])
            metric_idx += 1
        if metric_idx < len(metrics):
            c.drawString(col2_x, y, metrics[metric_idx])
            metric_idx += 1
        y -= 0.45 * cm

    y -= 0.4 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_left, y, "WHO Assessment Status:")
    c.setFont("Helvetica", 10)
    y -= 0.4 * cm
    for msg, color in report["who_msgs"]:
        c.setFillColor(color)
        c.drawString(margin_left + 0.3 * cm, y, f"• {msg}")
        y -= 0.4 * cm
    c.setFillColor(colors.black)

    y -= 0.3 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_left, y, "AI Recommendations:")
    c.setFont("Helvetica", 10)
    y -= 0.4 * cm
    for rec in report["recommendations"][:8]:
        c.drawString(margin_left + 0.3 * cm, y, rec.replace("**", ""))
        y -= 0.4 * cm

    hfa_buf = BytesIO()
    wfh_buf = BytesIO()
    
    plt.figure(figsize=(6, 3.8))
    plt.plot(list(report["hfa_curve"].keys()), list(report["hfa_curve"].values()), label="HFA Percentile", color="green")
    plt.scatter([report["hfa_p"]], [report["ht"]], color="blue", s=80, label="Child Height")
    plt.title("Height-for-Age Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hfa_buf, format="PNG")
    plt.close()
    hfa_buf.seek(0)

    plt.figure(figsize=(6, 3.8))
    plt.plot(list(report["wfh_curve"].keys()), list(report["wfh_curve"].values()), label="WFH Percentile", color="orange")
    plt.scatter([report["wfh_p"]], [report["wt"]], color="red", s=80, label="Child Weight")
    plt.title("Weight-for-Height Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(wfh_buf, format="PNG")
    plt.close()
    wfh_buf.seek(0)

    c.showPage()
    c.drawImage(ImageReader(hfa_buf), 2 * cm, height / 2, width=16 * cm, height=9 * cm)
    c.drawImage(ImageReader(wfh_buf), 2 * cm, 2 * cm, width=16 * cm, height=9 * cm)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def create_pdf_report_bytes(child_name, age_months, report):
    buf = create_pdf_report(child_name, age_months, report)
    return buf.getvalue()


# ------------------- DATA LOADERS & STORES -------------------
@st.cache_data
def load_students_data():
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
        df.columns = [c.strip() for c in df.columns]
        return df
    else:
        return pd.DataFrame([
            {"Child ID": f"C{i:03d}", "Child Name": f"Student {i}", "Parent Name": f"Parent {i}", "Place": "Coimbatore", "Phone Number": 9876543210}
            for i in range(1, 31)
        ])

def save_students_data(df: pd.DataFrame) -> bool:
    """Persist students DataFrame directly to Excel and refresh cache."""
    try:
        df.to_excel(ATTENDANCE_FILE, index=False)
        load_students_data.clear()
        return True
    except Exception as e:
        st.error(f"Failed to save student record to Excel: {e}")
        return False

def compute_child_attendance(child_id: str, attendance_store: dict) -> dict:
    """Calculate cumulative attendance metrics for a child across all recorded school days."""
    cid_u = str(child_id).strip().upper()
    total_days = 0
    present_days = 0
    date_history = []

    for date_str in sorted(attendance_store.keys(), reverse=True):
        records = attendance_store.get(date_str, {})
        status = None
        for k, v in records.items():
            if str(k).strip().upper() == cid_u:
                status = v
                break
        
        if status is not None:
            total_days += 1
            if status == "Present":
                present_days += 1
            date_history.append({"Date": date_str, "Status": status})

    absent_days = total_days - present_days
    pct = (present_days / total_days * 100.0) if total_days > 0 else 0.0

    if total_days == 0:
        badge_cls = "badge-info"
        badge_text = "No Records"
        status_label = "No Data"
    elif pct >= 90:
        badge_cls = "badge-healthy"
        badge_text = f"{pct:.1f}% (Excellent)"
        status_label = "Excellent"
    elif pct >= 75:
        badge_cls = "badge-overweight"
        badge_text = f"{pct:.1f}% (Good)"
        status_label = "Good"
    else:
        badge_cls = "badge-obese"
        badge_text = f"{pct:.1f}% (Low <75%)"
        status_label = "At Risk"

    return {
        "total_days": total_days,
        "present_days": present_days,
        "absent_days": absent_days,
        "percentage": pct,
        "badge_cls": badge_cls,
        "badge_text": badge_text,
        "status_label": status_label,
        "history": date_history
    }

def compute_class_attendance_summary(students_df: pd.DataFrame, attendance_store: dict) -> pd.DataFrame:
    """Build class-wide cumulative attendance roster for all students."""
    summary_list = []
    for _, row in students_df.iterrows():
        cid = str(row['Child ID'])
        cname = str(row['Child Name'])
        stats = compute_child_attendance(cid, attendance_store)
        summary_list.append({
            "Child ID": cid,
            "Child Name": cname,
            "Total Days Logged": stats["total_days"],
            "Present Days": stats["present_days"],
            "Absent Days": stats["absent_days"],
            "Attendance %": f"{stats['percentage']:.1f}%",
            "Raw Pct": stats["percentage"],
            "Status": stats["status_label"],
            "Parent Contact": row.get("Phone Number", "N/A"),
            "Location": row.get("Place", "N/A")
        })
    return pd.DataFrame(summary_list)

def reset_all_attendance_records() -> bool:
    """Reset all attendance records for a new academic year."""
    try:
        st.session_state.attendance_store = {}
        save_json_store(ATTENDANCE_JSON, {})
        return True
    except Exception as e:
        st.error(f"Error resetting attendance records: {e}")
        return False

@st.cache_data
def load_food_recommendations():
    if os.path.exists(FOOD_RECOMMENDATIONS_FILE):
        df = pd.read_csv(FOOD_RECOMMENDATIONS_FILE)
        df.columns = [c.strip() for c in df.columns]
        return df
    return pd.DataFrame()

@st.cache_data
def load_meal_schedule():
    if os.path.exists(MEAL_SCHEDULE_FILE):
        df = pd.read_csv(MEAL_SCHEDULE_FILE)
        df.columns = [c.strip() for c in df.columns]
        return df.dropna(subset=['Day'])
    return pd.DataFrame()

def load_json_store(filename, default_val):
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return default_val
    return default_val

def save_json_store(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

# ------------------- EMAIL & VERIFICATION UTILITIES -------------------
def generate_verification_code():
    return ''.join(random.choices(string.digits, k=6))

def send_verification_email(to_email, code, purpose="Verification"):
    """Send a verification code email from admin Gmail account."""
    try:
        msg = EmailMessage()
        msg['Subject'] = f"AI Growth Advisor — {purpose} Code: {code}"
        msg['From'] = ADMIN_EMAIL
        msg['To'] = to_email
        msg.set_content(f"""
Hello,

Your verification code for {purpose} is:

    {code}

Please enter this code in the AI Child Growth Advisor portal to proceed.
This code is valid for one-time use only.

If you did not request this, please ignore this email.

— AI Child Growth Advisor Admin
""")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(ADMIN_EMAIL, ADMIN_EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

students_df = load_students_data()
food_df = load_food_recommendations()
meal_df = load_meal_schedule()
growth_model, model_scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH, PARAMS_PATH)

DEFAULT_USERS = {
    "teacher": {
        "password": hash_password("teacher123"),
        "fullName": "Class Teacher (Admin)",
        "role": "Teacher",
        "childId": "",
        "email": "mystuntman009@gmail.com"
    },
    "parent_c002": {
        "password": hash_password("parent123"),
        "fullName": "Murugan (Kavya's Parent)",
        "role": "Parent",
        "childId": "C002",
        "email": ""
    },
    "parent_c001": {
        "password": hash_password("parent123"),
        "fullName": "Ramasamy (Arun's Parent)",
        "role": "Parent",
        "childId": "C001",
        "email": ""
    }
}

DEFAULT_REPORTS = [
    {
        "reportId": "R001",
        "childId": "C002",
        "childName": "Kavya",
        "title": "Kavya Growth Report \u2014 February 2026",
        "timestamp": "2026-02-18 14:30",
        "aiStatus": "Healthy",
        "hfa_p": 52.4,
        "wfh_p": 49.1,
        "bmi": 15.8,
        "pdf_filename": "C002_Kavya_Report.pdf"
    },
    {
        "reportId": "R002",
        "childId": "C001",
        "childName": "Arun",
        "title": "Arun Growth Report \u2014 February 2026",
        "timestamp": "2026-02-17 11:15",
        "aiStatus": "Underweight",
        "hfa_p": 42.1,
        "wfh_p": 2.8,
        "bmi": 13.2,
        "pdf_filename": "C001_Arun_Report.pdf"
    },
    {
        "reportId": "R003",
        "childId": "C003",
        "childName": "Prasad",
        "title": "Prasad Growth Report \u2014 February 2026",
        "timestamp": "2026-02-15 16:45",
        "aiStatus": "Healthy",
        "hfa_p": 60.5,
        "wfh_p": 55.0,
        "bmi": 16.1,
        "pdf_filename": "C003_Prasad_Report.pdf"
    }
]

# Always synchronize with disk so changes across tabs/admin actions are immediately active
st.session_state.users_store = load_json_store(USERS_JSON, DEFAULT_USERS)
st.session_state.attendance_store = load_json_store(ATTENDANCE_JSON, {})
st.session_state.messages_store = load_json_store(MESSAGES_JSON, [])
st.session_state.reports_store = load_json_store(REPORTS_JSON, DEFAULT_REPORTS)
st.session_state.verification_codes_store = load_json_store(VERIFICATION_CODES_JSON, [])
st.session_state.pending_teacher_store = load_json_store(PENDING_TEACHER_JSON, [])

if "current_user" not in st.session_state:
    st.session_state.current_user = None

def child_has_parent(child_id):
    """Check if a child already has a registered parent account."""
    cid_upper = str(child_id).strip().upper()
    for uname, udata in st.session_state.users_store.items():
        if udata.get("role") == "Parent" and str(udata.get("childId", "")).strip().upper() == cid_upper:
            return True, uname
    return False, None


def get_child_row(cid):
    matches = students_df[students_df['Child ID'].astype(str).str.upper() == str(cid).upper()]
    if not matches.empty:
        return matches.iloc[0]
    return None

def get_badge_class(status):
    st_str = str(status)
    if "Underweight" in st_str: return "badge-underweight"
    elif "Healthy" in st_str or "Normal" in st_str: return "badge-healthy"
    elif "Overweight" in st_str: return "badge-overweight"
    elif "Obese" in st_str: return "badge-obese"
    elif "Stunted" in st_str: return "badge-stunted"
    return "badge-info"

def get_pdf_bytes_for_report(rep):
    filename = rep.get("pdf_filename")
    if filename:
        filepath = os.path.join(TEMP_REPORTS_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return f.read()
    
    # Fallback PDF generator if file not found locally
    c_name = rep.get("childName", "Student")
    dummy_rep = {
        "hfa_p": rep.get("hfa_p", 50.0),
        "wfh_p": rep.get("wfh_p", 50.0),
        "bmi": rep.get("bmi", 15.5),
        "ai_status": rep.get("aiStatus", "Healthy"),
        "who_msgs": [("Growth check record", colors.green)],
        "recommendations": ["- Follow balanced diet and active play."],
        "hfa_curve": {50: 85.0},
        "wfh_curve": {50: 12.0},
        "ht": 85.0,
        "wt": 12.0
    }
    return create_pdf_report(c_name, 24, dummy_rep).getvalue()

# ------------------- AUTHENTICATION SCREEN -------------------
if st.session_state.current_user is None:
    with st.sidebar:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.85rem; padding: 0.6rem 0 0.4rem;">
            <div style="background: #FDFBD4; padding: 4px; border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 14px rgba(0,0,0,0.4); border: 1px solid #C05800;">
                <img src="data:image/png;base64,{LOGO_B64}" width="38" height="38" style="border-radius: 8px; object-fit: contain;" alt="Growth Advisor Logo">
            </div>
            <div>
                <div style="font-size: 1.25rem; font-weight: 800; color: {'#FDFBD4' if st.session_state.app_theme == 'dark' else '#38240D'}; letter-spacing: -0.02em; line-height: 1.15;">Growth Advisor</div>
                <div style="font-size: 0.72rem; color: #C05800; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">AI-POWERED PORTAL</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        cur_theme = st.session_state.get("app_theme", "dark")
        theme_btn_label = "☀️ Switch to Light Theme" if cur_theme == "dark" else "🌙 Switch to Dark Theme"
        if st.button(theme_btn_label, key="auth_theme_btn", use_container_width=True):
            st.session_state.app_theme = "light" if cur_theme == "dark" else "dark"
            st.rerun()

    st.markdown(f"""
    <div class="hero-banner">
        <div style="display: flex; align-items: center; gap: 1.1rem; margin-bottom: 0.9rem;">
            <div style="background: #ffffff; padding: 6px; border-radius: 16px; display: inline-flex; align-items: center; justify-content: center; box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25); border: 2px solid rgba(255, 255, 255, 0.4);">
                <img src="data:image/png;base64,{LOGO_B64}" width="54" height="54" style="border-radius: 12px; object-fit: contain;" alt="Growth Advisor Logo">
            </div>
            <div>
                <div class="hero-title" style="margin-bottom: 0.1rem; line-height: 1.15;">Growth Advisor</div>
                <div style="color: #FDFBD4; font-size: 0.85rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">AI-POWERED PORTAL</div>
            </div>
        </div>
        <div class="hero-subtitle">
            Welcome to the Child Growth & Health Management Portal. Log in or Register to access student details, attendance sheets, PDF growth report generation, nutrition plans, and parent-teacher communication.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, auth_col, col_r = st.columns([0.5, 2, 0.5])

    with auth_col:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("🔐 Portal Access")
        login_tab, register_tab, teacher_reg_tab, forgot_tab = st.tabs([
            "🔑 Login", "👨‍👩‍👧 Parent Register", "👩‍🏫 Teacher Register", "🔒 Forgot Password"
        ])

        # ---- LOGIN TAB ----
        with login_tab:
            if "failed_login_attempts" not in st.session_state:
                st.session_state.failed_login_attempts = 0

            with st.form("login_form"):
                login_user = st.text_input("Username", value="", key="login_user")
                login_pass = st.text_input("Password", type="password", value="", key="login_pass")
                submit_login = st.form_submit_button("🚀 Log In", use_container_width=True)

            if submit_login:
                if st.session_state.failed_login_attempts >= 5:
                    st.error("🚨 Account temporarily locked due to repeated failed attempts (Anti-Brute Force Protection). Please wait a moment or reset your password.")
                else:
                    user_info = st.session_state.users_store.get(login_user.strip().lower())
                    if user_info and verify_password(user_info["password"], login_pass):
                        st.session_state.failed_login_attempts = 0
                        # Auto-upgrade stored plaintext password to salted SHA-256 hash if needed
                        if user_info["password"] == login_pass:
                            st.session_state.users_store[login_user.strip().lower()]["password"] = hash_password(login_pass)
                            save_json_store(USERS_JSON, st.session_state.users_store)

                        st.session_state.current_user = {
                            "username": login_user.strip().lower(),
                            "fullName": user_info["fullName"],
                            "role": user_info["role"],
                            "childId": user_info.get("childId", ""),
                            "email": user_info.get("email", "")
                        }
                        st.success(f"Welcome back, {user_info['fullName']}!")
                        st.rerun()
                    else:
                        st.session_state.failed_login_attempts += 1
                        rem = max(0, 5 - st.session_state.failed_login_attempts)
                        st.error(f"❌ Invalid Username or Password. ({rem} login attempts remaining)")

        # ---- PARENT REGISTRATION TAB ----
        with register_tab:
            st.caption("Only **Parent** accounts can be registered here. Each child can have only **one** parent account.")
            with st.form("register_form"):
                reg_fullname = st.text_input("Full Name", value="", key="reg_name")
                reg_email = st.text_input("Email Address", value="", key="reg_email")
                reg_username = st.text_input("Choose Username", value="", key="reg_user")
                reg_password = st.text_input("Choose Password (Min 8 Characters)", type="password", value="", key="reg_pass")
                student_options = {f"{r['Child Name']} ({r['Child ID']})": r['Child ID'] for _, r in students_df.iterrows()}
                selected_student_str = st.selectbox("Select Your Child (Roll No / ID)", options=list(student_options.keys()))
                reg_child_id = student_options[selected_student_str]
                submit_reg = st.form_submit_button("📝 Register Parent Account", use_container_width=True)

            if submit_reg:
                u_key = reg_username.strip().lower()
                if not u_key or not reg_password:
                    st.warning("Please provide both a username and password.")
                elif len(reg_password) < 8:
                    st.error("⚠️ Password must be at least 8 characters long for security.")
                elif u_key in st.session_state.users_store:
                    st.error("Username already exists. Please choose a different username.")
                else:
                    # One-parent-per-child check
                    has_parent, existing_user = child_has_parent(reg_child_id)
                    if has_parent:
                        child_row = get_child_row(reg_child_id)
                        cname = child_row['Child Name'] if child_row is not None else reg_child_id
                        st.error(f"⚠️ An account already exists for **{cname} ({reg_child_id})**. Only one parent account is allowed per child.")
                    else:
                        st.session_state.users_store[u_key] = {
                            "password": hash_password(reg_password),
                            "fullName": reg_fullname if reg_fullname else u_key,
                            "role": "Parent",
                            "childId": reg_child_id,
                            "email": reg_email.strip()
                        }
                        save_json_store(USERS_JSON, st.session_state.users_store)
                        st.success("✅ Parent account registered successfully! You can now Log In.")

        # ---- TEACHER REGISTRATION TAB (Admin Code Required) ----
        with teacher_reg_tab:
            st.caption("Teacher accounts require **admin approval**. Request an invite code, then use it to register.")
            st.markdown("---")

            # Step 1: Request invite code
            st.markdown("**Step 1:** Request an invite code from the Admin")
            with st.form("teacher_request_form"):
                tr_fullname = st.text_input("Your Full Name", value="", key="tr_name")
                tr_email = st.text_input("Your Email Address", value="", key="tr_email")
                tr_submit = st.form_submit_button("📧 Request Teacher Invite Code", use_container_width=True)

            if tr_submit:
                if not tr_email.strip() or not tr_fullname.strip():
                    st.warning("Please provide your name and email.")
                else:
                    # Check if already requested
                    already = any(p['email'].lower() == tr_email.strip().lower() and p['status'] == 'pending' for p in st.session_state.pending_teacher_store)
                    if already:
                        st.info("A request from this email is already pending. Please wait for admin approval.")
                    else:
                        new_request = {
                            "fullName": tr_fullname.strip(),
                            "email": tr_email.strip(),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "status": "pending"
                        }
                        st.session_state.pending_teacher_store.append(new_request)
                        save_json_store(PENDING_TEACHER_JSON, st.session_state.pending_teacher_store)
                        st.success("✅ Teacher access request submitted! The admin will review and send an invite code to your email.")

            st.markdown("---")
            # Step 2: Register with invite code
            st.markdown("**Step 2:** Enter the invite code to create your Teacher account")
            with st.form("teacher_code_form"):
                tc_code = st.text_input("Enter 6-Digit Invite Code", value="", key="tc_code")
                tc_username = st.text_input("Choose Username", value="", key="tc_user")
                tc_password = st.text_input("Choose Password (Min 8 Characters)", type="password", value="", key="tc_pass")
                tc_submit = st.form_submit_button("🔓 Verify Code & Register", use_container_width=True)

            if tc_submit:
                tc_key = tc_username.strip().lower()
                if not tc_key or not tc_password or not tc_code.strip():
                    st.warning("Please fill in all fields.")
                elif len(tc_password) < 8:
                    st.error("⚠️ Password must be at least 8 characters long for security.")
                elif tc_key in st.session_state.users_store:
                    st.error("Username already exists.")
                else:
                    # Reload fresh codes from disk to avoid in-memory session lag
                    st.session_state.verification_codes_store = load_json_store(VERIFICATION_CODES_JSON, [])
                    valid_code = None
                    for vc in st.session_state.verification_codes_store:
                        if str(vc.get("code", "")).strip() == tc_code.strip() and str(vc.get("purpose", "")).lower() == "teacher registration" and not vc.get("used"):
                            valid_code = vc
                            break
                    if valid_code:
                        valid_code["used"] = True
                        save_json_store(VERIFICATION_CODES_JSON, st.session_state.verification_codes_store)
                        # Mark pending request as approved/registered
                        st.session_state.pending_teacher_store = load_json_store(PENDING_TEACHER_JSON, [])
                        for pr in st.session_state.pending_teacher_store:
                            if pr.get("email", "").lower() == valid_code.get("email", "").lower():
                                pr["status"] = "registered"
                        save_json_store(PENDING_TEACHER_JSON, st.session_state.pending_teacher_store)

                        st.session_state.users_store = load_json_store(USERS_JSON, DEFAULT_USERS)
                        st.session_state.users_store[tc_key] = {
                            "password": hash_password(tc_password),
                            "fullName": valid_code.get("fullName", tc_key),
                            "role": "Teacher",
                            "childId": "",
                            "email": valid_code.get("email", "")
                        }
                        save_json_store(USERS_JSON, st.session_state.users_store)
                        st.success("✅ Teacher account created successfully! You can now Log In.")
                    else:
                        st.error("❌ Invalid or expired invite code. Please contact the admin.")

        # ---- FORGOT PASSWORD TAB ----
        with forgot_tab:
            st.caption("Enter your username and registered email to receive a password reset code.")

            # Step 1: Request reset code
            st.markdown("**Step 1:** Request a reset code")
            with st.form("forgot_step1_form"):
                fp_username = st.text_input("Your Username", value="", key="fp_user")
                fp_email = st.text_input("Your Registered Email", value="", key="fp_email")
                fp_submit = st.form_submit_button("📧 Send Reset Code to Email", use_container_width=True)

            if fp_submit:
                fp_key = fp_username.strip().lower()
                st.session_state.users_store = load_json_store(USERS_JSON, DEFAULT_USERS)
                user_data = st.session_state.users_store.get(fp_key)
                if not user_data:
                    st.error("❌ Username not found.")
                elif not fp_email.strip():
                    st.warning("Please enter your email.")
                else:
                    code = generate_verification_code()
                    code_entry = {
                        "code": code,
                        "username": fp_key,
                        "email": fp_email.strip(),
                        "purpose": "Password Reset",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "used": False
                    }
                    st.session_state.verification_codes_store = load_json_store(VERIFICATION_CODES_JSON, [])
                    st.session_state.verification_codes_store.append(code_entry)
                    save_json_store(VERIFICATION_CODES_JSON, st.session_state.verification_codes_store)
                    sent = send_verification_email(fp_email.strip(), code, "Password Reset")
                    if sent:
                        st.success(f"✅ A 6-digit reset code has been sent to **{fp_email.strip()}**. Check your inbox (and spam folder).")
                    else:
                        st.warning(f"Could not send email. Please contact the admin. Your code for manual verification: ask admin.")

            st.markdown("---")
            # Step 2: Enter code & new password
            st.markdown("**Step 2:** Enter the reset code and set your new password")
            with st.form("forgot_step2_form"):
                fp2_username = st.text_input("Username", value="", key="fp2_user")
                fp2_code = st.text_input("Enter 6-Digit Reset Code", value="", key="fp2_code")
                fp2_newpass = st.text_input("New Password (Min 8 Characters)", type="password", value="", key="fp2_pass")
                fp2_submit = st.form_submit_button("🔐 Reset Password", use_container_width=True)

            if fp2_submit:
                fp2_key = fp2_username.strip().lower()
                st.session_state.users_store = load_json_store(USERS_JSON, DEFAULT_USERS)
                if not fp2_key or not fp2_code.strip() or not fp2_newpass:
                    st.warning("Please fill in all fields.")
                elif len(fp2_newpass) < 8:
                    st.error("⚠️ New password must be at least 8 characters long for security.")
                elif fp2_key not in st.session_state.users_store:
                    st.error("❌ Username not found.")
                else:
                    st.session_state.verification_codes_store = load_json_store(VERIFICATION_CODES_JSON, [])
                    valid_code = None
                    for vc in st.session_state.verification_codes_store:
                        if (str(vc.get("code", "")).strip() == fp2_code.strip() and
                            vc.get("username", "").lower() == fp2_key and
                            str(vc.get("purpose", "")).lower() == "password reset" and
                            not vc.get("used")):
                            valid_code = vc
                            break
                    if valid_code:
                        valid_code["used"] = True
                        save_json_store(VERIFICATION_CODES_JSON, st.session_state.verification_codes_store)
                        st.session_state.users_store[fp2_key]["password"] = hash_password(fp2_newpass)
                        save_json_store(USERS_JSON, st.session_state.users_store)
                        st.success("✅ Password has been reset successfully! You can now Log In with your new password.")
                    else:
                        st.error("❌ Invalid or expired reset code.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()


# ------------------- ABOUT PAGE COMPONENT -------------------
def render_about_page():
    st.markdown("<div class='section-header'>ℹ️ About AI Child Growth Advisor & Development Team</div>", unsafe_allow_html=True)

    # Lead Developers Card
    st.markdown("### 👨‍💻 Project Development Team")
    col_dev1, col_dev2 = st.columns(2)
    with col_dev1:
        st.markdown("""
        <div class="custom-card" style="border-top: 4px solid #6366f1; text-align: center; padding: 1.5rem 1rem;">
            <div style="font-size: 3rem;">👨‍💻</div>
            <div style="font-size: 1.35rem; font-weight: 800; color: #f8fafc; margin-top: 0.4rem; letter-spacing: 0.02em;">SACHIN M</div>
            <div style="font-size: 0.88rem; color: #818cf8; font-weight: 700; text-transform: uppercase; margin-top: 0.2rem;">Lead AI Engineer & System Architect</div>
            <div style="font-size: 0.88rem; color: #94a3b8; margin-top: 0.6rem; line-height: 1.5;">
                Deep Learning Model Optimization, GrowthNet 2.0 Neural Network, WHO LMS Standards Engine & Backend Systems.
            </div>
            <div style="margin-top: 0.8rem;">
                <span class="badge badge-info">PyTorch & Deep Learning</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_dev2:
        st.markdown("""
        <div class="custom-card" style="border-top: 4px solid #38bdf8; text-align: center; padding: 1.5rem 1rem;">
            <div style="font-size: 3rem;">👨‍💻</div>
            <div style="font-size: 1.35rem; font-weight: 800; color: #f8fafc; margin-top: 0.4rem; letter-spacing: 0.02em;">VISHAL KANNAN S I</div>
            <div style="font-size: 0.88rem; color: #38bdf8; font-weight: 700; text-transform: uppercase; margin-top: 0.2rem;">Lead Full-Stack Developer & UI/UX Architect</div>
            <div style="font-size: 0.88rem; color: #94a3b8; margin-top: 0.6rem; line-height: 1.5;">
                Interactive AI Meal Planner, Clinical Pediatric Standards Integration, Streamlit Dashboard UI & Multi-Portal Architecture.
            </div>
            <div style="margin-top: 0.8rem;">
                <span class="badge badge-healthy">Full-Stack & UX</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Details about the site
    st.markdown("### 🌐 About the Platform")
    st.markdown("""
    The **AI Child Growth Advisor & Pediatric Health Portal** is an advanced health and nutrition monitoring platform created specifically for **children aged 1 to 5 years (12–60 months)**. 

    Combining official **World Health Organization (WHO) Child Growth Standards (0–5y)** and **CDC/IAP pediatric BMI percentiles** with a deep neural network (**GrowthNet 2.0**), this platform bridges pediatric medical benchmarks with actionable, parent-friendly daily care and nutrition planning.
    """)

    st.markdown("### 🚀 Core Platform Modules")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("""
        <div class="custom-card">
            <div style="font-weight: 700; color: #818cf8; font-size: 1.05rem;">🧠 PyTorch GrowthNet 2.0 AI Engine</div>
            <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 0.3rem;">
                4-layer deep neural network trained on 29,000+ pediatric benchmark cases with 9 domain-engineered features achieving <b>99.0% diagnostic test accuracy</b> across all WHO health classifications.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-card">
            <div style="font-weight: 700; color: #38bdf8; font-size: 1.05rem;">🍽️ Interactive AI Daily Meal Planner</div>
            <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 0.3rem;">
                Structured 5-meal daily schedules (Breakfast, Snack, Lunch, Snack, Dinner) tailored to ICMR-NIN RDA 2024, AskNestle, and Harvard Healthy Eating Plate guidelines with Precision Nutrition hand-size portion guides.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-card">
            <div style="font-weight: 700; color: #34d399; font-size: 1.05rem;">📋 Cumulative Multi-Day Attendance</div>
            <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 0.3rem;">
                Persistent attendance tracking with individual student cumulative percentage scoring, low-attendance alerts (<75%), and academic year reset tools.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown("""
        <div class="custom-card">
            <div style="font-weight: 700; color: #f59e0b; font-size: 1.05rem;">🎓 Student Database Management (CRUD)</div>
            <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 0.3rem;">
                Live Excel & JSON registry synchronization allowing teachers to add, edit, and delete student records with instant roster export.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-card">
            <div style="font-weight: 700; color: #ec4899; font-size: 1.05rem;">📄 Automated Clinical PDF Reports</div>
            <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 0.3rem;">
                Instant ReportLab PDF generation for growth analysis curves and downloadable printable 5-meal daily nutrition schedules.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-card">
            <div style="font-weight: 700; color: #a855f7; font-size: 1.05rem;">💬 Secure Parent-Teacher Messaging</div>
            <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 0.3rem;">
                Direct two-way progress notes and child dietary coordination between class teachers and parents.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Technology Stack
    st.markdown("### 🛠️ Built With Modern Technology")
    tech_cols = st.columns(6)
    tech_cols[0].markdown("<div class='custom-card' style='text-align:center; padding:0.6rem;'>🔥<br><b>PyTorch</b></div>", unsafe_allow_html=True)
    tech_cols[1].markdown("<div class='custom-card' style='text-align:center; padding:0.6rem;'>🐍<br><b>Python 3.11</b></div>", unsafe_allow_html=True)
    tech_cols[2].markdown("<div class='custom-card' style='text-align:center; padding:0.6rem;'>⚡<br><b>Streamlit</b></div>", unsafe_allow_html=True)
    tech_cols[3].markdown("<div class='custom-card' style='text-align:center; padding:0.6rem;'>🔬<br><b>Scikit-Learn</b></div>", unsafe_allow_html=True)
    tech_cols[4].markdown("<div class='custom-card' style='text-align:center; padding:0.6rem;'>📄<br><b>ReportLab</b></div>", unsafe_allow_html=True)
    tech_cols[5].markdown("<div class='custom-card' style='text-align:center; padding:0.6rem;'>📊<br><b>Pandas</b></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Official Medical and Legal Disclaimer
    st.markdown("### ⚠️ Official Medical & Legal Disclaimer")
    st.markdown("""
    <div style="background: rgba(239, 68, 68, 0.08); border: 2px solid rgba(239, 68, 68, 0.3); border-radius: 14px; padding: 1.25rem 1.5rem; line-height: 1.7; color: #e2e8f0;">
        <div style="font-size: 1.1rem; font-weight: 800; color: #f87171; display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span>🛡️</span> Medical & Legal Notice
        </div>
        <p style="margin: 0 0 0.6rem 0; font-size: 0.92rem;">
            <b>1. Informational & Screening Purpose Only:</b> The AI Child Growth Advisor, its neural network predictions (GrowthNet 2.0), WHO percentile curves, nutritional meal recommendations, and PDF growth charts are generated for informational screening and growth-monitoring support only.
        </p>
        <p style="margin: 0 0 0.6rem 0; font-size: 0.92rem;">
            <b>2. Not Clinical Medical Advice:</b> This application does <b>not</b> provide medical diagnosis, therapeutic prescription, or clinical treatment plans. It is not intended to replace consultation, diagnosis, or treatment by a licensed pediatrician, healthcare provider, or clinical pediatric dietitian.
        </p>
        <p style="margin: 0; font-size: 0.92rem;">
            <b>3. Parental & Clinical Consultation:</b> Parents and educators should always seek the direct advice of a qualified pediatrician regarding any medical conditions, feeding difficulties, severe stunting/wasting flags, allergic reactions, or specific developmental health concerns.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ------------------- LOGGED IN SESSION & SIDEBAR -------------------
current_user = st.session_state.get("current_user")
if not current_user:
    st.stop()

user_role = current_user.get("role", "")

st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 0.85rem; padding: 0.6rem 0 0.4rem;">
    <div style="background: #FDFBD4; padding: 4px; border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 14px rgba(0,0,0,0.4); border: 1px solid #C05800;">
        <img src="data:image/png;base64,{LOGO_B64}" width="38" height="38" style="border-radius: 8px; object-fit: contain;" alt="Growth Advisor Logo">
    </div>
    <div>
        <div style="font-size: 1.32rem; font-weight: 800; color: #FDFBD4; letter-spacing: -0.02em; line-height: 1.15;">Growth Advisor</div>
        <div style="font-size: 0.72rem; color: #C05800; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">AI-POWERED PORTAL</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(113,54,0,0.45), rgba(192,88,0,0.2)); border: 1px solid rgba(192,88,0,0.35); padding: 0.75rem 0.9rem; border-radius: 12px; margin: 0.8rem 0 0.5rem;">
    <div style="font-weight: 700; color: {'#FDFBD4' if st.session_state.app_theme == 'dark' else '#38240D'}; font-size: 0.95rem;">👤 {current_user['fullName']}</div>
    <div style="font-size: 0.8rem; color: #C05800; margin-top: 0.15rem; font-weight: 600;">Role: <b style="color:{'#FDFBD4' if st.session_state.app_theme == 'dark' else '#38240D'};">{user_role}</b></div>
    {f'<div style="font-size: 0.8rem; color: #713600; margin-top: 0.1rem;">Roll No: <b style="color:#C05800;">{current_user["childId"]}</b></div>' if user_role == 'Parent' else ''}
</div>
""", unsafe_allow_html=True)

# Theme Toggle Button
cur_theme = st.session_state.get("app_theme", "dark")
theme_btn_label = "☀️ Switch to Light Theme" if cur_theme == "dark" else "🌙 Switch to Dark Theme"
if st.sidebar.button(theme_btn_label, key="theme_toggle_btn", use_container_width=True):
    st.session_state.app_theme = "light" if cur_theme == "dark" else "dark"
    st.rerun()

# ---- Navigation menu with styled buttons ----
if "active_nav" not in st.session_state:
    st.session_state.active_nav = None

if user_role == "Teacher":
    nav_options = [
        ("🏠 Home / Overview",        "home"),
        ("👶 Children Directory",     "children"),
        ("🎓 Student Database (CRUD)", "student_crud"),
        ("📋 Attendance Sheet",       "attendance"),
        ("⚡ Generate Report",        "generate"),
        ("📄 Reports Repository",     "reports"),
        ("🍽️ AI Meal Planner",        "meal_planner"),
        ("🥗 Food Recommendations",   "food"),
        ("📅 Meal Schedule",          "meal"),
        ("💬 Messages",               "messages"),
        ("🛡️ Admin Panel",            "admin_panel"),
        ("ℹ️ About Project",          "about"),
        ("⚙️ Account Settings",       "settings"),
    ]
    nav_label_map = {
        "home":       "🏠 Home / Overview",
        "children":   "👶 Children Directory",
        "student_crud": "🎓 Student Database (CRUD)",
        "attendance": "📋 Teacher Attendance Sheet",
        "generate":   "⚡ Generate Growth Report",
        "reports":    "📄 Student Reports Repository",
        "meal_planner": "🍽️ AI Daily Meal Planner",
        "food":       "🥗 Food Recommendations",
        "meal":       "📅 Weekly Meal Schedule",
        "messages":   "💬 Parent-Teacher Messages",
        "admin_panel": "🛡️ Admin Panel",
        "about":       "ℹ️ About the Project & Team",
        "settings":    "⚙️ Account Settings",
    }
else:
    nav_options = [
        ("🏠 My Child Profile",       "home"),
        ("📋 Attendance & Stats",     "attendance"),
        ("📄 Growth Reports",         "reports"),
        ("⚡ Generate Report",        "generate"),
        ("🍽️ AI Meal Planner",        "meal_planner"),
        ("🥗 Food Recommendations",   "food"),
        ("📅 Meal Schedule",          "meal"),
        ("💬 Chat with Teacher",      "messages"),
        ("ℹ️ About Project",          "about"),
        ("⚙️ Account Settings",       "settings"),
    ]
    nav_label_map = {
        "home":       "🏠 My Child Profile",
        "attendance": "📋 My Child's Attendance & Stats",
        "reports":    "📄 My Child's Growth Reports",
        "generate":   "⚡ Generate Report for My Child",
        "meal_planner": "🍽️ My Child's AI Meal Planner",
        "food":       "🥗 Food Recommendations",
        "meal":       "📅 Weekly Meal Schedule",
        "messages":   "💬 Chat with Teacher",
        "about":      "ℹ️ About the Project & Team",
        "settings":   "⚙️ Account Settings",
    }

default_key = nav_options[0][1]
if st.session_state.active_nav is None or st.session_state.active_nav not in nav_label_map:
    st.session_state.active_nav = default_key

st.sidebar.markdown('<div class="sidebar-nav-label">Navigation</div>', unsafe_allow_html=True)

for label, key in nav_options:
    is_active = (st.session_state.active_nav == key)
    btn_type = "primary" if is_active else "secondary"
    if st.sidebar.button(label, key=f"nav_{key}", type=btn_type, use_container_width=True):
        st.session_state.active_nav = key
        st.rerun()

st.sidebar.markdown("---")

# Logout button
logout_container = st.sidebar.container()
logout_container.markdown('<div class="logout-btn">', unsafe_allow_html=True)
if logout_container.button("🚪 Log Out", key="logout_btn", use_container_width=True):
    st.session_state.current_user = None
    st.session_state.active_nav = None
    st.rerun()
logout_container.markdown('</div>', unsafe_allow_html=True)

# Resolve the active page label for downstream comparisons
nav_selection = nav_label_map.get(st.session_state.active_nav, nav_label_map[default_key])

# ==============================================================================
# TEACHER VIEWS
# ==============================================================================
if user_role == "Teacher":

    # 1. Teacher Home
    if nav_selection == "🏠 Home / Overview":
        st.markdown(f"""
        <div class="hero-banner">
            <div style="display: flex; align-items: center; gap: 1.1rem; margin-bottom: 0.8rem;">
                <div style="background: #ffffff; padding: 6px; border-radius: 16px; display: inline-flex; align-items: center; justify-content: center; box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25); border: 2px solid rgba(255, 255, 255, 0.4);">
                    <img src="data:image/png;base64,{LOGO_B64}" width="50" height="50" style="border-radius: 12px; object-fit: contain;" alt="Growth Advisor Logo">
                </div>
                <div>
                    <div class="hero-title" style="margin-bottom: 0.1rem; line-height: 1.15;">Growth Advisor</div>
                    <div style="color: #a5b4fc; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">AI-POWERED PORTAL — TEACHER & ADMIN DASHBOARD</div>
                </div>
            </div>
            <div class="hero-subtitle">
                Logged in as <b>{current_user['fullName']}</b>. Manage daily attendance, generate and view student PDF growth reports, check WHO nutrition recommendations, and communicate directly with parents.
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="metric-card"><div class="metric-val">30</div><div class="metric-lbl">Enrolled Children</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-val">{len(st.session_state.reports_store)}</div><div class="metric-lbl">PDF Reports Stored</div></div>', unsafe_allow_html=True)
        c3.markdown('<div class="metric-card"><div class="metric-val">6 Days</div><div class="metric-lbl">Meal Schedule</div></div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 14px; padding: 1rem 1.25rem; margin-top: 1.5rem; display: flex; align-items: flex-start; gap: 0.9rem;">
            <span style="font-size: 1.4rem; line-height: 1;">⚠️</span>
            <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;">
                <b style="color: #f87171; font-size: 0.92rem;">Clinical & Medical Disclaimer:</b><br>
                The AI Child Growth Advisor provides informational and screening assistance derived from World Health Organization (WHO) Growth Standards and deep learning assessment models. <b>This tool does not provide medical diagnoses or replace professional pediatric healthcare.</b> Always consult a certified pediatrician or qualified medical professional for clinical evaluations and health decisions.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Children Directory
    elif nav_selection == "👶 Children Directory":
        st.markdown("<div class='section-header'>👶 Enrolled Students Directory (30 Children)</div>", unsafe_allow_html=True)
        search_query = st.text_input("🔍 Search Student by Name, Child ID, Parent, or Location", placeholder="e.g. C001, Arun, Peelamedu")
        
        filtered_df = students_df.copy()
        if search_query:
            mask = filtered_df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
            filtered_df = filtered_df[mask]

        st.dataframe(filtered_df[['Child ID', 'Child Name', 'Parent Name', 'Place', 'Phone Number']], use_container_width=True, hide_index=True)

        st.markdown("### 📇 Student Cards")
        grid_cols = st.columns(3)
        for idx, row in filtered_df.reset_index().iterrows():
            col = grid_cols[idx % 3]
            with col:
                st.markdown(f"""
                <div class="custom-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="card-heading" style="font-size: 1.1rem; font-weight: 700;">{row['Child Name']}</span>
                        <span class="badge badge-info">{row['Child ID']}</span>
                    </div>
                    <div class="card-subtext" style="font-size: 0.9rem; margin-top: 0.5rem; line-height: 1.6;">
                        👨‍👩‍👧 Parent: <b>{row['Parent Name']}</b><br>
                        📍 Location: <b>{row['Place']}</b><br>
                        📞 Phone: <b>{row['Phone Number']}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # 2.5 Student Database Management (CRUD)
    elif nav_selection == "🎓 Student Database (CRUD)":
        st.markdown("<div class='section-header'>🎓 Student Database & Registry Management (CRUD)</div>", unsafe_allow_html=True)
        st.caption("Add new students, update existing records, delete former students, and manage the student registry.")

        crud_tab1, crud_tab2, crud_tab3, crud_tab4 = st.tabs([
            "➕ Add New Student",
            "✏️ Edit / Update Student",
            "🗑️ Delete Student",
            "📋 Master Registry & Export"
        ])

        with crud_tab1:
            st.markdown("### ➕ Register New Student into Database")
            with st.form("add_student_form", clear_on_submit=True):
                existing_ids = [str(x).strip().upper() for x in students_df['Child ID'].tolist()]
                max_num = 0
                for cid_val in existing_ids:
                    nums = re.findall(r"\d+", cid_val)
                    if nums:
                        max_num = max(max_num, int(nums[0]))
                suggested_id = f"C{max_num+1:03d}"

                col_as1, col_as2 = st.columns(2)
                with col_as1:
                    new_cid = st.text_input("Child ID (Roll No)", value=suggested_id, help="Must be unique")
                    new_cname = st.text_input("Child Full Name *", placeholder="e.g. Diya Sharma")
                with col_as2:
                    new_pname = st.text_input("Parent / Guardian Name *", placeholder="e.g. Rajesh Sharma")
                    new_place = st.text_input("Location / Place", value="Coimbatore")

                new_phone = st.text_input("Contact Phone Number *", placeholder="e.g. 9876543210")
                submit_add = st.form_submit_button("➕ Save Student to Database", use_container_width=True)

            if submit_add:
                new_cid_clean = str(new_cid).strip().upper()
                new_cname_clean = str(new_cname).strip()
                new_pname_clean = str(new_pname).strip()
                new_phone_clean = str(new_phone).strip()

                if not new_cid_clean or not new_cname_clean or not new_pname_clean:
                    st.error("❌ Child ID, Child Name, and Parent Name are required.")
                elif new_cid_clean in existing_ids:
                    st.error(f"❌ Student ID '{new_cid_clean}' already exists! Please choose a unique ID.")
                else:
                    new_row = pd.DataFrame([{
                        "Child ID": new_cid_clean,
                        "Child Name": new_cname_clean,
                        "Parent Name": new_pname_clean,
                        "Place": new_place.strip() or "Coimbatore",
                        "Phone Number": int(new_phone_clean) if new_phone_clean.isdigit() else new_phone_clean
                    }])
                    updated_students_df = pd.concat([students_df, new_row], ignore_index=True)
                    if save_students_data(updated_students_df):
                        parent_username = new_cid_clean.lower()
                        if parent_username not in st.session_state.users_store:
                            st.session_state.users_store[parent_username] = {
                                "password": hash_password("parent123"),
                                "fullName": new_pname_clean,
                                "role": "Parent",
                                "childId": new_cid_clean,
                                "email": f"{parent_username}@school.com"
                            }
                            save_json_store(USERS_JSON, st.session_state.users_store)
                        st.success(f"🎉 Successfully added student **{new_cname_clean} ({new_cid_clean})** to database!")
                        st.rerun()

        with crud_tab2:
            st.markdown("### ✏️ Edit & Update Existing Student Details")
            st_list = {f"{r['Child Name']} ({r['Child ID']})": r['Child ID'] for _, r in students_df.iterrows()}
            selected_edit_label = st.selectbox("Select Student to Edit", options=list(st_list.keys()), key="crud_select_edit")
            selected_edit_cid = st_list[selected_edit_label]
            curr_row = students_df[students_df['Child ID'].astype(str) == str(selected_edit_cid)].iloc[0]

            with st.form("edit_student_form"):
                col_es1, col_es2 = st.columns(2)
                with col_es1:
                    edit_cid_disp = st.text_input("Child ID (Locked)", value=str(curr_row['Child ID']), disabled=True)
                    edit_cname = st.text_input("Child Full Name", value=str(curr_row['Child Name']))
                with col_es2:
                    edit_pname = st.text_input("Parent / Guardian Name", value=str(curr_row['Parent Name']))
                    edit_place = st.text_input("Location / Place", value=str(curr_row.get('Place', 'Coimbatore')))
                
                edit_phone = st.text_input("Contact Phone Number", value=str(curr_row.get('Phone Number', '')))
                submit_edit = st.form_submit_button("💾 Save Updates to Database", use_container_width=True)

            if submit_edit:
                students_df.loc[students_df['Child ID'].astype(str) == str(selected_edit_cid), 'Child Name'] = edit_cname.strip()
                students_df.loc[students_df['Child ID'].astype(str) == str(selected_edit_cid), 'Parent Name'] = edit_pname.strip()
                students_df.loc[students_df['Child ID'].astype(str) == str(selected_edit_cid), 'Place'] = edit_place.strip()
                students_df.loc[students_df['Child ID'].astype(str) == str(selected_edit_cid), 'Phone Number'] = int(edit_phone.strip()) if edit_phone.strip().isdigit() else edit_phone.strip()
                
                if save_students_data(students_df):
                    st.success(f"✅ Successfully updated details for **{edit_cname} ({selected_edit_cid})**!")
                    st.rerun()

        with crud_tab3:
            st.markdown("### 🗑️ Delete Student from Database")
            selected_del_label = st.selectbox("Select Student to Delete", options=list(st_list.keys()), key="crud_select_del")
            selected_del_cid = st_list[selected_del_label]
            del_row = students_df[students_df['Child ID'].astype(str) == str(selected_del_cid)].iloc[0]

            st.warning(f"⚠️ You are about to delete **{del_row['Child Name']} (ID: {selected_del_cid})** from the school database. This will permanently remove their records.")
            del_confirm = st.checkbox(f"I confirm that I want to permanently delete {del_row['Child Name']} ({selected_del_cid})")
            
            if st.button("🗑️ Permanently Delete Student", type="primary", disabled=not del_confirm):
                pruned_df = students_df[students_df['Child ID'].astype(str) != str(selected_del_cid)].copy()
                if save_students_data(pruned_df):
                    for dt, att_map in st.session_state.attendance_store.items():
                        if selected_del_cid in att_map:
                            del att_map[selected_del_cid]
                    save_json_store(ATTENDANCE_JSON, st.session_state.attendance_store)
                    st.success(f"🗑️ Student {del_row['Child Name']} ({selected_del_cid}) was deleted.")
                    st.rerun()

        with crud_tab4:
            st.markdown(f"### 📋 Master Student Directory ({len(students_df)} Students Enrolled)")
            st.dataframe(students_df, use_container_width=True, hide_index=True)
            
            excel_buf = BytesIO()
            students_df.to_excel(excel_buf, index=False)
            st.download_button(
                label="📥 Download Master Student Registry (Excel)",
                data=excel_buf.getvalue(),
                file_name="Child_Attendance_Data_Updated.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    # 3. Teacher Attendance Sheet
    elif nav_selection == "📋 Teacher Attendance Sheet":
        st.markdown("<div class='section-header'>📋 Student Attendance Management & Historical Tracking</div>", unsafe_allow_html=True)

        att_summary_df = compute_class_attendance_summary(students_df, st.session_state.attendance_store)
        total_dates_logged = len(st.session_state.attendance_store)
        
        total_all_records = sum(len(d) for d in st.session_state.attendance_store.values())
        total_all_presents = sum(sum(1 for v in d.values() if v == "Present") for d in st.session_state.attendance_store.values())
        class_overall_pct = (total_all_presents / total_all_records * 100.0) if total_all_records > 0 else 0.0
        low_att_count = len(att_summary_df[att_summary_df["Raw Pct"] < 75.0]) if total_dates_logged > 0 else 0

        tm1, tm2, tm3, tm4 = st.columns(4)
        tm1.metric("Enrolled Students", len(students_df))
        tm2.metric("Total Days Logged", f"{total_dates_logged} Days")
        tm3.metric("Class Attendance Rate", f"{class_overall_pct:.1f}%")
        tm4.metric("At-Risk Students (<75%)", f"{low_att_count} Students", delta=f"-{low_att_count}" if low_att_count > 0 else "0", delta_color="inverse")

        st.markdown("---")

        att_tab1, att_tab2, att_tab3 = st.tabs([
            "📝 Daily Attendance Sheet",
            "📊 Cumulative Class Attendance Roster",
            "📅 Master Date-by-Date Matrix"
        ])

        with att_tab1:
            st.markdown("### 📝 Mark / Review Daily Attendance")
            
            c_date1, c_date2 = st.columns([1.5, 2.5])
            with c_date1:
                selected_date = st.date_input("Select School Date", datetime.now(), key="att_date_picker").strftime("%Y-%m-%d")
            
            date_records = st.session_state.attendance_store.get(selected_date, {})
            
            with c_date2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                qa_col1, qa_col2 = st.columns(2)
                mark_all_p = qa_col1.button("🟢 Mark All Present", use_container_width=True)
                mark_all_a = qa_col2.button("🔴 Mark All Absent", use_container_width=True)
                
                if mark_all_p:
                    for _, r in students_df.iterrows():
                        date_records[str(r['Child ID'])] = "Present"
                    st.session_state.attendance_store[selected_date] = date_records
                    save_json_store(ATTENDANCE_JSON, st.session_state.attendance_store)
                    st.success(f"✅ Marked all {len(students_df)} students as Present for {selected_date}!")
                    st.rerun()

                if mark_all_a:
                    for _, r in students_df.iterrows():
                        date_records[str(r['Child ID'])] = "Absent"
                    st.session_state.attendance_store[selected_date] = date_records
                    save_json_store(ATTENDANCE_JSON, st.session_state.attendance_store)
                    st.warning(f"⚠️ Marked all {len(students_df)} students as Absent for {selected_date}.")
                    st.rerun()

            with st.expander("🔄 Academic Year Attendance Reset", expanded=False):
                st.markdown("Clear all recorded attendance logs to begin a fresh academic session.")
                confirm_reset = st.checkbox("I understand that this will erase all historical attendance records for the new school year.", key="chk_reset_att")
                if st.button("🚨 Reset All Attendance for New Academic Year", type="primary", disabled=not confirm_reset):
                    if reset_all_attendance_records():
                        st.success("🎉 All attendance records have been reset for the new academic year!")
                        st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            with st.form("teacher_daily_att_form"):
                st.markdown(f"#### 📋 Marking Sheet for **{selected_date}**")
                attendance_inputs = {}
                att_cols = st.columns(2)
                
                for idx, row in students_df.iterrows():
                    cid = str(row['Child ID'])
                    cname = str(row['Child Name'])
                    prev_status = date_records.get(cid, "Present")
                    child_stats = compute_child_attendance(cid, st.session_state.attendance_store)
                    
                    bar_color = "#10b981" if child_stats['percentage'] >= 90 else ("#f59e0b" if child_stats['percentage'] >= 75 else "#ef4444")
                    
                    c = att_cols[idx % 2]
                    with c:
                        st.markdown(f"""
                        <div class="custom-card" style="padding: 0.85rem 1rem; margin-bottom: 0.6rem; border-left: 4px solid {bar_color};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="card-heading" style="font-weight: 700; font-size: 1rem;">{cname} <span class="card-subtext" style="font-size: 0.8rem;">({cid})</span></span>
                                <span class="badge {child_stats['badge_cls']}">{child_stats['badge_text']}</span>
                            </div>
                            <div class="card-subtext" style="font-size: 0.82rem; margin-top: 0.3rem;">
                                Logged: <b>{child_stats['present_days']}/{child_stats['total_days']} Days</b> | 📞 {row.get('Phone Number', 'N/A')}
                            </div>
                            <div class="att-progress-bg">
                                <div class="att-progress-bar" style="width: {child_stats['percentage']}%; background: {bar_color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        status = st.radio(
                            f"Status for {cname}",
                            options=["Present", "Absent"],
                            index=0 if prev_status == "Present" else 1,
                            key=f"att_in_{selected_date}_{cid}",
                            horizontal=True,
                            label_visibility="collapsed"
                        )
                        attendance_inputs[cid] = status

                submit_att = st.form_submit_button(f"💾 Save Attendance for {selected_date}", use_container_width=True)

            if submit_att:
                st.session_state.attendance_store[selected_date] = attendance_inputs
                save_json_store(ATTENDANCE_JSON, st.session_state.attendance_store)
                st.success(f"✅ Attendance records saved for {selected_date}!")
                st.rerun()

            if selected_date in st.session_state.attendance_store:
                records = st.session_state.attendance_store[selected_date]
                t_count = len(records)
                p_count = sum(1 for v in records.values() if v == "Present")
                a_count = t_count - p_count
                d_pct = (p_count / t_count * 100) if t_count > 0 else 0
                st.markdown(f"**Day Summary ({selected_date}):** Present: **{p_count}** | Absent: **{a_count}** | Rate: **{d_pct:.1f}%**")

        with att_tab2:
            st.markdown("### 📊 Cumulative Class Attendance Roster")
            st.caption("Individual cumulative attendance percentages calculated across all recorded school dates.")
            
            filter_status = st.selectbox("Filter by Status", ["All Students", "Excellent (≥90%)", "Good (75–89%)", "At Risk (<75%)"])
            
            display_roster = att_summary_df.copy()
            if filter_status == "Excellent (≥90%)":
                display_roster = display_roster[display_roster["Raw Pct"] >= 90.0]
            elif filter_status == "Good (75–89%)":
                display_roster = display_roster[(display_roster["Raw Pct"] >= 75.0) & (display_roster["Raw Pct"] < 90.0)]
            elif filter_status == "At Risk (<75%)":
                display_roster = display_roster[display_roster["Raw Pct"] < 75.0]

            st.dataframe(
                display_roster[["Child ID", "Child Name", "Total Days Logged", "Present Days", "Absent Days", "Attendance %", "Status", "Parent Contact", "Location"]],
                use_container_width=True,
                hide_index=True
            )

        with att_tab3:
            st.markdown("### 📅 Master Date-by-Date Attendance Matrix")
            all_dates = sorted(st.session_state.attendance_store.keys(), reverse=True)
            if all_dates:
                matrix_rows = []
                for _, r in students_df.iterrows():
                    cid = str(r['Child ID'])
                    cname = str(r['Child Name'])
                    row_data = {"Child ID": cid, "Child Name": cname}
                    for d in all_dates:
                        row_data[d] = st.session_state.attendance_store.get(d, {}).get(cid, "—")
                    stats = compute_child_attendance(cid, st.session_state.attendance_store)
                    row_data["Cumulative %"] = f"{stats['percentage']:.1f}%"
                    matrix_rows.append(row_data)
                
                matrix_df = pd.DataFrame(matrix_rows)
                st.dataframe(matrix_df, use_container_width=True, hide_index=True)
                
                csv_bytes = matrix_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Attendance Matrix (CSV)",
                    data=csv_bytes,
                    file_name="Master_Attendance_Matrix.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No attendance records logged yet. Use the Daily Attendance Sheet tab to log records.")

    # 4. Generate Report Page (Teacher)
    elif nav_selection == "⚡ Generate Growth Report":
        st.markdown("<div class='section-header'>⚡ Generate New AI PDF Growth Report</div>", unsafe_allow_html=True)
        st.caption("Select student and enter measurements to run PyTorch GrowthNet model, compute WHO percentiles, and generate PDF report.")

        student_options = {f"{r['Child Name']} ({r['Child ID']})": (r['Child ID'], r['Child Name']) for _, r in students_df.iterrows()}
        selected_st = st.selectbox("Select Student", options=list(student_options.keys()))
        target_cid, target_cname = student_options[selected_st]

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            g_sex = st.radio("Sex", options=["Male", "Female"], horizontal=True)
            g_sex_code = "M" if g_sex == "Male" else "F"
            g_age_years = st.number_input("Age (in Years)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
            g_age_months = int(g_age_years * 12)
        with col_g2:
            g_height = st.number_input("Height (cm)", min_value=40.0, max_value=130.0, value=85.0, step=0.1)
            g_weight = st.number_input("Weight (kg)", min_value=1.0, max_value=40.0, value=12.0, step=0.1)

        btn_gen = st.button("⚡ Generate & Save PDF Growth Report", use_container_width=True)

        if btn_gen:
            with st.spinner("Analyzing WHO growth standards & generating PDF report..."):
                report_calc = generate_report(g_age_months, g_height, g_weight, g_sex_code, growth_model, model_scaler)
                pdf_buf = create_pdf_report(target_cname, g_age_months, report_calc)
                pdf_bytes = pdf_buf.getvalue()
                
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_filename = f"{target_cid}_{target_cname}_Report_{timestamp_str}.pdf"
                filepath = os.path.join(TEMP_REPORTS_DIR, pdf_filename)
                with open(filepath, "wb") as f:
                    f.write(pdf_bytes)

                new_rep_record = {
                    "reportId": f"R{len(st.session_state.reports_store)+1:03d}",
                    "childId": target_cid,
                    "childName": target_cname,
                    "title": f"{target_cname} Growth Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "aiStatus": report_calc["ai_status"],
                    "hfa_p": report_calc["hfa_p"],
                    "wfh_p": report_calc["wfh_p"],
                    "bmi": report_calc["bmi"],
                    "pdf_filename": pdf_filename
                }
                st.session_state.reports_store.insert(0, new_rep_record)
                save_json_store(REPORTS_JSON, st.session_state.reports_store)

            st.success(f"✅ Growth Report generated for {target_cname} ({target_cid})! Status: {report_calc['ai_status']}")
            
            c_res1, c_res2, c_res3 = st.columns(3)
            c_res1.metric("Status", report_calc['ai_status'])
            c_res2.metric("Height Percentile", f"P{report_calc['hfa_p']:.1f}")
            c_res3.metric("Weight/Height Percentile", f"P{report_calc['wfh_p']:.1f}")
            
            st.download_button(
                label=f"📄 Download PDF Report for {target_cname}",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                use_container_width=True
            )

    # 5. Teacher Reports Repository
    elif nav_selection == "📄 Student Reports Repository":
        st.markdown("<div class='section-header'>📄 Student Growth Reports Repository</div>", unsafe_allow_html=True)
        st.caption("Master repository of all generated PDF growth reports for all 30 enrolled students.")

        r_search = st.text_input("🔍 Search Reports by Child ID (e.g. C002) or Child Name", placeholder="e.g. C002, Kavya, Arun")

        all_reports = st.session_state.reports_store
        filtered_reports = all_reports
        if r_search.strip():
            q = r_search.strip().lower()
            filtered_reports = [r for r in all_reports if q in str(r.get('childId', '')).lower() or q in str(r.get('childName', '')).lower() or q in str(r.get('title', '')).lower()]

        st.markdown(f"**Showing {len(filtered_reports)} of {len(all_reports)} Generated PDF Reports**")

        for idx, rep in enumerate(filtered_reports):
            b_cls = get_badge_class(rep.get('aiStatus', 'Healthy'))
            st.markdown(f"""
            <div class="custom-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="card-heading" style="font-size: 1.2rem; font-weight: 700;">📄 {rep['title']}</span>
                    <span class="badge {b_cls}">{rep.get('aiStatus', 'Healthy')}</span>
                </div>
                <div class="card-meta" style="font-size: 0.9rem; margin-top: 0.6rem; display: flex; gap: 1.5rem; flex-wrap: wrap;">
                    <span>🆔 Roll No: <b>{rep['childId']}</b></span>
                    <span>🧒 Child Name: <b>{rep['childName']}</b></span>
                    <span>🕒 Generated: <b>{rep['timestamp']}</b></span>
                </div>
                <div class="card-subtext" style="margin-top: 0.6rem; font-size: 0.85rem;">
                    Height Pctl: <b>P{rep.get('hfa_p', 50):.1f}</b> | Weight/Height Pctl: <b>P{rep.get('wfh_p', 50):.1f}</b> | BMI: <b>{rep.get('bmi', 15.0):.1f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            pdf_bytes = get_pdf_bytes_for_report(rep)
            st.download_button(
                label=f"📄 Download PDF Report for {rep['childName']} ({rep['childId']})",
                data=pdf_bytes,
                file_name=f"{rep['childId']}_{rep['childName']}_Growth_Report.pdf",
                mime="application/pdf",
                key=f"dl_t_{rep.get('reportId', idx)}"
            )
            st.markdown("<br>", unsafe_allow_html=True)

    # 6. AI Daily Meal Planner (Teacher)
    elif nav_selection == "🍽️ AI Daily Meal Planner":
        st.markdown("<div class='section-header'>🍽️ AI Daily Meal Planner (1–5 Years)</div>", unsafe_allow_html=True)
        st.caption("Customized 5-meal daily pediatric schedule based on AskNestle, ICMR-NIN RDA 2024, WHO & Harvard Healthy Eating Plate guidelines.")

        st_col1, st_col2 = st.columns(2)
        with st_col1:
            student_options = {f"{r['Child Name']} ({r['Child ID']})": (r['Child ID'], r['Child Name']) for _, r in students_df.iterrows()}
            selected_st = st.selectbox("Select Enrolled Student (or choose Custom Input)", options=["Custom Child Profile"] + list(student_options.keys()), key="t_mp_child")
            
            if selected_st != "Custom Child Profile":
                target_cid, target_cname = student_options[selected_st]
                mp_child_name = target_cname
                rep_for_child = next((r for r in st.session_state.reports_store if str(r.get('childId', '')).upper() == target_cid.upper()), None)
                default_status = rep_for_child.get("aiStatus", "Healthy") if rep_for_child else "Healthy"
            else:
                target_cid, target_cname = "CUSTOM", "Child"
                mp_child_name = st.text_input("Child Name", value="Aarav", key="t_mp_custom_name")
                default_status = "Healthy"

            mp_sex = st.radio("Child Sex", ["Male", "Female"], horizontal=True, key="t_mp_sex")
            mp_age_years = st.slider("Child Age (Years)", min_value=1.0, max_value=5.0, value=2.5, step=0.1, key="t_mp_age_y")
            mp_age_months = float(mp_age_years * 12)

        with st_col2:
            status_options = ["Healthy", "Underweight", "Stunted", "Stunted & Underweight", "Overweight", "Obese"]
            stat_idx = status_options.index(default_status) if default_status in status_options else 0
            mp_status = st.selectbox("Current Growth Diagnostic Status", status_options, index=stat_idx, key="t_mp_status")
            mp_diet = st.selectbox("Dietary Preference", ["Vegetarian", "Eggetarian", "Non-Vegetarian"], key="t_mp_diet")
            mp_act = st.select_slider("Daily Activity Level", options=["Sedentary", "Moderate", "Active"], value="Moderate", key="t_mp_act")

        meal_plan_res = generate_custom_meal_plan(mp_age_months, mp_sex, mp_diet, mp_status, mp_act)

        st.markdown("---")
        st.markdown(f"### 📊 Daily Nutritional Targets for {mp_child_name} ({mp_diet})")
        
        target = meal_plan_res["target_rda"]
        tot = meal_plan_res["plan_totals"]
        
        m_c1, m_c2, m_c3, m_c4, m_c5, m_c6 = st.columns(6)
        m_c1.metric("Calories", f"{tot['calories']} kcal", f"Goal: {target['calories']}")
        m_c2.metric("Protein", f"{tot['protein_g']} g", f"RDA: {target['protein_g']}g")
        m_c3.metric("Healthy Fats", f"{tot['fat_g']} g", f"RDA: {target['fat_g']}g")
        m_c4.metric("Carbohydrates", f"{tot['carbs_g']} g", f"RDA: {target['carbs_g']}g")
        m_c5.metric("Calcium", f"{target['calcium_mg']} mg", "ICMR-NIN")
        m_c6.metric("Daily Water", f"{target['water_ml']} ml", "Hydration")

        st.markdown("---")
        st.markdown("### 🕒 Structured 5-Meal Daily Schedule (AskNestle & NIOS)")
        
        meal_icons = {
            "Breakfast": "🌅",
            "Mid-Morning": "🍎",
            "Lunch": "🍛",
            "Evening Snack": "🥛",
            "Dinner": "🍲"
        }
        
        for m_type, m_val in meal_plan_res["meals"].items():
            icon = meal_icons.get(m_type, "🍽️")
            st.markdown(f"""
            <div class="custom-card" style="margin-bottom: 0.9rem; border-left: 4px solid #C05800;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="card-heading" style="font-size: 1.15rem; font-weight: 700;">{icon} {m_type}: {m_val['name']}</span>
                    <span class="badge badge-info">~{m_val['cal']} kcal</span>
                </div>
                <div class="card-subtext" style="font-size: 0.92rem; margin-top: 0.4rem;">
                    {m_val['desc']}
                </div>
                <div class="card-meta" style="display: flex; flex-wrap: wrap; gap: 1.2rem; margin-top: 0.6rem; font-size: 0.85rem;">
                    <span>✋ <b>Portion:</b> {m_val['hand_portion']}</span>
                    <span>✨ <b>Key Nutrients:</b> {m_val['nutrients']}</span>
                    <span>🥩 <b>Protein:</b> {m_val['p']}g | <b>Carbs:</b> {m_val['c']}g | <b>Fats:</b> {m_val['f']}g</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🖐️ Precision Nutrition Visual Hand-Size Portion Guide")
        st.caption("How to measure child-appropriate portions quickly at home without weighing food:")
        
        hp_c1, hp_c2, hp_c3, hp_c4 = st.columns(4)
        hp_map = [
            (hp_c1, "Palm", "Protein", "Dal, Paneer, Eggs, Chicken, Fish", "✋", meal_plan_res["hand_portions"]["Palm"]["serving_1_3"] if mp_age_months < 36 else meal_plan_res["hand_portions"]["Palm"]["serving_3_5"]),
            (hp_c2, "Fist", "Veggies & Fruits", "Spinach, Carrots, Apple, Papaya", "✊", meal_plan_res["hand_portions"]["Fist"]["serving_1_3"] if mp_age_months < 36 else meal_plan_res["hand_portions"]["Fist"]["serving_3_5"]),
            (hp_c3, "Cupped Hand", "Grains & Carbs", "Roti, Rice, Poha, Dalia, Oats", "🤲", meal_plan_res["hand_portions"]["Cupped Hand"]["serving_1_3"] if mp_age_months < 36 else meal_plan_res["hand_portions"]["Cupped Hand"]["serving_3_5"]),
            (hp_c4, "Thumb", "Healthy Fats", "Desi Ghee, Butter, Nut Powders", "👍", meal_plan_res["hand_portions"]["Thumb"]["serving_1_3"] if mp_age_months < 36 else meal_plan_res["hand_portions"]["Thumb"]["serving_3_5"]),
        ]
        for col, hname, hnut, hexamples, hicon, hserv in hp_map:
            with col:
                st.markdown(f"""
                <div class="custom-card" style="text-align: center; padding: 1rem 0.8rem;">
                    <div style="font-size: 2rem;">{hicon}</div>
                    <div class="card-heading" style="font-weight: 700; margin-top: 0.3rem;">{hname} = {hnut}</div>
                    <div class="card-subtext" style="font-size: 0.8rem; margin-top: 0.2rem;">{hexamples}</div>
                    <div style="font-size: 0.82rem; color: #C05800; font-weight: 700; margin-top: 0.5rem; background: rgba(192, 88, 0, 0.12); padding: 4px; border-radius: 6px;">{hserv}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"### 💡 Pediatric Clinical Advice ({meal_plan_res['clinical_advice']['title']})")
        st.info(f"**Focus:** {meal_plan_res['clinical_advice']['focus']}")
        for tip in meal_plan_res['clinical_advice']['tips']:
            st.markdown(f"• {tip}")

        st.markdown("<br>", unsafe_allow_html=True)
        pdf_meal_buf = create_meal_plan_pdf(mp_child_name, meal_plan_res)
        st.download_button(
            label=f"📥 Download Printable Daily Meal Plan PDF for {mp_child_name}",
            data=pdf_meal_buf.getvalue(),
            file_name=f"{mp_child_name}_Daily_Meal_Plan.pdf",
            mime="application/pdf",
            key="t_dl_meal_plan_btn",
            use_container_width=True
        )

    # 7. Teacher Messages Portal
    elif nav_selection == "💬 Parent-Teacher Messages":
        st.markdown("<div class='section-header'>💬 Teacher Messaging Portal</div>", unsafe_allow_html=True)
        student_options = {f"{r['Child Name']} ({r['Child ID']})": r['Child ID'] for _, r in students_df.iterrows()}
        selected_label = st.selectbox("Select Student to Message", options=list(student_options.keys()))
        selected_cid = student_options[selected_label]

        with st.form("teacher_msg_form"):
            t_message = st.text_area("Write message to Parent", placeholder="Type progress note or reply...")
            t_submit = st.form_submit_button("✉️ Send Message to Parent")

        if t_submit:
            if t_message.strip():
                new_msg = {
                    "childId": selected_cid,
                    "childName": selected_label.split(" (")[0],
                    "senderRole": "Teacher",
                    "senderName": current_user['fullName'],
                    "message": t_message,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                st.session_state.messages_store.append(new_msg)
                save_json_store(MESSAGES_JSON, st.session_state.messages_store)
                st.success("✅ Message sent to Parent!")
                st.rerun()

        st.markdown("---")
        st.markdown(f"### 💬 Chat Stream for {selected_label}")
        child_msgs = [m for m in st.session_state.messages_store if m.get('childId') == selected_cid]
        if child_msgs:
            for m in child_msgs:
                s_role = m.get('senderRole', 'Teacher')
                bubble_cls = "chat-bubble-teacher" if s_role == "Teacher" else "chat-bubble-parent"
                color_hdr = "#818cf8" if s_role == "Teacher" else "#34d399"
                escaped_msg = html.escape(str(m['message']))
                escaped_name = html.escape(str(m.get('senderName', s_role)))
                st.markdown(f"""
                <div class="{bubble_cls}">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: {color_hdr};">
                        <span><b>{s_role}:</b> {escaped_name}</span>
                        <span>🕒 {m['timestamp']}</span>
                    </div>
                    <div style="font-size: 1rem; color: #f8fafc; margin-top: 0.4rem; white-space: pre-wrap;">
                        {escaped_msg}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # 9. Admin Panel (Teacher Only)
    elif nav_selection == "🛡️ Admin Panel":
        st.markdown("<div class='section-header'>🛡️ Teacher & Administrator Control Panel</div>", unsafe_allow_html=True)

        admin_tab1, admin_tab2, admin_tab3 = st.tabs([
            "📋 Pending Requests", "👥 Manage Users", "🔑 Verification Codes"
        ])

        # ---- Pending Teacher Requests ----
        with admin_tab1:
            st.markdown("### 📋 Pending Teacher Registration Requests")
            pending = [p for p in st.session_state.pending_teacher_store if p.get('status') == 'pending']
            if pending:
                for idx, req in enumerate(pending):
                    st.markdown(f"""
                    <div class="custom-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">📧 {req['fullName']}</span>
                            <span class="badge badge-info">{req['status'].upper()}</span>
                        </div>
                        <div style="font-size: 0.9rem; color: #94a3b8; margin-top: 0.3rem;">
                            Email: <b>{req['email']}</b> | Requested: <b>{req['timestamp']}</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    col_approve, col_reject = st.columns(2)
                    with col_approve:
                        if st.button(f"✅ Approve & Send Code", key=f"approve_{idx}", use_container_width=True):
                            code = generate_verification_code()
                            code_entry = {
                                "code": code,
                                "email": req['email'],
                                "fullName": req['fullName'],
                                "purpose": "Teacher Registration",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                "used": False
                            }
                            st.session_state.verification_codes_store.append(code_entry)
                            save_json_store(VERIFICATION_CODES_JSON, st.session_state.verification_codes_store)
                            req['status'] = 'approved'
                            save_json_store(PENDING_TEACHER_JSON, st.session_state.pending_teacher_store)
                            sent = send_verification_email(req['email'], code, "Teacher Registration")
                            if sent:
                                st.success(f"✅ Invite code **{code}** sent to {req['email']}!")
                            else:
                                st.warning(f"Email failed. Manually share code: **{code}**")
                            st.rerun()
                    with col_reject:
                        if st.button(f"❌ Reject Request", key=f"reject_{idx}", use_container_width=True):
                            req['status'] = 'rejected'
                            save_json_store(PENDING_TEACHER_JSON, st.session_state.pending_teacher_store)
                            st.info("Request rejected.")
                            st.rerun()
            else:
                st.info("No pending teacher registration requests.")

            st.markdown("---")
            st.markdown("### 📜 All Teacher Requests History")
            all_reqs = st.session_state.pending_teacher_store
            if all_reqs:
                for req in all_reqs:
                    status_color = "#34d399" if req['status'] == 'registered' else ("#fbbf24" if req['status'] == 'approved' else ("#f87171" if req['status'] == 'rejected' else "#818cf8"))
                    st.markdown(f"""
                    <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 0.8rem; margin-bottom: 0.5rem;">
                        <span style="color: #f8fafc; font-weight: 600;">{req['fullName']}</span>
                        <span style="color: #94a3b8;"> | {req['email']}</span>
                        <span style="float: right; color: {status_color}; font-weight: 700;">{req['status'].upper()}</span>
                    </div>
                    """, unsafe_allow_html=True)

        # ---- Manage Users ----
        with admin_tab2:
            st.markdown("### 👥 All Registered Users")
            users = st.session_state.users_store
            for uname, udata in list(users.items()):
                role_badge = "badge-healthy" if udata['role'] == 'Teacher' else "badge-info"
                st.markdown(f"""
                <div class="custom-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.05rem; font-weight: 700; color: #f8fafc;">👤 {udata['fullName']}</span>
                        <span class="badge {role_badge}">{udata['role']}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.3rem;">
                        Username: <b>{uname}</b>
                        {f' | Child ID: <b>{udata.get("childId", "")}</b>' if udata.get("childId") else ''}
                        {f' | Email: <b>{udata.get("email", "")}</b>' if udata.get("email") else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if uname != current_user.get("username", ""):
                    if st.button(f"🗑️ Delete User: {uname}", key=f"del_user_{uname}", use_container_width=True):
                        del st.session_state.users_store[uname]
                        save_json_store(USERS_JSON, st.session_state.users_store)
                        st.success(f"User '{uname}' deleted.")
                        st.rerun()

            st.markdown("---")
            st.markdown("### 🔐 Reset Password for Any User")
            with st.form("admin_reset_form"):
                reset_user = st.selectbox("Select User", options=list(users.keys()))
                reset_new_pass = st.text_input("New Password (Min 8 Characters)", type="password", key="admin_reset_pass")
                reset_submit = st.form_submit_button("🔐 Reset Password", use_container_width=True)
            if reset_submit:
                if not reset_new_pass:
                    st.warning("Please enter a new password.")
                elif len(reset_new_pass) < 8:
                    st.error("⚠️ Password must be at least 8 characters long for security.")
                else:
                    st.session_state.users_store[reset_user]["password"] = hash_password(reset_new_pass)
                    save_json_store(USERS_JSON, st.session_state.users_store)
                    st.success(f"✅ Password reset for '{reset_user}'.")

        # ---- Active Verification Codes ----
        with admin_tab3:
            st.markdown("### 🔑 All Verification Codes")
            codes = st.session_state.verification_codes_store
            if codes:
                for vc in reversed(codes):
                    used_badge = "badge-obese" if vc.get("used") else "badge-healthy"
                    used_text = "USED" if vc.get("used") else "ACTIVE"
                    st.markdown(f"""
                    <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 0.8rem; margin-bottom: 0.5rem;">
                        <span style="color: #38bdf8; font-weight: 700; font-size: 1.1rem;">{vc['code']}</span>
                        <span style="color: #94a3b8;"> | {vc.get('purpose', 'N/A')} | {vc.get('email', 'N/A')} | {vc.get('timestamp', '')}</span>
                        <span class="badge {used_badge}" style="float: right;">{used_text}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No verification codes generated yet.")

    # 10. Account Settings (Teacher)
    elif nav_selection == "⚙️ Account Settings":
        st.markdown("<div class='section-header'>⚙️ Account Settings</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="custom-card">
            <div class="card-title">👤 Your Profile</div>
            <div style="color: #e2e8f0; font-size: 1rem; line-height: 2;">
                <b>Username:</b> {current_user.get('username', '')}<br>
                <b>Full Name:</b> {current_user.get('fullName', '')}<br>
                <b>Role:</b> {current_user.get('role', '')}<br>
                <b>Email:</b> {current_user.get('email', 'Not set')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔐 Change Password")
        with st.form("change_pass_form"):
            cp_old = st.text_input("Current Password", type="password", key="cp_old")
            cp_new = st.text_input("New Password (Min 8 Characters)", type="password", key="cp_new")
            cp_confirm = st.text_input("Confirm New Password", type="password", key="cp_confirm")
            cp_submit = st.form_submit_button("🔐 Change Password", use_container_width=True)

        if cp_submit:
            uname = current_user.get("username", "")
            user_data = st.session_state.users_store.get(uname)
            if not user_data or not verify_password(user_data["password"], cp_old):
                st.error("❌ Current password is incorrect.")
            elif cp_new != cp_confirm:
                st.error("❌ New passwords do not match.")
            elif not cp_new:
                st.warning("Please enter a new password.")
            elif len(cp_new) < 8:
                st.error("⚠️ New password must be at least 8 characters long for security.")
            else:
                st.session_state.users_store[uname]["password"] = hash_password(cp_new)
                save_json_store(USERS_JSON, st.session_state.users_store)
                st.success("✅ Password changed successfully!")

        st.markdown("---")
        st.markdown("### 🗑️ Delete My Account")
        st.warning("⚠️ This action is permanent and cannot be undone.")
        with st.form("delete_account_form"):
            da_password = st.text_input("Enter your password to confirm deletion", type="password", key="da_pass")
            da_confirm = st.checkbox("I understand this will permanently delete my account", key="da_check")
            da_submit = st.form_submit_button("🗑️ Delete My Account", use_container_width=True)

        if da_submit:
            uname = current_user.get("username", "")
            user_data = st.session_state.users_store.get(uname)
            if not da_confirm:
                st.error("Please confirm you understand this action is permanent.")
            elif not user_data or not verify_password(user_data["password"], da_password):
                st.error("❌ Incorrect password.")
            else:
                del st.session_state.users_store[uname]
                save_json_store(USERS_JSON, st.session_state.users_store)
                st.session_state.current_user = None
                st.session_state.active_nav = None
                st.success("Account deleted. Redirecting to login...")
                st.rerun()


# ==============================================================================
else:
    parent_cid = str(current_user.get("childId", "C001")).strip()
    child_info = get_child_row(parent_cid)
    child_name = str(child_info['Child Name']).strip() if child_info is not None else "Child"

    # 1. Parent Home / Profile
    if nav_selection == "🏠 My Child Profile":
        st.markdown(f"""
        <div class="hero-banner">
            <div style="display: flex; align-items: center; gap: 1.1rem; margin-bottom: 0.8rem;">
                <div style="background: #ffffff; padding: 6px; border-radius: 16px; display: inline-flex; align-items: center; justify-content: center; box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25); border: 2px solid rgba(255, 255, 255, 0.4);">
                    <img src="data:image/png;base64,{LOGO_B64}" width="50" height="50" style="border-radius: 12px; object-fit: contain;" alt="Growth Advisor Logo">
                </div>
                <div>
                    <div class="hero-title" style="margin-bottom: 0.1rem; line-height: 1.15;">Growth Advisor</div>
                    <div style="color: #a5b4fc; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">AI-POWERED PORTAL — PARENT DASHBOARD</div>
                </div>
            </div>
            <div class="hero-subtitle">
                Welcome, <b>{current_user['fullName']}</b>! Monitor your child <b>{child_name} ({parent_cid})</b>'s growth status, download generated PDF reports, and view attendance percentage.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if child_info is not None:
            st.markdown(f"""
            <div class="custom-card">
                <div class="card-title">📇 Registered Student Details</div>
                <div class="card-meta" style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; font-size: 1rem;">
                    <div>🆔 <b>Roll No / Child ID:</b> {child_info['Child ID']}</div>
                    <div>🧒 <b>Child Full Name:</b> {child_info['Child Name']}</div>
                    <div>👨‍👩‍👧 <b>Parent / Guardian:</b> {child_info['Parent Name']}</div>
                    <div>📍 <b>Location:</b> {child_info['Place']}</div>
                    <div>📞 <b>Contact Number:</b> {child_info['Phone Number']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 14px; padding: 1rem 1.25rem; margin-top: 1.2rem; display: flex; align-items: flex-start; gap: 0.9rem;">
                <span style="font-size: 1.4rem; line-height: 1;">⚠️</span>
                <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;">
                    <b style="color: #f87171; font-size: 0.92rem;">Parent Health & Safety Disclaimer:</b><br>
                    All growth statistics, percentiles, and AI recommendations are provided for informational tracking based on WHO Standards. <b>This application does not provide clinical diagnoses.</b> Please consult your pediatrician or healthcare provider for specific medical advice regarding your child's development.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 2. My Child's Attendance & Stats
    elif nav_selection == "📋 My Child's Attendance & Stats":
        st.markdown(f"<div class='section-header'>📋 Attendance Record & Performance for {child_name} ({parent_cid})</div>", unsafe_allow_html=True)

        p_stats = compute_child_attendance(parent_cid, st.session_state.attendance_store)

        if p_stats["total_days"] > 0:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total School Days", p_stats["total_days"])
            m2.metric("Days Present", p_stats["present_days"])
            m3.metric("Days Absent", p_stats["absent_days"])
            m4.metric("Attendance Score", f"{p_stats['percentage']:.1f}%")

            bar_color = "#10b981" if p_stats['percentage'] >= 90 else ("#f59e0b" if p_stats['percentage'] >= 75 else "#ef4444")
            
            st.markdown(f"""
            <div class="custom-card" style="margin-top: 1rem; border-left: 4px solid {bar_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">📊 Overall Cumulative Attendance: {p_stats['percentage']:.1f}%</span>
                    <span class="badge {p_stats['badge_cls']}">{p_stats['badge_text']}</span>
                </div>
                <div class="att-progress-bg" style="height: 10px; margin-top: 10px;">
                    <div class="att-progress-bar" style="width: {p_stats['percentage']}%; background: {bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if p_stats['percentage'] < 75.0:
                st.warning(f"⚠️ **Attendance Notice**: {child_name}'s cumulative attendance is currently **{p_stats['percentage']:.1f}%**, which is below the recommended 75% threshold. Please ensure regular attendance.")
            else:
                st.success(f"🌟 **Great Attendance Record!** {child_name} has maintained a **{p_stats['percentage']:.1f}%** attendance rate.")

            st.markdown("### 📜 Complete Date-by-Date Attendance Log")
            df_parent_att = pd.DataFrame(p_stats["history"])
            st.dataframe(df_parent_att, use_container_width=True, hide_index=True)
        else:
            st.info(f"No attendance records logged yet for {child_name} ({parent_cid}). Please check back after the class teacher submits the daily attendance sheet.")

    # 3. Parent Reports Download Module
    elif nav_selection == "📄 My Child's Growth Reports":
        st.markdown(f"<div class='section-header'>📄 Downloadable PDF Growth Reports for {child_name} ({parent_cid})</div>", unsafe_allow_html=True)

        parent_reports = []
        p_cid_u = parent_cid.upper()
        p_cname_u = child_name.upper()

        for r in st.session_state.reports_store:
            r_cid = str(r.get('childId', '')).strip().upper()
            r_cname = str(r.get('childName', '')).strip().upper()
            r_title = str(r.get('title', '')).strip().upper()

            if (p_cid_u and r_cid == p_cid_u) or (p_cid_u and p_cid_u in r_title) or (p_cname_u and p_cname_u in r_cname) or (p_cname_u and p_cname_u in r_title):
                parent_reports.append(r)

        if parent_reports:
            st.markdown(f"**Found {len(parent_reports)} Generated PDF Growth Reports for {child_name}**")
            for idx, rep in enumerate(parent_reports):
                b_cls = get_badge_class(rep.get('aiStatus', 'Healthy'))
                st.markdown(f"""
                <div class="custom-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">📄 {rep['title']}</span>
                        <span class="badge {b_cls}">{rep.get('aiStatus', 'Healthy')}</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 0.6rem;">
                        🕒 <b>Generated On:</b> {rep['timestamp']}<br>
                        📊 <b>Height Percentile:</b> P{rep.get('hfa_p', 50):.1f} | <b>Weight/Height Percentile:</b> P{rep.get('wfh_p', 50):.1f} | <b>BMI:</b> {rep.get('bmi', 15.0):.1f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                pdf_bytes = get_pdf_bytes_for_report(rep)
                st.download_button(
                    label=f"📄 Download PDF Growth Report for {child_name}",
                    data=pdf_bytes,
                    file_name=f"{parent_cid}_{child_name}_Growth_Report.pdf",
                    mime="application/pdf",
                    key=f"dl_p_{rep.get('reportId', idx)}",
                    use_container_width=True
                )
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info(f"No PDF growth reports generated yet for {child_name} ({parent_cid}). You can generate one right now using the button below!")

    # 4. Parent Generator Form
    elif nav_selection == "⚡ Generate Report for My Child":
        st.markdown(f"<div class='section-header'>⚡ Generate Growth Report for {child_name} ({parent_cid})</div>", unsafe_allow_html=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            p_sex = st.radio("Sex", options=["Male", "Female"], horizontal=True)
            p_sex_code = "M" if p_sex == "Male" else "F"
            p_age_years = st.number_input("Age (in Years)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
            p_age_months = int(p_age_years * 12)
        with col_p2:
            p_height = st.number_input("Height (cm)", min_value=40.0, max_value=130.0, value=85.0, step=0.1)
            p_weight = st.number_input("Weight (kg)", min_value=1.0, max_value=40.0, value=12.0, step=0.1)

        p_gen_btn = st.button(f"⚡ Generate & Save Report for {child_name}", use_container_width=True)

        if p_gen_btn:
            with st.spinner("Analyzing measurements & creating downloadable PDF report..."):
                report_calc = generate_report(p_age_months, p_height, p_weight, p_sex_code, growth_model, model_scaler)
                pdf_buf = create_pdf_report(child_name, p_age_months, report_calc)
                pdf_bytes = pdf_buf.getvalue()

                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_filename = f"{parent_cid}_{child_name}_Report_{timestamp_str}.pdf"
                filepath = os.path.join(TEMP_REPORTS_DIR, pdf_filename)
                with open(filepath, "wb") as f:
                    f.write(pdf_bytes)

                new_rep_record = {
                    "reportId": f"R{len(st.session_state.reports_store)+1:03d}",
                    "childId": parent_cid,
                    "childName": child_name,
                    "title": f"{child_name} Growth Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "aiStatus": report_calc["ai_status"],
                    "hfa_p": report_calc["hfa_p"],
                    "wfh_p": report_calc["wfh_p"],
                    "bmi": report_calc["bmi"],
                    "pdf_filename": pdf_filename
                }
                st.session_state.reports_store.insert(0, new_rep_record)
                save_json_store(REPORTS_JSON, st.session_state.reports_store)

            st.success(f"✅ Growth Report generated for {child_name}! Status: {report_calc['ai_status']}")
            st.download_button(
                label=f"📄 Download Generated PDF Report for {child_name}",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                use_container_width=True
            )

    # 5. My Child's AI Meal Planner (Parent)
    elif nav_selection == "🍽️ My Child's AI Meal Planner":
        st.markdown(f"<div class='section-header'>🍽️ AI Daily Meal Planner for {child_name} ({parent_cid})</div>", unsafe_allow_html=True)
        st.caption("Personalized full-day nutrition plan based on AskNestle, ICMR-NIN RDA 2024, WHO & Harvard Healthy Eating Plate guidelines.")

        p_rep = next((r for r in st.session_state.reports_store if str(r.get('childId', '')).upper() == parent_cid.upper()), None)
        p_default_status = p_rep.get("aiStatus", "Healthy") if p_rep else "Healthy"

        col_pm1, col_pm2 = st.columns(2)
        with col_pm1:
            p_mp_sex = st.radio("Sex", ["Male", "Female"], horizontal=True, key="p_mp_sex_in")
            p_mp_age_y = st.slider("Age (Years)", min_value=1.0, max_value=5.0, value=2.5, step=0.1, key="p_mp_age_in")
            p_mp_age_m = float(p_mp_age_y * 12)
        with col_pm2:
            status_opts = ["Healthy", "Underweight", "Stunted", "Stunted & Underweight", "Overweight", "Obese"]
            st_idx = status_opts.index(p_default_status) if p_default_status in status_opts else 0
            p_mp_status = st.selectbox("Growth Status Focus", status_opts, index=st_idx, key="p_mp_stat_in")
            p_mp_diet = st.selectbox("Family Dietary Preference", ["Vegetarian", "Eggetarian", "Non-Vegetarian"], key="p_mp_diet_in")
            p_mp_act = st.select_slider("Activity Level", options=["Sedentary", "Moderate", "Active"], value="Moderate", key="p_mp_act_in")

        p_meal_plan = generate_custom_meal_plan(p_mp_age_m, p_mp_sex, p_mp_diet, p_mp_status, p_mp_act)

        st.markdown("---")
        st.markdown(f"### 📊 Daily Nutritional Balance for {child_name}")
        
        p_tgt = p_meal_plan["target_rda"]
        p_tot = p_meal_plan["plan_totals"]
        
        pm_c1, pm_c2, pm_c3, pm_c4, pm_c5, pm_c6 = st.columns(6)
        pm_c1.metric("Calories", f"{p_tot['calories']} kcal", f"Goal: {p_tgt['calories']}")
        pm_c2.metric("Protein", f"{p_tot['protein_g']} g", f"RDA: {p_tgt['protein_g']}g")
        pm_c3.metric("Healthy Fats", f"{p_tot['fat_g']} g", f"RDA: {p_tgt['fat_g']}g")
        pm_c4.metric("Carbs", f"{p_tot['carbs_g']} g", f"RDA: {p_tgt['carbs_g']}g")
        pm_c5.metric("Calcium", f"{p_tgt['calcium_mg']} mg", "ICMR-NIN")
        pm_c6.metric("Water", f"{p_tgt['water_ml']} ml", "Hydration")

        st.markdown("---")
        st.markdown(f"### 🕒 {child_name}'s 5-Meal Daily Schedule (AskNestle & NIOS)")
        
        p_meal_icons = {
            "Breakfast": "🌅",
            "Mid-Morning": "🍎",
            "Lunch": "🍛",
            "Evening Snack": "🥛",
            "Dinner": "🍲"
        }
        
        for m_type, m_val in p_meal_plan["meals"].items():
            icon = p_meal_icons.get(m_type, "🍽️")
            st.markdown(f"""
            <div class="custom-card" style="margin-bottom: 0.9rem; border-left: 4px solid #38bdf8;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">{icon} {m_type}: {m_val['name']}</span>
                    <span class="badge badge-info">~{m_val['cal']} kcal</span>
                </div>
                <div style="color: #94a3b8; font-size: 0.92rem; margin-top: 0.4rem;">
                    {m_val['desc']}
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 1.2rem; margin-top: 0.6rem; font-size: 0.85rem; color: #cbd5e1;">
                    <span>✋ <b>Serving Portion:</b> {m_val['hand_portion']}</span>
                    <span>✨ <b>Key Nutrients:</b> {m_val['nutrients']}</span>
                    <span>🥩 <b>Protein:</b> {m_val['p']}g | <b>Carbs:</b> {m_val['c']}g | <b>Fats:</b> {m_val['f']}g</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🖐️ Hand-Size Portion Guide for Parents")
        st.caption("A simple way to portion food without measuring cups:")
        
        hpc1, hpc2, hpc3, hpc4 = st.columns(4)
        hp_map_p = [
            (hpc1, "Palm", "Protein", "Dal, Paneer, Eggs, Chicken, Fish", "✋", p_meal_plan["hand_portions"]["Palm"]["serving_1_3"] if p_mp_age_m < 36 else p_meal_plan["hand_portions"]["Palm"]["serving_3_5"]),
            (hpc2, "Fist", "Veggies & Fruits", "Spinach, Carrots, Apple, Papaya", "✊", p_meal_plan["hand_portions"]["Fist"]["serving_1_3"] if p_mp_age_m < 36 else p_meal_plan["hand_portions"]["Fist"]["serving_3_5"]),
            (hpc3, "Cupped Hand", "Grains & Carbs", "Roti, Rice, Poha, Dalia, Oats", "🤲", p_meal_plan["hand_portions"]["Cupped Hand"]["serving_1_3"] if p_mp_age_m < 36 else p_meal_plan["hand_portions"]["Cupped Hand"]["serving_3_5"]),
            (hpc4, "Thumb", "Healthy Fats", "Desi Ghee, Butter, Nut Powders", "👍", p_meal_plan["hand_portions"]["Thumb"]["serving_1_3"] if p_mp_age_m < 36 else p_meal_plan["hand_portions"]["Thumb"]["serving_3_5"]),
        ]
        for col, hname, hnut, hexamples, hicon, hserv in hp_map_p:
            with col:
                st.markdown(f"""
                <div class="custom-card" style="text-align: center; padding: 1rem 0.8rem;">
                    <div style="font-size: 2rem;">{hicon}</div>
                    <div style="font-weight: 700; color: #f8fafc; margin-top: 0.3rem;">{hname} = {hnut}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem;">{hexamples}</div>
                    <div style="font-size: 0.82rem; color: #38bdf8; font-weight: 600; margin-top: 0.5rem; background: rgba(56, 189, 248, 0.1); padding: 4px; border-radius: 6px;">{hserv}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"### 💡 Pediatric Nutritional Guidance ({p_meal_plan['clinical_advice']['title']})")
        st.info(f"**Focus:** {p_meal_plan['clinical_advice']['focus']}")
        for tip in p_meal_plan['clinical_advice']['tips']:
            st.markdown(f"• {tip}")

        st.markdown("<br>", unsafe_allow_html=True)
        p_pdf_meal = create_meal_plan_pdf(child_name, p_meal_plan)
        st.download_button(
            label=f"📥 Download Printable Daily Meal Plan PDF for {child_name}",
            data=p_pdf_meal.getvalue(),
            file_name=f"{parent_cid}_{child_name}_Daily_Meal_Plan.pdf",
            mime="application/pdf",
            key="p_dl_meal_plan_btn",
            use_container_width=True
        )

    # 6. Chat with Teacher (Parent View)
    elif nav_selection == "💬 Chat with Teacher":
        st.markdown(f"<div class='section-header'>💬 Chat with Class Teacher ({child_name} - {parent_cid})</div>", unsafe_allow_html=True)

        with st.form("parent_msg_form"):
            p_message = st.text_area("Write message to Class Teacher", placeholder="Ask about health, diet, or progress...")
            p_submit = st.form_submit_button("✉️ Send Message to Teacher")

        if p_submit:
            if p_message.strip():
                new_msg = {
                    "childId": parent_cid,
                    "childName": child_name,
                    "senderRole": "Parent",
                    "senderName": current_user['fullName'],
                    "message": p_message,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                st.session_state.messages_store.append(new_msg)
                save_json_store(MESSAGES_JSON, st.session_state.messages_store)
                st.success("✅ Message sent to Teacher!")
                st.rerun()

        st.markdown("---")
        st.markdown("### 💬 Chat Stream with Teacher")
        child_msgs = [m for m in st.session_state.messages_store if m.get('childId') == parent_cid]
        if child_msgs:
            for m in child_msgs:
                s_role = m.get('senderRole', 'Parent')
                bubble_cls = "chat-bubble-teacher" if s_role == "Teacher" else "chat-bubble-parent"
                color_hdr = "#818cf8" if s_role == "Teacher" else "#34d399"
                escaped_msg = html.escape(str(m['message']))
                escaped_name = html.escape(str(m.get('senderName', s_role)))
                st.markdown(f"""
                <div class="{bubble_cls}">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: {color_hdr};">
                        <span><b>{s_role}:</b> {escaped_name}</span>
                        <span>🕒 {m['timestamp']}</span>
                    </div>
                    <div style="font-size: 1rem; color: #f8fafc; margin-top: 0.4rem; white-space: pre-wrap;">
                        {escaped_msg}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No messages exchanged yet with the class teacher. Send a note above!")

    # 6. Parent Account Settings
    elif nav_selection == "⚙️ Account Settings":
        st.markdown("<div class='section-header'>⚙️ Account Settings (Parent)</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="custom-card">
            <div class="card-title">👤 Your Profile</div>
            <div style="color: #e2e8f0; font-size: 1rem; line-height: 2;">
                <b>Username:</b> {current_user.get('username', '')}<br>
                <b>Full Name:</b> {current_user.get('fullName', '')}<br>
                <b>Role:</b> {current_user.get('role', '')}<br>
                <b>Linked Child:</b> {child_name} ({parent_cid})<br>
                <b>Email:</b> {current_user.get('email', 'Not set')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔐 Change Password")
        with st.form("parent_change_pass_form"):
            cp_old = st.text_input("Current Password", type="password", key="p_cp_old")
            cp_new = st.text_input("New Password (Min 8 Characters)", type="password", key="p_cp_new")
            cp_confirm = st.text_input("Confirm New Password", type="password", key="p_cp_confirm")
            cp_submit = st.form_submit_button("🔐 Change Password", use_container_width=True)

        if cp_submit:
            uname = current_user.get("username", "")
            user_data = st.session_state.users_store.get(uname)
            if not user_data or not verify_password(user_data["password"], cp_old):
                st.error("❌ Current password is incorrect.")
            elif cp_new != cp_confirm:
                st.error("❌ New passwords do not match.")
            elif not cp_new:
                st.warning("Please enter a new password.")
            elif len(cp_new) < 8:
                st.error("⚠️ New password must be at least 8 characters long for security.")
            else:
                st.session_state.users_store[uname]["password"] = hash_password(cp_new)
                save_json_store(USERS_JSON, st.session_state.users_store)
                st.success("✅ Password changed successfully!")

        st.markdown("---")
        st.markdown("### 🗑️ Delete My Account")
        st.warning("⚠️ This action is permanent and cannot be undone.")
        with st.form("parent_delete_account_form"):
            da_password = st.text_input("Enter your password to confirm deletion", type="password", key="p_da_pass")
            da_confirm = st.checkbox("I understand this will permanently delete my account", key="p_da_check")
            da_submit = st.form_submit_button("🗑️ Delete My Account", use_container_width=True)

        if da_submit:
            uname = current_user.get("username", "")
            user_data = st.session_state.users_store.get(uname)
            if not da_confirm:
                st.error("Please confirm you understand this action is permanent.")
            elif not user_data or not verify_password(user_data["password"], da_password):
                st.error("❌ Incorrect password.")
            else:
                del st.session_state.users_store[uname]
                save_json_store(USERS_JSON, st.session_state.users_store)
                st.session_state.current_user = None
                st.session_state.active_nav = None
                st.success("Account deleted. Redirecting to login...")
                st.rerun()

# ==============================================================================
# COMMON VIEWS (FOOD RECOMMENDATIONS & MEAL SCHEDULE)
# ==============================================================================
if nav_selection == "🥗 Food Recommendations":
    st.markdown("<div class='section-header'>🥗 Child Nutrition & Food Recommendations</div>", unsafe_allow_html=True)
    if not food_df.empty:
        category_options = list(food_df['Category'].unique())
        selected_category = st.radio("Select Guidance Type", options=category_options, horizontal=True)
        cat_df = food_df[food_df['Category'] == selected_category]
        status_options = ["All"] + list(cat_df['Status'].unique())
        selected_status = st.selectbox(f"Filter by {selected_category} Target:", options=status_options)

        if selected_status != "All":
            cat_df = cat_df[cat_df['Status'] == selected_status]

        for _, row in cat_df.iterrows():
            status_name = str(row['Status'])
            badge_cls = get_badge_class(status_name)

            st.markdown(f"""
            <div class="custom-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <span class="card-heading" style="font-size: 1.4rem; font-weight: 800;">{status_name}</span>
                    <span class="badge {badge_cls}">{row['Category']}</span>
                </div>
                <div class="card-subtext" style="background: rgba(192, 88, 0, 0.08); padding: 0.8rem 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    🎯 <b>Nutritional Goal:</b> {row.get('Goal', 'N/A')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**🍽️ Recommended Foods:**\n\n{row.get('Recommended Foods', 'N/A')}")
                st.markdown(f"**🥄 How to Eat & Meal Pattern:**\n\n{row.get('Meal Pattern', 'N/A')}")
                st.markdown(f"**🥗 Snack Ideas:**\n\n{row.get('Snack Ideas', 'N/A')}")

            with col2:
                st.markdown(f"**🚫 Foods to Avoid:**\n\n{row.get('Avoid', 'N/A')}")
                st.markdown(f"**💡 Activity & Health Tips:**\n\n{row.get('Activity / Tip', 'N/A')}")
                st.markdown(f"**⚡ Key Nutrients:**\n\n{row.get('Key Nutrients', 'N/A')}")

            st.markdown("---")

elif nav_selection == "📅 Weekly Meal Schedule":
    st.markdown("<div class='section-header'>📅 Weekly Child Meal Schedule</div>", unsafe_allow_html=True)
    if not meal_df.empty:
        st.dataframe(meal_df[['Day', 'Meal', 'Calories (kcal)', 'Protein (g)', 'Carbs (g)', 'Fat (g)', 'Notes']], use_container_width=True, hide_index=True)
        st.markdown("### 🍱 Daily Meal Breakdown Cards")
        for _, row in meal_df.iterrows():
            st.markdown(f"""
            <div class="custom-card">
                <div class="card-heading" style="font-size: 1.2rem; font-weight: 700; color: #C05800 !important;">📅 {row['Day']}</div>
                <div class="card-heading" style="font-size: 1.1rem; font-weight: 600; margin-top: 0.3rem;">{row['Meal']}</div>
                <div style="margin-top: 0.8rem; display: flex; gap: 1rem; flex-wrap: wrap;">
                    <span class="badge badge-info">🔥 Calories: {row.get('Calories (kcal)', 'N/A')}</span>
                    <span class="badge badge-healthy">🥩 Protein: {row.get('Protein (g)', 'N/A')}</span>
                    <span class="badge badge-overweight">🌾 Carbs: {row.get('Carbs (g)', 'N/A')}</span>
                    <span class="badge badge-underweight">🥑 Fat: {row.get('Fat (g)', 'N/A')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

elif nav_selection == "ℹ️ About the Project & Team":
    render_about_page()

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 0.85rem; color: #64748b; padding: 1rem 0 0.5rem; line-height: 1.6;">
    <b>AI Child Growth Advisor</b> • Powered by Streamlit, WHO Growth Standards & PyTorch<br>
    <span style="font-size: 0.78rem; color: #94a3b8;">⚠️ <i>Educational and screening support tool. Not a substitute for professional pediatric clinical evaluation.</i></span>
</div>
""", unsafe_allow_html=True)
