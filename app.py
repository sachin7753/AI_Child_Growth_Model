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
WFH_BOYS_FILE = "tab_wfh_boys_p_0_5.xlsx"
WFH_GIRLS_FILE = "tab_wfh_girls_p_0_5.xlsx"
MODEL_PATH = "growth_model.pth"
SCALER_PATH = "scaler.joblib"
PARAMS_PATH = "best_params.json"
DAYS_PER_MONTH = 30.4375
CLASS_LABELS = {0: "Underweight", 1: "Healthy", 2: "Overweight", 3: "Obese", 4: "Stunted", 5: "Normal Ht"}

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

# ------------------- CUSTOM CSS STYLING -------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%);
        color: white;
        padding: 2.2rem 2rem;
        border-radius: 18px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #e0e7ff;
        max-width: 750px;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
    }

    .custom-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .chat-bubble-teacher {
        background: #1e1b4b;
        border: 1px solid #4338ca;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .chat-bubble-parent {
        background: #022c22;
        border: 1px solid #059669;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-underweight { background-color: #fef3c7; color: #92400e; }
    .badge-healthy { background-color: #d1fae5; color: #065f46; }
    .badge-overweight { background-color: #ffedd5; color: #9a3412; }
    .badge-obese { background-color: #fee2e2; color: #991b1b; }
    .badge-stunted { background-color: #f3e8ff; color: #6b21a8; }
    .badge-info { background-color: #e0f2fe; color: #075985; }

    .section-header {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
        border-left: 4px solid #6366f1;
        padding-left: 0.8rem;
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---- Sidebar Navigation Styling ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        padding: 0.65rem 1rem;
        border-radius: 12px;
        border: 1px solid transparent;
        background: transparent;
        color: #cbd5e1;
        font-size: 0.92rem;
        font-weight: 600;
        font-family: 'Plus Jakarta Sans', sans-serif;
        transition: all 0.25s cubic-bezier(.4,0,.2,1);
        margin-bottom: 2px;
        cursor: pointer;
        letter-spacing: 0.01em;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.3);
        color: #e0e7ff;
        transform: translateX(4px);
    }
    section[data-testid="stSidebar"] .stButton > button:active {
        transform: translateX(4px) scale(0.98);
    }
    section[data-testid="stSidebar"] .stButton > button:focus:not(:focus-visible) {
        box-shadow: none;
    }
    /* Active nav item styling — applied via key matching */
    .nav-active > button {
        background: linear-gradient(135deg, #4338ca 0%, #6366f1 100%) !important;
        border-color: #818cf8 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
        font-weight: 700 !important;
    }
    .nav-active > button:hover {
        transform: translateX(0px) !important;
    }
    /* Sidebar nav label */
    .sidebar-nav-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #475569;
        padding: 0.6rem 0.4rem 0.35rem;
        margin-top: 0.2rem;
    }
    /* Logout button override */
    .logout-btn > button {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        color: #f87171 !important;
    }
    .logout-btn > button:hover {
        background: rgba(239, 68, 68, 0.25) !important;
        border-color: #ef4444 !important;
        color: #fecaca !important;
        transform: translateX(0) !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- AI MODEL & WHO BACKEND ENGINE -------------------
class GrowthNet(nn.Module):
    def __init__(self, n_layers=2, n_units=64, dropout_rate=0.3):
        super().__init__()
        layers = []
        in_features = 4
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, n_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = n_units
        layers.append(nn.Linear(in_features, len(CLASS_LABELS)))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model_and_scaler(model_path=MODEL_PATH, scaler_path=SCALER_PATH, params_path=PARAMS_PATH):
    try:
        if os.path.exists(params_path) and os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(params_path, 'r') as f:
                best_params = json.load(f)
            model = GrowthNet(n_layers=best_params['n_layers'], n_units=best_params['n_units'], dropout_rate=best_params['dropout_rate'])
            model.load_state_dict(torch.load(model_path))
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
        return {float(re.findall(r"\d+", c)[0]): row0[c] + frac * (row1[c] - row0[c]) for c in pcols}
    return {float(re.findall(r"\d+", c)[0]): float(row[c]) for c in pcols}

def est_percentile(value, curve):
    pts = sorted(curve.items(), key=lambda item: item[1])
    values = [v for p, v in pts]
    percs = [p for p, v in pts]
    if value <= values[0]:
        return percs[0]
    if value >= values[-1]:
        return percs[-1]
    j = np.searchsorted(values, value, side="right")
    v0, v1, p0, p1 = values[j - 1], values[j], percs[j - 1], percs[j]
    return p0 + (value - v0) / (v1 - v0) * (p1 - p0)

def ai_predict(model, scaler, age_m, ht, wt, sex, wfh_p, hfa_p):
    bmi = wt / ((ht / 100) ** 2)
    confidence_score = 0.92
    status = "Healthy"

    if model is not None and scaler is not None:
        try:
            input_data = np.array([[age_m, ht, wt, 1 if sex == "M" else 0]])
            input_scaled = scaler.transform(input_data)
            x = torch.tensor(input_scaled, dtype=torch.float32)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx_tensor = torch.max(probs, dim=1)
                pred_idx = int(pred_idx_tensor.item())
                confidence_score = confidence.item()
                status = CLASS_LABELS.get(pred_idx, "Healthy")
        except Exception:
            pass

    if wfh_p < 3:
        status = "Underweight"
    elif wfh_p > 85:
        status = "Obese" if bmi >= 30 else "Overweight"
    elif bmi >= 30:
        status = "Obese"
    elif bmi >= 25:
        status = "Overweight"
    elif hfa_p < 3 and status in ["Healthy", "Normal Ht"]:
        status = "Stunted"
    elif status == "Underweight" and wfh_p >= 5 and hfa_p < 5:
        status = "Stunted"

    return status, confidence_score

def build_age_meal_ideas(age_m):
    if age_m < 6:
        return ["Exclusive breastmilk/formula feeding as advised by pediatrician.", "Feed on demand 8-12 times/day."]
    if age_m < 9:
        return ["Start with 2-3 tbsp thick mashed foods twice daily.", "Add one new food every 3 days."]
    if age_m < 12:
        return ["3 soft meals + 1 snack; keep texture soft and mashed.", "Offer iron-rich foods daily (dal, egg yolk, cereal)."]
    if age_m < 24:
        return ["3 family meals + 2 snacks with child-sized portions.", "Include protein at each meal (egg, paneer, dal, fish/chicken)."]
    return ["3 meals + 2 healthy snacks following fixed meal schedule.", "Balanced plate: 1/2 vegetables-fruits, 1/4 protein, 1/4 grains."]

def get_ai_recommendations(status, age_m, wfh_p, hfa_p, bmi):
    recs = {
        "summary": f"Status: {status} (BMI: {bmi:.1f} | Wt-for-Ht: P{wfh_p:.1f})",
        "what_to_eat": [],
        "how_to_eat": [],
        "meal_ideas": build_age_meal_ideas(age_m),
    }

    if status in ["Obese", "Overweight"]:
        recs["what_to_eat"] = ["Vegetables, fruits, lean proteins, whole grains, and plain curd.", "High-fiber snacks: roasted chana, fruit slices, sprouts."]
        recs["how_to_eat"] = ["Serve fixed portions using a small plate.", "Avoid screen-time eating.", "60 mins active play daily."]
    elif status == "Underweight":
        recs["what_to_eat"] = ["Nutritious calorie-dense foods: eggs, paneer, nut powders, banana, sweet potato.", "Healthy fats: ghee, peanut butter."]
        recs["how_to_eat"] = ["Offer 5-6 small meals/snacks daily.", "Add one calorie booster per meal.", "Track weight monthly."]
    elif status == "Stunted":
        recs["what_to_eat"] = ["Protein-rich foods: dal, egg, fish/chicken, paneer, soybean.", "Leafy vegetables, ragi, fruits."]
        recs["how_to_eat"] = ["Include protein in breakfast, lunch, and dinner.", "Pair iron foods with vitamin C foods."]
    else:
        recs["what_to_eat"] = ["Balanced plate with grains, protein, vegetables, fruits, and dairy.", "Rotate foods weekly."]
        recs["how_to_eat"] = ["3 meals + 2 planned snacks at consistent times.", "60+ mins active play daily."]

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
    hfa_ref, hfa_pcols = load_ref(HFA_BOYS_FILE if sex == "M" else HFA_GIRLS_FILE, r"age|day|month")
    wfh_ref, wfh_pcols = load_ref(WFH_BOYS_FILE if sex == "M" else WFH_GIRLS_FILE, r"height|length")
    
    age_d = age_m * DAYS_PER_MONTH
    if hfa_ref is not None:
        table_val = age_d if float(hfa_ref.iloc[:, 0].max()) > 120 else age_m
        hfa_curve = interp_curve(hfa_ref, hfa_pcols, table_val)
        hfa_p = est_percentile(ht, hfa_curve)
    else:
        hfa_curve = {50: ht}
        hfa_p = 50.0

    if wfh_ref is not None:
        wfh_curve = interp_curve(wfh_ref, wfh_pcols, ht)
        wfh_p = est_percentile(wt, wfh_curve)
    else:
        wfh_curve = {50: wt}
        wfh_p = 50.0

    ai_status, confidence = ai_predict(model, scaler, age_m, ht, wt, sex, wfh_p, hfa_p)
    bmi = wt / ((ht / 100) ** 2)

    who_msgs = []
    if wfh_p < 3:
        who_msgs.append((f"Wasting risk (P{wfh_p:.1f})", colors.red))
    elif wfh_p > 85:
        who_msgs.append((f"Overweight risk (P{wfh_p:.1f})", colors.red))
    else:
        who_msgs.append(("Wt-for-height healthy.", colors.green))

    if hfa_p < 3:
        who_msgs.append((f"Stunting risk (P{hfa_p:.1f})", colors.red))
    else:
        who_msgs.append(("Ht-for-age healthy.", colors.green))

    rec_plan = get_ai_recommendations(ai_status, age_m, wfh_p, hfa_p, bmi)
    recommendations = flatten_recommendation_plan(rec_plan)

    return {
        "wfh_p": wfh_p,
        "hfa_p": hfa_p,
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
    st.markdown(f"""
    <div class="hero-banner">
        <div style="display: flex; align-items: center; gap: 1.1rem; margin-bottom: 0.9rem;">
            <div style="background: #ffffff; padding: 6px; border-radius: 16px; display: inline-flex; align-items: center; justify-content: center; box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25); border: 2px solid rgba(255, 255, 255, 0.4);">
                <img src="data:image/png;base64,{LOGO_B64}" width="54" height="54" style="border-radius: 12px; object-fit: contain;" alt="Growth Advisor Logo">
            </div>
            <div>
                <div class="hero-title" style="margin-bottom: 0.1rem; line-height: 1.15;">Growth Advisor</div>
                <div style="color: #a5b4fc; font-size: 0.85rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">AI-POWERED PORTAL</div>
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


# ------------------- LOGGED IN SESSION & SIDEBAR -------------------
current_user = st.session_state.get("current_user")
if not current_user:
    st.stop()

user_role = current_user.get("role", "")

st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 0.85rem; padding: 0.6rem 0 0.4rem;">
    <div style="background: #ffffff; padding: 4px; border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 14px rgba(0,0,0,0.3);">
        <img src="data:image/png;base64,{LOGO_B64}" width="38" height="38" style="border-radius: 8px; object-fit: contain;" alt="Growth Advisor Logo">
    </div>
    <div>
        <div style="font-size: 1.32rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em; line-height: 1.15;">Growth Advisor</div>
        <div style="font-size: 0.72rem; color: #818cf8; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">AI-POWERED PORTAL</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(56,189,248,0.1)); border: 1px solid rgba(99,102,241,0.25); padding: 0.75rem 0.9rem; border-radius: 12px; margin: 0.8rem 0;">
    <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">👤 {current_user['fullName']}</div>
    <div style="font-size: 0.8rem; color: #818cf8; margin-top: 0.15rem;">Role: <b>{user_role}</b></div>
    {f'<div style="font-size: 0.8rem; color: #34d399; margin-top: 0.1rem;">Roll No: <b>{current_user["childId"]}</b></div>' if user_role == 'Parent' else ''}
</div>
""", unsafe_allow_html=True)

# ---- Navigation menu with styled buttons ----
if "active_nav" not in st.session_state:
    st.session_state.active_nav = None

if user_role == "Teacher":
    nav_options = [
        ("🏠 Home / Overview",        "home"),
        ("👶 Children Directory",     "children"),
        ("📋 Attendance Sheet",       "attendance"),
        ("⚡ Generate Report",        "generate"),
        ("📄 Reports Repository",     "reports"),
        ("🥗 Food Recommendations",   "food"),
        ("📅 Meal Schedule",          "meal"),
        ("💬 Messages",               "messages"),
        ("🛡️ Admin Panel",            "admin_panel"),
        ("⚙️ Account Settings",       "settings"),
    ]
    nav_label_map = {
        "home":       "🏠 Home / Overview",
        "children":   "👶 Children Directory",
        "attendance": "📋 Teacher Attendance Sheet",
        "generate":   "⚡ Generate Growth Report",
        "reports":    "📄 Student Reports Repository",
        "food":       "🥗 Food Recommendations",
        "meal":       "📅 Weekly Meal Schedule",
        "messages":   "💬 Parent-Teacher Messages",
        "admin_panel": "🛡️ Admin Panel",
        "settings":    "⚙️ Account Settings",
    }
else:
    nav_options = [
        ("🏠 My Child Profile",       "home"),
        ("📋 Attendance & Stats",     "attendance"),
        ("📄 Growth Reports",         "reports"),
        ("⚡ Generate Report",        "generate"),
        ("🥗 Food Recommendations",   "food"),
        ("📅 Meal Schedule",          "meal"),
        ("💬 Chat with Teacher",      "messages"),
        ("⚙️ Account Settings",       "settings"),
    ]
    nav_label_map = {
        "home":       "🏠 My Child Profile",
        "attendance": "📋 My Child's Attendance & Stats",
        "reports":    "📄 My Child's Growth Reports",
        "generate":   "⚡ Generate Report for My Child",
        "food":       "🥗 Food Recommendations",
        "meal":       "📅 Weekly Meal Schedule",
        "messages":   "💬 Chat with Teacher",
        "settings":   "⚙️ Account Settings",
    }

default_key = nav_options[0][1]
if st.session_state.active_nav is None or st.session_state.active_nav not in nav_label_map:
    st.session_state.active_nav = default_key

st.sidebar.markdown('<div class="sidebar-nav-label">Navigation</div>', unsafe_allow_html=True)

for label, key in nav_options:
    is_active = (st.session_state.active_nav == key)
    container = st.sidebar.container()
    if is_active:
        container.markdown('<div class="nav-active">', unsafe_allow_html=True)
    if container.button(label, key=f"nav_{key}", use_container_width=True):
        st.session_state.active_nav = key
        st.rerun()
    if is_active:
        container.markdown('</div>', unsafe_allow_html=True)

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
                        <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">{row['Child Name']}</span>
                        <span class="badge badge-info">{row['Child ID']}</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #94a3b8; margin-top: 0.5rem;">
                        👨‍👩‍👧 Parent: <b>{row['Parent Name']}</b><br>
                        📍 Location: <b>{row['Place']}</b><br>
                        📞 Phone: <b>{row['Phone Number']}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # 3. Teacher Attendance Sheet
    elif nav_selection == "📋 Teacher Attendance Sheet":
        st.markdown("<div class='section-header'>📋 Daily Student Attendance Sheet</div>", unsafe_allow_html=True)
        selected_date = st.date_input("Select Date", datetime.now()).strftime("%Y-%m-%d")
        date_records = st.session_state.attendance_store.get(selected_date, {})

        with st.form("teacher_att_form"):
            st.subheader(f"Mark Attendance for {selected_date}")
            attendance_inputs = {}
            cols = st.columns(2)
            for idx, row in students_df.iterrows():
                cid = str(row['Child ID'])
                cname = str(row['Child Name'])
                prev_status = date_records.get(cid, "Present")
                c = cols[idx % 2]
                status = c.radio(
                    f"{cname} ({cid})",
                    options=["Present", "Absent"],
                    index=0 if prev_status == "Present" else 1,
                    key=f"att_{selected_date}_{cid}",
                    horizontal=True
                )
                attendance_inputs[cid] = status

            submit_att = st.form_submit_button("💾 Save Attendance Records", use_container_width=True)

        if submit_att:
            st.session_state.attendance_store[selected_date] = attendance_inputs
            save_json_store(ATTENDANCE_JSON, st.session_state.attendance_store)
            st.success(f"✅ Attendance saved successfully for {selected_date}!")
            st.rerun()

        if selected_date in st.session_state.attendance_store:
            records = st.session_state.attendance_store[selected_date]
            total = len(records)
            present = sum(1 for v in records.values() if v == "Present")
            absent = total - present
            pct = (present / total * 100) if total > 0 else 0

            st.markdown("### 📊 Attendance Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Students", total)
            m2.metric("Present", present)
            m3.metric("Absent", absent)
            m4.metric("Attendance Rate", f"{pct:.1f}%")

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
                    <span style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">📄 {rep['title']}</span>
                    <span class="badge {b_cls}">{rep.get('aiStatus', 'Healthy')}</span>
                </div>
                <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 0.6rem; display: flex; gap: 1.5rem;">
                    <span>🆔 Roll No: <b>{rep['childId']}</b></span>
                    <span>🧒 Child Name: <b>{rep['childName']}</b></span>
                    <span>🕒 Generated: <b>{rep['timestamp']}</b></span>
                </div>
                <div style="margin-top: 0.6rem; font-size: 0.85rem; color: #94a3b8;">
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

    # 6. Teacher Messages Portal
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
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; color: #e2e8f0; font-size: 1rem;">
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
        st.markdown(f"<div class='section-header'>📋 Attendance Record & Percentage for {child_name} ({parent_cid})</div>", unsafe_allow_html=True)

        child_att_log = []
        for date_str, recs in st.session_state.attendance_store.items():
            if parent_cid in recs:
                child_att_log.append({"Date": date_str, "Status": recs[parent_cid]})

        if child_att_log:
            df_att = pd.DataFrame(child_att_log).sort_values(by="Date", ascending=False)
            total_days = len(df_att)
            present_days = len(df_att[df_att["Status"] == "Present"])
            absent_days = total_days - present_days
            att_pct = (present_days / total_days * 100) if total_days > 0 else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Recorded Days", total_days)
            m2.metric("Present Days", present_days)
            m3.metric("Absent Days", absent_days)
            m4.metric("Attendance Percentage", f"{att_pct:.1f}%")

            st.progress(att_pct / 100.0)
            st.markdown("### 📜 Attendance History Log")
            st.dataframe(df_att, use_container_width=True, hide_index=True)
        else:
            st.info(f"No attendance records logged yet for {child_name} ({parent_cid}). Please check back after teacher marks attendance.")

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

    # 5. Chat with Teacher (Parent View)
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
                    <span style="font-size: 1.4rem; font-weight: 800; color: #f8fafc;">{status_name}</span>
                    <span class="badge {badge_cls}">{row['Category']}</span>
                </div>
                <div style="background: rgba(255, 255, 255, 0.03); padding: 0.8rem 1rem; border-radius: 8px; margin-bottom: 1rem;">
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
                <div style="font-size: 1.2rem; font-weight: 700; color: #38bdf8;">📅 {row['Day']}</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #f8fafc; margin-top: 0.3rem;">{row['Meal']}</div>
                <div style="margin-top: 0.8rem; display: flex; gap: 1rem; flex-wrap: wrap;">
                    <span class="badge badge-info">🔥 Calories: {row.get('Calories (kcal)', 'N/A')}</span>
                    <span class="badge badge-healthy">🥩 Protein: {row.get('Protein (g)', 'N/A')}</span>
                    <span class="badge badge-overweight">🌾 Carbs: {row.get('Carbs (g)', 'N/A')}</span>
                    <span class="badge badge-underweight">🥑 Fat: {row.get('Fat (g)', 'N/A')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 0.85rem; color: #64748b; padding: 1rem 0 0.5rem; line-height: 1.6;">
    <b>AI Child Growth Advisor</b> • Powered by Streamlit, WHO Growth Standards & PyTorch<br>
    <span style="font-size: 0.78rem; color: #94a3b8;">⚠️ <i>Educational and screening support tool. Not a substitute for professional pediatric clinical evaluation.</i></span>
</div>
""", unsafe_allow_html=True)
