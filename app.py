import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from datetime import datetime

# ========================================================
# Page Configuration
# ========================================================
st.set_page_config(
    page_title="Advanced AI Health Intelligence Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================================================
# Advanced Styling with Glassmorphism & Animations
# ========================================================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        animation: gradientShift 15s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp {
        background: transparent;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
    }

    .glass-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 45px 0 rgba(31, 38, 135, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        backdrop-filter: blur(25px);
        border-radius: 30px;
        padding: 50px;
        margin: 30px 0;
        border: 2px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        animation: float 6s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }

    .hero-title {
        font-size: 4.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        text-shadow: 0 4px 20px rgba(255, 255, 255, 0.3);
        animation: pulse 3s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }

    .hero-subtitle {
        font-size: 1.4rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 15px;
        font-weight: 400;
    }

    /* Advanced Metrics Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(15px);
        border-radius: 18px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }

    .metric-card:hover::before {
        left: 100%;
    }

    .metric-card:hover {
        transform: scale(1.05);
        border: 1px solid rgba(255, 255, 255, 0.4);
    }

    .metric-icon {
        font-size: 3rem;
        margin-bottom: 15px;
        display: inline-block;
        animation: bounce 2s infinite;
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }

    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        margin: 10px 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }

    .metric-label {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .metric-change {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7);
        margin-top: 8px;
    }

    /* AI Insight Cards */
    .insight-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 25px;
        margin: 15px 0;
        border-left: 5px solid #10b981;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
        animation: slideInLeft 0.6s ease-out;
    }

    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    .insight-warning {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(239, 68, 68, 0.15) 100%);
    }

    .insight-danger {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.15) 100%);
    }

    .insight-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
    }

    .insight-text {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9);
        line-height: 1.6;
    }

    /* Section Headers */
    .section-header {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 20px 30px;
        margin: 30px 0 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        display: flex;
        align-items: center;
        gap: 15px;
    }

    .section-icon {
        font-size: 2rem;
        animation: rotate 3s linear infinite;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }

    .section-subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.7);
        margin: 5px 0 0 0;
    }

    /* Interactive Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 15px 35px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }

    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 18px;
        padding: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 600;
        font-size: 1.02rem;
        padding: 14px 32px;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(
            135deg,
            rgba(102, 126, 234, 0.3) 0%,
            rgba(118, 75, 162, 0.3) 100%
        );
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.35);
    }

    /* Data Tables */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Plotly Charts Enhancement */
    .js-plotly-plot {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }

    [data-testid="stSidebar"] .element-container {
        color: white;
    }

    /* Loading Animation */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }

    .loading {
        background: linear-gradient(90deg, rgba(255,255,255,0.1) 25%, rgba(255,255,255,0.2) 50%, rgba(255,255,255,0.1) 75%);
        background-size: 1000px 100%;
        animation: shimmer 2s infinite;
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 6px 15px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }

    .status-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }

    .status-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }

    .status-danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }

    /* Tooltip Enhancement */
    .tooltip-custom {
        background: rgba(0, 0, 0, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 10px;
        color: white;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ========================================================
# Enhanced Data Generation with More Features
# ========================================================
@st.cache_data
def generate_advanced_data(n_users: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)

    # Basic demographics
    user_id = np.arange(1, n_users + 1)
    age = rng.integers(15, 75, size=n_users)
    age_group = pd.cut(
        age,
        bins=[14, 22, 30, 40, 50, 60, 75],
        labels=["Gen Z", "Millennials", "Young Adults", "Middle Age", "Senior", "Elderly"],
    )
    gender = rng.choice(["Male", "Female", "Other"], size=n_users, p=[0.48, 0.48, 0.04])
    location = rng.choice(["Urban", "Suburban", "Rural"], size=n_users, p=[0.55, 0.35, 0.10])
    occupation = rng.choice(
        ["Student", "Professional", "Healthcare", "Tech", "Retired", "Other"],
        size=n_users,
        p=[0.20, 0.30, 0.15, 0.20, 0.10, 0.05],
    )

    # Digital behavior
    screen_hours = np.clip(rng.normal(7.2, 2.5, size=n_users), 1.0, 18.0)
    sleep_hours = np.clip(
        7.8 - 0.35 * (screen_hours - 7) + rng.normal(0, 0.9, size=n_users),
        3.0,
        12.0,
    )

    # Mental health indicators
    stress = np.clip(
        5.5 + 0.65 * (screen_hours - 7) + rng.normal(0, 1.3, size=n_users),
        1,
        10,
    )
    anxiety = np.clip(
        5.0 + 0.55 * stress + rng.normal(0, 1.1, size=n_users),
        1,
        10,
    )
    depression = np.clip(
        4.5 + 0.45 * stress + 0.3 * anxiety + rng.normal(0, 1.2, size=n_users),
        1,
        10,
    )
    focus = np.clip(
        8.0 - 0.45 * (screen_hours - 7) - 0.35 * stress + rng.normal(0, 0.9, size=n_users),
        1,
        10,
    )
    wellbeing = np.clip(
        8.2 - 0.45 * stress + 0.45 * (sleep_hours - 7.5) + rng.normal(0, 0.8, size=n_users),
        1,
        10,
    )
    mood = np.clip(
        7.5 - 0.35 * stress + 0.25 * wellbeing + rng.normal(0, 1.0, size=n_users),
        1,
        10,
    )
    energy = np.clip(
        7.8 - 0.3 * stress + 0.4 * (sleep_hours - 7.5) + rng.normal(0, 0.9, size=n_users),
        1,
        10,
    )

    # App usage patterns
    phone_unlocks = np.clip(
        rng.normal(110 + 15 * (screen_hours - 7), 35, size=n_users),
        10,
        400,
    ).astype(int)
    notifications = np.clip(
        rng.normal(95 + 12 * (screen_hours - 7), 32, size=n_users),
        5,
        350,
    ).astype(int)
    social_minutes = np.clip(
        rng.normal(165 + 18 * (screen_hours - 7), 55, size=n_users),
        5,
        480,
    ).astype(int)
    gaming_minutes = np.clip(
        rng.normal(70 + 10 * (screen_hours - 7), 40, size=n_users),
        0,
        360,
    ).astype(int)
    work_minutes = np.clip(
        rng.normal(220, 80, size=n_users),
        0,
        600,
    ).astype(int)

    # Exercise and lifestyle
    exercise_minutes = np.clip(
        rng.normal(40 - 2.5 * (screen_hours - 7), 28, size=n_users),
        0,
        200,
    ).astype(int)
    outdoor_time = np.clip(
        rng.normal(55 - 3.5 * (screen_hours - 7), 35, size=n_users),
        0,
        280,
    ).astype(int)

    # Health metrics
    bmi = np.clip(rng.normal(24.5 + 0.3 * (screen_hours - 7), 4.5, size=n_users), 16, 45)
    heart_rate = np.clip(
        rng.normal(72 + 2 * stress, 10, size=n_users),
        50,
        120,
    ).astype(int)
    steps_daily = np.clip(
        rng.normal(7500 - 300 * (screen_hours - 7), 2500, size=n_users),
        1000,
        20000,
    ).astype(int)

    # Social factors
    social_support = np.clip(
        rng.normal(7.0 - 0.2 * stress, 1.5, size=n_users),
        1,
        10,
    )
    loneliness = np.clip(
        rng.normal(4.5 + 0.4 * stress - 0.5 * social_support, 1.8, size=n_users),
        1,
        10,
    )

    # Risk calculation with enhanced features
    logit = (
        0.62 * (screen_hours - 7)
        + 0.58 * (stress - 5.5)
        + 0.52 * (anxiety - 5)
        + 0.48 * (depression - 4.5)
        - 0.46 * (sleep_hours - 7.5)
        - 0.42 * (wellbeing - 8)
        - 0.38 * (mood - 7.5)
        - 0.35 * (energy - 7.8)
        + 0.22 * ((phone_unlocks - 110) / 60)
        + 0.18 * ((social_minutes - 165) / 70)
        - 0.15 * ((exercise_minutes - 40) / 35)
        - 0.12 * ((outdoor_time - 55) / 55)
        + 0.2 * (loneliness - 4.5)
        - 0.18 * (social_support - 7)
        + rng.normal(0, 0.8, size=n_users)
    )

    risk_score = 1.0 / (1.0 + np.exp(-logit))
    risk_score = np.clip(risk_score, 0.01, 0.99)
    high_risk = (rng.random(n_users) < risk_score).astype(int)

    risk_segment = pd.cut(
        risk_score,
        bins=[0.0, 0.30, 0.60, 1.0],
        labels=["Low Risk", "Moderate Risk", "High Risk"],
    )

    # Engagement metrics
    last_active = pd.Timestamp.now() - pd.to_timedelta(
        rng.integers(0, 45, size=n_users), unit="D"
    )
    engagement_score = np.clip(
        rng.normal(7.8 - 0.25 * (screen_hours - 7), 1.6, size=n_users),
        1,
        10,
    )

    # Treatment history
    seeking_help = rng.choice([0, 1], size=n_users, p=[0.7, 0.3])
    medication = rng.choice([0, 1], size=n_users, p=[0.8, 0.2])

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "age": age,
            "age_group": age_group,
            "gender": gender,
            "location": location,
            "occupation": occupation,
            "screen_hours": screen_hours.round(2),
            "sleep_hours": sleep_hours.round(2),
            "stress": stress.round(1),
            "anxiety": anxiety.round(1),
            "depression": depression.round(1),
            "focus": focus.round(1),
            "wellbeing": wellbeing.round(1),
            "mood": mood.round(1),
            "energy": energy.round(1),
            "phone_unlocks": phone_unlocks,
            "notifications": notifications,
            "social_minutes": social_minutes,
            "gaming_minutes": gaming_minutes,
            "work_minutes": work_minutes,
            "exercise_minutes": exercise_minutes,
            "outdoor_time": outdoor_time,
            "bmi": bmi.round(1),
            "heart_rate": heart_rate,
            "steps_daily": steps_daily,
            "social_support": social_support.round(1),
            "loneliness": loneliness.round(1),
            "seeking_help": seeking_help,
            "medication": medication,
            "risk_score": risk_score,
            "high_risk": high_risk,
            "risk_segment": risk_segment,
            "last_active": last_active,
            "engagement_score": engagement_score.round(1),
        }
    )

    # Time series data (90 days)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=90, freq="D")
    time_series = pd.DataFrame(
        {
            "date": dates,
            "avg_screen": 7.0
            + np.sin(np.arange(90) / 7) * 1.8
            + rng.normal(0, 0.4, 90),
            "avg_stress": 5.8
            + np.sin(np.arange(90) / 10) * 1.4
            + rng.normal(0, 0.4, 90),
            "avg_wellbeing": 7.8
            - np.sin(np.arange(90) / 10) * 1.2
            + rng.normal(0, 0.4, 90),
            "high_risk_count": (
                800
                + np.sin(np.arange(90) / 7) * 120
                + rng.normal(0, 40, 90)
            ).astype(int),
            "avg_sleep": 7.5
            - np.sin(np.arange(90) / 12) * 0.8
            + rng.normal(0, 0.3, 90),
            "engagement": 7.5
            + np.sin(np.arange(90) / 8) * 1.0
            + rng.normal(0, 0.3, 90),
        }
    )

    # Hourly patterns
    hours = list(range(24))
    hourly = pd.DataFrame(
        {
            "hour": hours,
            "screen_time": [
                25 if h < 6 else 55 + 20 * np.sin((h - 6) / 3.5) for h in hours
            ],
            "notifications": [
                12 if h < 6 else 35 + 18 * np.sin((h - 6) / 3.8) for h in hours
            ],
            "stress": [
                3.5 if h < 6 else 5.5 + 2.5 * np.sin((h - 10) / 3.2) for h in hours
            ],
            "energy": [
                4 if h < 6 else 7.5 - 1.5 * np.sin((h - 14) / 4) for h in hours
            ],
        }
    )

    return df, time_series, hourly


# ========================================================
# Helper Functions
# ========================================================
def calculate_metrics(df_metrics: pd.DataFrame, threshold: float = 0.5):
    y_true = df_metrics["high_risk"].values
    y_score = df_metrics["risk_score"].values
    y_pred = (y_score >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": auc,
        "ap": ap,
        "brier": brier,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def generate_ai_insights(df_view: pd.DataFrame, metrics: dict):
    insights = []

    if len(df_view) == 0:
        return insights

    # Critical risk concentration
    high_risk_pct = (df_view["risk_segment"] == "High Risk").mean() * 100
    if high_risk_pct > 35:
        insights.append(
            {
                "type": "danger",
                "title": "🚨 CRITICAL: High Risk Alert",
                "text": f"{high_risk_pct:.1f}% of monitored population in high-risk category. Systematic intervention programs are recommended.",
                "priority": "critical",
            }
        )
    elif high_risk_pct > 25:
        insights.append(
            {
                "type": "warning",
                "title": "⚠️ WARNING: Elevated Risk Levels",
                "text": f"{high_risk_pct:.1f}% high-risk users detected. Targeted support strategies are advised.",
                "priority": "high",
            }
        )

    # Screen time impact analysis
    high_risk_screen = df_view[df_view["risk_segment"] == "High Risk"][
        "screen_hours"
    ].mean()
    low_risk_screen = df_view[df_view["risk_segment"] == "Low Risk"][
        "screen_hours"
    ].mean()
    if high_risk_screen - low_risk_screen > 2.5:
        insights.append(
            {
                "type": "info",
                "title": "📱 Digital Exposure Correlation",
                "text": f"High-risk cohort shows {high_risk_screen - low_risk_screen:.1f}h higher daily screen time compared to low-risk users.",
                "priority": "medium",
            }
        )

    # Sleep deprivation alert
    avg_sleep = df_view["sleep_hours"].mean()
    if avg_sleep < 6.5:
        critical_sleep = (df_view["sleep_hours"] < 6).sum()
        insights.append(
            {
                "type": "warning",
                "title": "😴 Sleep Deficit Detected",
                "text": f"Population average sleep: {avg_sleep:.1f}h. {critical_sleep:,} users are below 6 hours.",
                "priority": "high",
            }
        )

    # Mental health trends
    avg_stress = df_view["stress"].mean()
    avg_anxiety = df_view["anxiety"].mean()
    if avg_stress > 6.5 or avg_anxiety > 6.5:
        insights.append(
            {
                "type": "warning",
                "title": "🧠 Mental Health Stress Indicators",
                "text": f"Elevated psychological metrics: Stress {avg_stress:.1f}/10, Anxiety {avg_anxiety:.1f}/10.",
                "priority": "high",
            }
        )

    # Model performance
    if metrics["auc"] > 0.88:
        insights.append(
            {
                "type": "success",
                "title": "✅ Model Performance: Strong Signal",
                "text": f"Predictive performance on this dataset is strong (AUC: {metrics['auc']:.3f}). Risk stratification is internally consistent with the available features.",
                "priority": "low",
            }
        )
    elif metrics["auc"] < 0.75:
        insights.append(
            {
                "type": "warning",
                "title": "⚠️ Model Performance Warning",
                "text": f"AUC {metrics['auc']:.3f} is below the preferred range for reliable screening. Consider recalibration or feature refinement.",
                "priority": "medium",
            }
        )

    # Social isolation patterns
    avg_loneliness = df_view["loneliness"].mean()
    if avg_loneliness > 6.0:
        insights.append(
            {
                "type": "warning",
                "title": "👥 Social Isolation Signal",
                "text": f"Average loneliness score: {avg_loneliness:.1f}/10. Social support and engagement may be helpful.",
                "priority": "medium",
            }
        )

    # Physical activity deficit
    avg_exercise = df_view["exercise_minutes"].mean()
    if avg_exercise < 25:
        sedentary_count = (df_view["exercise_minutes"] < 15).sum()
        insights.append(
            {
                "type": "info",
                "title": "🏃 Physical Activity Deficit",
                "text": f"Average exercise: {avg_exercise:.0f} min/day. {sedentary_count:,} users show very low activity.",
                "priority": "medium",
            }
        )

    return sorted(
        insights,
        key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}[x["priority"]],
    )


# ========================================================
# Load Data
# ========================================================
df, time_series_df, hourly_df = generate_advanced_data()

# ========================================================
# Sidebar Configuration
# ========================================================
with st.sidebar:
    st.markdown(
        """
    <div style='text-align: center; padding: 20px;'>
        <h1 style='font-size: 2.5rem; margin: 0; color: white;'>🧬</h1>
        <h3 style='margin: 10px 0; color: white;'>AI Health Intelligence</h3>
        <p style='color: rgba(255,255,255,0.7); font-size: 0.9rem;'>Advanced analytics for digital wellbeing</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### 🎯 Risk Configuration")
    threshold = st.slider(
        "Risk Classification Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Adjust sensitivity for high-risk classification",
    )

    st.markdown("---")
    st.markdown("### 🔍 Advanced Filters")

    selected_segments = st.multiselect(
        "Risk Segments",
        options=["Low Risk", "Moderate Risk", "High Risk"],
        default=["Low Risk", "Moderate Risk", "High Risk"],
    )

    selected_age_groups = st.multiselect(
        "Age Demographics",
        options=df["age_group"].unique().tolist(),
        default=df["age_group"].unique().tolist(),
    )

    selected_gender = st.multiselect(
        "Gender",
        options=["Male", "Female", "Other"],
        default=["Male", "Female", "Other"],
    )

    selected_occupation = st.multiselect(
        "Occupation Type",
        options=df["occupation"].unique().tolist(),
        default=df["occupation"].unique().tolist(),
    )

    screen_time_range = st.slider(
        "Screen Time (hours/day)",
        min_value=0.0,
        max_value=18.0,
        value=(0.0, 18.0),
        step=0.5,
    )

    stress_range = st.slider(
        "Stress Level Range",
        min_value=1.0,
        max_value=10.0,
        value=(1.0, 10.0),
        step=0.5,
    )

    st.markdown("---")
    st.markdown("### 📊 Display Preferences")

    show_animations = st.checkbox("Enable Animations", value=True)
    show_insights = st.checkbox("AI-Powered Insights", value=True)
    show_advanced_metrics = st.checkbox("Advanced Analytics", value=True)
    real_time_mode = st.checkbox("Real-Time Mode (UI only)", value=False)

    st.markdown("---")
    st.markdown("### 📥 Data Export")

    export_format = st.selectbox("Export Format", ["CSV", "JSON"])

    if st.button("📊 Export Dataset", use_container_width=True):
        if export_format == "CSV":
            data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                data=data,
                file_name=f"health_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        elif export_format == "JSON":
            data = df.to_json(orient="records", indent=2).encode("utf-8")
            st.download_button(
                "⬇️ Download JSON",
                data=data,
                file_name=f"health_data_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True,
            )



# Apply filters
mask = (
    df["risk_segment"].isin(selected_segments)
    & df["age_group"].isin(selected_age_groups)
    & df["gender"].isin(selected_gender)
    & df["occupation"].isin(selected_occupation)
    & (df["screen_hours"] >= screen_time_range[0])
    & (df["screen_hours"] <= screen_time_range[1])
    & (df["stress"] >= stress_range[0])
    & (df["stress"] <= stress_range[1])
)

df_filtered = df[mask].copy()
no_filter_data = len(df_filtered) == 0
plot_df = df_filtered.copy() if not no_filter_data else df.copy()

plot_df["flagged"] = (plot_df["risk_score"] >= threshold).astype(int)

if plot_df["high_risk"].nunique() < 2:
    df_metrics = df.copy()
    metrics_scope = "full dataset (filtered data insufficient)"
else:
    df_metrics = plot_df.copy()
    metrics_scope = "filtered subset" if not no_filter_data else "full dataset"

metrics = calculate_metrics(df_metrics, threshold)
insights = generate_ai_insights(plot_df, metrics) if show_insights else []

if no_filter_data:
    st.warning("⚠️ No data matches current filters. Displaying full dataset.")

kpi_df = plot_df

# ========================================================
# Hero Section
# ========================================================
st.markdown(
    """
<div class='hero-section'>
    <div style='text-align: center;'>
        <h1 class='hero-title'>🧬 AI Health Intelligence Platform</h1>
        <p class='hero-subtitle'>
            Advanced predictive analytics for digital wellbeing and mental health risk assessment
        </p>
        <p style='color: rgba(255,255,255,0.6); font-size: 0.95rem; margin-top: 15px;'>
            Interactive monitoring • Machine learning predictions • Scenario-based interventions
        </p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ========================================================
# Key Metrics Dashboard
# ========================================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_users = len(plot_df)
    st.markdown(
        f"""
    <div class='metric-card'>
        <div class='metric-icon'>👥</div>
        <div class='metric-value'>{total_users:,}</div>
        <div class='metric-label'>Active Users</div>
        <div class='metric-change'>Monitoring scope</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    high_risk_count = (plot_df["risk_segment"] == "High Risk").sum()
    high_risk_pct = (high_risk_count / len(plot_df) * 100) if len(plot_df) > 0 else 0
    st.markdown(
        f"""
    <div class='metric-card'>
        <div class='metric-icon'>🚨</div>
        <div class='metric-value'>{high_risk_count:,}</div>
        <div class='metric-label'>High Risk</div>
        <div class='metric-change'>{high_risk_pct:.1f}% of population</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
    <div class='metric-card'>
        <div class='metric-icon'>🎯</div>
        <div class='metric-value'>{metrics['auc']:.3f}</div>
        <div class='metric-label'>Model AUC</div>
        <div class='metric-change'>Predictive performance</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    avg_screen = kpi_df["screen_hours"].mean()
    screen_delta = avg_screen - 7.0
    st.markdown(
        f"""
    <div class='metric-card'>
        <div class='metric-icon'>📱</div>
        <div class='metric-value'>{avg_screen:.1f}h</div>
        <div class='metric-label'>Avg Screen Time</div>
        <div class='metric-change'>{"↑" if screen_delta > 0 else "↓"} {abs(screen_delta):.1f}h vs baseline</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col5:
    avg_wellbeing = kpi_df["wellbeing"].mean()
    st.markdown(
        f"""
    <div class='metric-card'>
        <div class='metric-icon'>💚</div>
        <div class='metric-value'>{avg_wellbeing:.1f}</div>
        <div class='metric-label'>Wellbeing Score</div>
        <div class='metric-change'>Out of 10.0</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Secondary KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Precision", f"{metrics['precision']:.1%}", f"F1: {metrics['f1']:.3f}")

with col2:
    st.metric(
        "Recall",
        f"{metrics['recall']:.1%}",
        f"Specificity: {metrics['specificity']:.3f}",
    )

with col3:
    flagged_pct = plot_df["flagged"].mean()
    st.metric("Flagged Users", f"{flagged_pct:.1%}", f"At threshold {threshold:.2f}")

with col4:
    avg_stress = kpi_df["stress"].mean()
    st.metric(
        "Avg Stress",
        f"{avg_stress:.1f}/10",
        f"Anxiety: {kpi_df['anxiety'].mean():.1f}/10",
    )

# ========================================================
# AI Insights Section
# ========================================================
if show_insights and insights:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
    <div class='section-header'>
        <div class='section-icon'>🤖</div>
        <div>
            <div class='section-title'>AI-Powered Insights</div>
            <div class='section-subtitle'>Machine learning analysis and recommendations</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    for insight in insights[:6]:
        alert_class = ""
        if insight["type"] == "warning":
            alert_class = "insight-warning"
        elif insight["type"] == "danger":
            alert_class = "insight-danger"

        st.markdown(
            f"""
        <div class='insight-card {alert_class}'>
            <div class='insight-title'>{insight['title']}</div>
            <div class='insight-text'>{insight['text']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ========================================================
# Main Tabs
# ========================================================
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Executive Dashboard",
        "Risk Analytics",
        "Behavioral Insights",
        "Model Performance",
        "Scenario Simulator",
        "Clinical Reports",
    ]
)

# ========================================================
# TAB 1: Executive Dashboard
# ========================================================
with tab1:
    st.markdown(
        """
    <div class='section-header'>
        <div class='section-icon'>📊</div>
        <div>
            <div class='section-title'>Population Health Overview</div>
            <div class='section-subtitle'>Comprehensive metrics and trend analysis</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown("#### Risk Score Distribution & Segmentation")
        fig = px.histogram(
            plot_df,
            x="risk_score",
            color="risk_segment",
            nbins=60,
            color_discrete_map={
                "Low Risk": "#10b981",
                "Moderate Risk": "#f59e0b",
                "High Risk": "#ef4444",
            },
            labels={"risk_score": "Risk Score", "count": "User Count"},
        )
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#ffffff",
            line_width=3,
            annotation_text=f"Threshold: {threshold:.2f}",
            annotation_position="top right",
        )
        fig.update_layout(
            height=450,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", size=12),
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Risk Segment Distribution")
        segment_counts = plot_df["risk_segment"].value_counts()
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=segment_counts.index,
                    values=segment_counts.values,
                    hole=0.6,
                    marker=dict(
                        colors=["#10b981", "#f59e0b", "#ef4444"],
                        line=dict(color="rgba(0,0,0,0.5)", width=3),
                    ),
                    textfont=dict(size=15, color="#fff"),
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            height=450,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", size=12),
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    # 90-Day Trends
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 90-Day Population Health Trends")

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Screen Time Trajectory",
            "Stress Evolution",
            "Wellbeing Index",
            "High-Risk Population",
            "Sleep Patterns",
            "Engagement Metrics",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Screen time
    fig.add_trace(
        go.Scatter(
            x=time_series_df["date"],
            y=time_series_df["avg_screen"],
            mode="lines",
            name="Screen Time",
            line=dict(color="#3b82f6", width=3),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.3)",
        ),
        row=1,
        col=1,
    )

    # Stress
    fig.add_trace(
        go.Scatter(
            x=time_series_df["date"],
            y=time_series_df["avg_stress"],
            mode="lines",
            name="Stress",
            line=dict(color="#ef4444", width=3),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.3)",
        ),
        row=1,
        col=2,
    )

    # Wellbeing
    fig.add_trace(
        go.Scatter(
            x=time_series_df["date"],
            y=time_series_df["avg_wellbeing"],
            mode="lines",
            name="Wellbeing",
            line=dict(color="#10b981", width=3),
            fill="tozeroy",
            fillcolor="rgba(16,185,129,0.3)",
        ),
        row=1,
        col=3,
    )

    # High-risk count
    fig.add_trace(
        go.Bar(
            x=time_series_df["date"],
            y=time_series_df["high_risk_count"],
            name="High-Risk Users",
            marker=dict(color="#f59e0b"),
        ),
        row=2,
        col=1,
    )

    # Sleep
    fig.add_trace(
        go.Scatter(
            x=time_series_df["date"],
            y=time_series_df["avg_sleep"],
            mode="lines+markers",
            name="Sleep",
            line=dict(color="#8b5cf6", width=3),
            fill="tozeroy",
            fillcolor="rgba(139,92,246,0.3)",
        ),
        row=2,
        col=2,
    )

    # Engagement
    fig.add_trace(
        go.Scatter(
            x=time_series_df["date"],
            y=time_series_df["engagement"],
            mode="lines",
            name="Engagement",
            line=dict(color="#ec4899", width=3),
            fill="tozeroy",
            fillcolor="rgba(236,72,153,0.3)",
        ),
        row=2,
        col=3,
    )

    fig.update_layout(
        height=700,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff", size=11),
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")

    st.plotly_chart(fig, use_container_width=True)

    # Demographics Analysis
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### Age Distribution (Avg Risk)")
        age_risk = plot_df.groupby("age_group")["risk_score"].mean().reset_index()
        fig = px.bar(
            age_risk,
            x="age_group",
            y="risk_score",
            color="risk_score",
            color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
        )
        fig.update_layout(
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Gender Distribution")
        gender_counts = plot_df["gender"].value_counts()
        fig = px.bar(
            x=gender_counts.index,
            y=gender_counts.values,
            color=gender_counts.index,
            color_discrete_map={
                "Male": "#3b82f6",
                "Female": "#ec4899",
                "Other": "#8b5cf6",
            },
        )
        fig.update_layout(
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("#### Location Distribution")
        location_counts = plot_df["location"].value_counts()
        fig = px.pie(
            values=location_counts.values,
            names=location_counts.index,
            color_discrete_sequence=["#3b82f6", "#8b5cf6", "#10b981"],
            hole=0.4,
        )
        fig.update_layout(
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("#### Occupation (Avg Risk)")
        occ_risk = (
            plot_df.groupby("occupation")["risk_score"]
            .mean()
            .sort_values(ascending=False)
        )
        fig = px.bar(
            x=occ_risk.values,
            y=occ_risk.index,
            orientation="h",
            color=occ_risk.values,
            color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
        )
        fig.update_layout(
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

# ========================================================
# TAB 2: Risk Analytics
# ========================================================
with tab2:
    st.markdown(
        """
    <div class='section-header'>
        <div class='section-icon'>🎯</div>
        <div>
            <div class='section-title'>Advanced Risk Assessment</div>
            <div class='section-subtitle'>Deep dive into risk factors and correlations</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("#### Multi-Factor Correlation Matrix")
        corr_cols = [
            "screen_hours",
            "sleep_hours",
            "stress",
            "anxiety",
            "depression",
            "wellbeing",
            "mood",
            "energy",
            "social_support",
            "loneliness",
            "risk_score",
        ]
        corr_matrix = plot_df[corr_cols].corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale=[[0, "#ef4444"], [0.5, "#f3f4f6"], [1, "#10b981"]],
                text=corr_matrix.values.round(2),
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Correlation"),
            )
        )
        fig.update_layout(
            height=550,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", size=10),
            margin=dict(l=120, r=40, t=20, b=120),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Risk Distribution by Segment")
        fig = go.Figure()
        for segment in ["Low Risk", "Moderate Risk", "High Risk"]:
            data = plot_df[plot_df["risk_segment"] == segment]["risk_score"]
            color = {
                "Low Risk": "#10b981",
                "Moderate Risk": "#f59e0b",
                "High Risk": "#ef4444",
            }[segment]
            fig.add_trace(
                go.Box(y=data, name=segment, marker_color=color, boxmean="sd")
            )
        fig.update_layout(
            height=550,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            yaxis_title="Risk Score",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ROC and Confusion Matrix
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confusion Matrix Analysis")
        cm = confusion_matrix(
            df_metrics["high_risk"].values,
            (df_metrics["risk_score"].values >= threshold).astype(int),
        )
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        cm_display = [[tn, fp], [fn, tp]]
        fig = go.Figure(
            data=go.Heatmap(
                z=cm_display,
                x=["Predicted Negative", "Predicted Positive"],
                y=["Actual Negative", "Actual Positive"],
                text=cm_display,
                texttemplate="%{text}",
                textfont={"size": 24},
                colorscale=[[0, "#1e293b"], [1, "#3b82f6"]],
                showscale=False,
            )
        )
        fig.update_layout(
            height=450,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", size=13),
            margin=dict(l=100, r=20, t=20, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### ROC Curve Analysis")
        fpr, tpr, _ = roc_curve(
            df_metrics["high_risk"].values, df_metrics["risk_score"].values
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC (AUC = {metrics['auc']:.3f})",
                line=dict(color="#3b82f6", width=4),
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.3)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(color="#64748b", width=2, dash="dash"),
            )
        )
        fig.update_layout(
            height=450,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Risk Factor Importance (Conceptual)")

    features = [
        "Screen Time",
        "Stress",
        "Sleep Quality",
        "Anxiety",
        "Depression",
        "Wellbeing",
        "Social Support",
        "Loneliness",
        "Phone Unlocks",
        "Social Media",
        "Exercise",
        "Outdoor Time",
    ]
    importance = [0.62, 0.58, 0.46, 0.52, 0.48, 0.42, 0.18, 0.2, 0.22, 0.18, 0.15, 0.12]

    fig = px.bar(
        x=importance,
        y=features,
        orientation="h",
        color=importance,
        color_continuous_scale=["#64748b", "#3b82f6", "#8b5cf6", "#ec4899"],
    )
    fig.update_layout(
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        xaxis_title="Relative Importance (Model Coefficients)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Risk scatter plots
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    sample = (
        plot_df.sample(n=min(2000, len(plot_df)), random_state=42)
        if len(plot_df) > 0
        else plot_df
    )

    with col1:
        st.markdown("#### Screen Time vs Sleep")
        if len(sample) > 0:
            fig = px.scatter(
                sample,
                x="screen_hours",
                y="sleep_hours",
                color="risk_score",
                size="stress",
                color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                hover_data=["stress", "wellbeing", "age"],
            )
            fig.update_layout(
                height=450,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Stress vs Wellbeing")
        if len(sample) > 0:
            fig = px.scatter(
                sample,
                x="stress",
                y="wellbeing",
                color="risk_segment",
                size="risk_score",
                color_discrete_map={
                    "Low Risk": "#10b981",
                    "Moderate Risk": "#f59e0b",
                    "High Risk": "#ef4444",
                },
                hover_data=["screen_hours", "sleep_hours", "depression"],
            )
            fig.update_layout(
                height=450,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff"),
            )
            st.plotly_chart(fig, use_container_width=True)

# ========================================================
# TAB 3: Behavioral Insights
# ========================================================
with tab3:
    st.markdown(
        """
    <div class='section-header'>
        <div class='section-icon'>📱</div>
        <div>
            <div class='section-title'>Digital Behavior Analysis</div>
            <div class='section-subtitle'>Usage patterns and lifestyle correlations</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 24-hour patterns
    st.markdown("#### Circadian Activity Patterns")
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=hourly_df["hour"],
            y=hourly_df["screen_time"],
            name="Screen Time",
            mode="lines+markers",
            line=dict(color="#3b82f6", width=3),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.3)",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=hourly_df["hour"],
            y=hourly_df["notifications"],
            name="Notifications",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=hourly_df["hour"],
            y=hourly_df["stress"],
            name="Stress Level",
            mode="lines+markers",
            line=dict(color="#ef4444", width=2, dash="dash"),
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=hourly_df["hour"],
            y=hourly_df["energy"],
            name="Energy Level",
            mode="lines",
            line=dict(color="#10b981", width=2, dash="dot"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        hovermode="x unified",
    )

    fig.update_xaxes(
        title_text="Hour of Day", showgrid=True, gridcolor="rgba(255,255,255,0.1)"
    )
    fig.update_yaxes(
        title_text="Activity Level",
        secondary_y=False,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.1)",
    )
    fig.update_yaxes(title_text="Psychological Metrics", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### App Usage Distribution")
        usage_data = pd.DataFrame(
            {
                "Category": ["Social Media", "Work/Study", "Gaming", "Entertainment", "Other"],
                "Minutes": [
                    plot_df["social_minutes"].mean(),
                    plot_df["work_minutes"].mean(),
                    plot_df["gaming_minutes"].mean(),
                    80,
                    45,
                ],
            }
        )
        fig = px.pie(
            usage_data,
            values="Minutes",
            names="Category",
            color_discrete_sequence=[
                "#3b82f6",
                "#10b981",
                "#8b5cf6",
                "#ec4899",
                "#64748b",
            ],
            hole=0.5,
        )
        fig.update_layout(
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Digital Interaction Metrics")
        interaction_data = (
            plot_df.groupby("risk_segment")
            .agg({"phone_unlocks": "mean", "notifications": "mean"})
            .reset_index()
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=interaction_data["risk_segment"],
                y=interaction_data["phone_unlocks"],
                name="Phone Unlocks",
                marker_color="#3b82f6",
            )
        )
        fig.add_trace(
            go.Bar(
                x=interaction_data["risk_segment"],
                y=interaction_data["notifications"],
                name="Notifications",
                marker_color="#f59e0b",
            )
        )
        fig.update_layout(
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("#### Physical Activity Balance")
        balance_data = pd.DataFrame(
            {
                "Activity": ["Exercise", "Outdoor", "Screen Time"],
                "Minutes": [
                    plot_df["exercise_minutes"].mean(),
                    plot_df["outdoor_time"].mean(),
                    plot_df["screen_hours"].mean() * 60,
                ],
            }
        )
        fig = px.bar(
            balance_data,
            x="Activity",
            y="Minutes",
            color="Activity",
            color_discrete_sequence=["#10b981", "#8b5cf6", "#ef4444"],
        )
        fig.update_layout(
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Health metrics
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Population Health Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        high_screen = len(plot_df[plot_df["screen_hours"] > 10])
        st.metric(
            "Excessive Screen Use",
            f"{high_screen:,}",
            f"{(high_screen/len(plot_df)*100):.1f}% of users",
        )

    with col2:
        sleep_deficit = len(plot_df[plot_df["sleep_hours"] < 6])
        st.metric(
            "Sleep Deficit",
            f"{sleep_deficit:,}",
            f"{(sleep_deficit/len(plot_df)*100):.1f}% critical",
        )

    with col3:
        high_stress = len(plot_df[plot_df["stress"] > 7])
        st.metric(
            "High Stress",
            f"{high_stress:,}",
            f"{(high_stress/len(plot_df)*100):.1f}% elevated",
        )

    with col4:
        sedentary = len(plot_df[plot_df["exercise_minutes"] < 20])
        st.metric(
            "Sedentary Lifestyle",
            f"{sedentary:,}",
            f"{(sedentary/len(plot_df)*100):.1f}% inactive",
        )

# ========================================================
# TAB 4: Model Performance
# ========================================================
with tab4:
    st.markdown(
        """
    <div class='section-header'>
        <div class='section-icon'>🔬</div>
        <div>
            <div class='section-title'>ML Model Performance</div>
            <div class='section-subtitle'>Evaluation and diagnostics on the current cohort</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Performance metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    metrics_data = [
        ("AUC-ROC", metrics["auc"], "🎯"),
        ("Precision", metrics["precision"], "🔍"),
        ("Recall", metrics["recall"], "📊"),
        ("F1 Score", metrics["f1"], "⚡"),
        ("Brier Score", metrics["brier"], "📈"),
    ]

    for i, (label, value, icon) in enumerate(metrics_data):
        with [col1, col2, col3, col4, col5][i]:
            st.markdown(
                f"""
            <div class='metric-card'>
                <div class='metric-icon'>{icon}</div>
                <div class='metric-value'>{value:.3f}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.caption(f"Analysis scope: {metrics_scope}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Threshold analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Threshold Optimization Curve")
        thresholds = np.linspace(0.1, 0.9, 60)
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for t in thresholds:
            temp_metrics = calculate_metrics(df_metrics, t)
            precision_scores.append(temp_metrics["precision"])
            recall_scores.append(temp_metrics["recall"])
            f1_scores.append(temp_metrics["f1"])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=precision_scores,
                mode="lines",
                name="Precision",
                line=dict(color="#10b981", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=recall_scores,
                mode="lines",
                name="Recall",
                line=dict(color="#3b82f6", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=f1_scores,
                mode="lines",
                name="F1 Score",
                line=dict(color="#8b5cf6", width=3),
            )
        )
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#f59e0b",
            line_width=3,
            annotation_text=f"Current: {threshold:.2f}",
        )
        fig.update_layout(
            height=450,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            xaxis_title="Decision Threshold",
            yaxis_title="Score",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Precision-Recall Curve")
        precision_vals, recall_vals, _ = precision_recall_curve(
            df_metrics["high_risk"].values, df_metrics["risk_score"].values
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=recall_vals,
                y=precision_vals,
                mode="lines",
                name=f"PR Curve (AP = {metrics['ap']:.3f})",
                line=dict(color="#8b5cf6", width=4),
                fill="tozeroy",
                fillcolor="rgba(139,92,246,0.3)",
            )
        )
        fig.update_layout(
            height=450,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            xaxis_title="Recall",
            yaxis_title="Precision",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Calibration
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Model Calibration Analysis")

    df_cal = df_metrics.copy()
    df_cal["score_bin"] = pd.qcut(
        df_cal["risk_score"], q=10, duplicates="drop"
    )
    cal_stats = (
        df_cal.groupby("score_bin")
        .agg({"risk_score": "mean", "high_risk": "mean", "user_id": "count"})
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cal_stats["risk_score"],
            y=cal_stats["high_risk"],
            mode="markers+lines",
            name="Observed Rate",
            marker=dict(size=cal_stats["user_id"] / 25, color="#3b82f6"),
            line=dict(color="#3b82f6", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="#10b981", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        xaxis_title="Predicted Risk Score",
        yaxis_title="Observed Event Rate",
    )
    st.plotly_chart(fig, use_container_width=True)

# ========================================================
# TAB 5: Scenario Simulator
# ========================================================
with tab5:
    st.markdown(
        """
    <div class='section-header'>
        <div class='section-icon'>🧪</div>
        <div>
            <div class='section-title'>Interactive Risk Simulator</div>
            <div class='section-subtitle'>Test behavioral interventions and quantify risk impact</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if real_time_mode:
        st.markdown(
            """
        <div style='background: rgba(34,197,94,0.1); padding: 14px 18px; border-radius: 10px;
                    border-left: 4px solid #22c55e; margin-bottom: 14px;'>
            <span style='color:#bbf7d0; font-size:0.9rem;'>
                Real-time mode enabled — use this simulator as a live what-if console for new profiles.
            </span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
    <div style='background: rgba(59,130,246,0.12); padding: 20px; border-radius: 12px;
                border-left: 4px solid #3b82f6; margin: 16px 0;'>
        <strong>How this works:</strong><br>
        1) Configure a single user profile using the sliders below.<br>
        2) The risk engine predicts mental health risk in real-time based on the inputs.<br>
        3) Intervention scenarios show how small changes shift the risk curve.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Helper: risk computation for scenarios
    def compute_risk_from_features(params: dict) -> float:
        logit_val = (
            0.62 * (params["screen"] - 7)
            + 0.58 * (params["stress"] - 5.5)
            + 0.52 * (params["anxiety"] - 5)
            + 0.48 * (params["depression"] - 4.5)
            - 0.46 * (params["sleep"] - 7.5)
            - 0.42 * (params["wellbeing"] - 8)
            - 0.38 * (params["mood"] - 7.5)
            - 0.35 * (params["energy"] - 7.8)
            + 0.22 * ((params["unlocks"] - 110) / 60.0)
            + 0.18 * ((params["social"] - 165) / 70.0)
            - 0.15 * ((params["exercise"] - 40) / 35.0)
            - 0.12 * ((params["outdoor"] - 55) / 55.0)
            + 0.2 * (params["loneliness"] - 4.5)
            - 0.18 * (params["social_support"] - 7)
        )
        return float(np.clip(1.0 / (1.0 + np.exp(-logit_val)), 0.01, 0.99))

    # Sliders: base scenario
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 📱 Digital Behavior")
        sim_screen = st.slider(
            "Screen Time (hours/day)", 1.0, 18.0, 7.0, 0.5, key="sim_screen"
        )
        sim_unlocks = st.slider(
            "Phone Unlocks/day", 10, 400, 110, 5, key="sim_unlocks"
        )
        sim_notifications = st.slider(
            "Notifications/day", 5, 350, 95, 5, key="sim_notif"
        )
        sim_social = st.slider(
            "Social Media (min/day)", 5, 480, 165, 10, key="sim_social"
        )
        sim_gaming = st.slider(
            "Gaming (min/day)", 0, 360, 70, 10, key="sim_gaming"
        )

    with col2:
        st.markdown("##### 😴 Health & Wellbeing")
        sim_sleep = st.slider(
            "Sleep (hours/day)", 3.0, 12.0, 7.5, 0.5, key="sim_sleep"
        )
        sim_stress = st.slider(
            "Stress Level (1-10)", 1.0, 10.0, 5.5, 0.5, key="sim_stress"
        )
        sim_anxiety = st.slider(
            "Anxiety Level (1-10)", 1.0, 10.0, 5.0, 0.5, key="sim_anxiety"
        )
        sim_depression = st.slider(
            "Depression (1-10)", 1.0, 10.0, 4.5, 0.5, key="sim_depression"
        )
        sim_wellbeing = st.slider(
            "Wellbeing (1-10)", 1.0, 10.0, 8.0, 0.5, key="sim_wellbeing"
        )
        sim_mood = st.slider(
            "Mood (1-10)", 1.0, 10.0, 7.5, 0.5, key="sim_mood"
        )
        sim_energy = st.slider(
            "Energy (1-10)", 1.0, 10.0, 7.8, 0.5, key="sim_energy"
        )

    with col3:
        st.markdown("##### 🏃 Physical & Social")
        sim_exercise = st.slider(
            "Exercise (min/day)", 0, 200, 40, 5, key="sim_exercise"
        )
        sim_outdoor = st.slider(
            "Outdoor Time (min/day)", 0, 280, 55, 10, key="sim_outdoor"
        )
        sim_social_support = st.slider(
            "Social Support (1-10)", 1.0, 10.0, 7.0, 0.5, key="sim_social_support"
        )
        sim_loneliness = st.slider(
            "Loneliness (1-10)", 1.0, 10.0, 4.5, 0.5, key="sim_loneliness"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset to Population Baseline", use_container_width=True):
            st.rerun()

    # Bundle base features
    base_features = {
        "screen": sim_screen,
        "sleep": sim_sleep,
        "stress": sim_stress,
        "anxiety": sim_anxiety,
        "depression": sim_depression,
        "wellbeing": sim_wellbeing,
        "mood": sim_mood,
        "energy": sim_energy,
        "unlocks": sim_unlocks,
        "social": sim_social,
        "exercise": sim_exercise,
        "outdoor": sim_outdoor,
        "loneliness": sim_loneliness,
        "social_support": sim_social_support,
    }

    # Base scenario risk
    scenario_risk = compute_risk_from_features(base_features)
    scenario_segment = (
        "Low Risk"
        if scenario_risk < 0.30
        else "Moderate Risk"
        if scenario_risk < 0.60
        else "High Risk"
    )
    scenario_flagged = "Yes" if scenario_risk >= threshold else "No"
    percentile = (
        (df["risk_score"] < scenario_risk).mean() * 100 if len(df) > 0 else 0.0
    )

    # Results cards
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Simulation Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        color_risk = (
            "#ef4444"
            if scenario_risk > 0.6
            else "#f59e0b"
            if scenario_risk > 0.3
            else "#10b981"
        )
        st.markdown(
            f"""
        <div class='metric-card'>
            <div class='metric-label'>PREDICTED RISK SCORE</div>
            <div class='metric-value' style='color: {color_risk};'>
                {scenario_risk:.3f}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        segment_color = {
            "Low Risk": "#10b981",
            "Moderate Risk": "#f59e0b",
            "High Risk": "#ef4444",
        }[scenario_segment]
        st.markdown(
            f"""
        <div class='metric-card'>
            <div class='metric-label'>RISK CATEGORY</div>
            <div class='metric-value' style='color: {segment_color}; font-size: 2rem;'>
                {scenario_segment}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        flag_color = "#ef4444" if scenario_flagged == "Yes" else "#10b981"
        st.markdown(
            f"""
        <div class='metric-card'>
            <div class='metric-label'>INTERVENTION FLAG</div>
            <div class='metric-value' style='color: {flag_color}; font-size: 2.5rem;'>
                {scenario_flagged}
            </div>
            <div class='metric-change'>At threshold {threshold:.2f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class='metric-card'>
            <div class='metric-label'>POPULATION PERCENTILE</div>
            <div class='metric-value'>{percentile:.0f}th</div>
            <div class='metric-change'>Risk ranking vs full population</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Scenario vs population + radar
    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 📊 Scenario vs Population Averages")

        comparison_data = pd.DataFrame(
            {
                "Metric": [
                    "Screen Time",
                    "Sleep",
                    "Stress",
                    "Anxiety",
                    "Wellbeing",
                    "Exercise (×10min)",
                    "Social Support",
                ],
                "Your Scenario": [
                    sim_screen,
                    sim_sleep,
                    sim_stress,
                    sim_anxiety,
                    sim_wellbeing,
                    sim_exercise / 10.0,
                    sim_social_support,
                ],
                "Population Avg": [
                    df["screen_hours"].mean(),
                    df["sleep_hours"].mean(),
                    df["stress"].mean(),
                    df["anxiety"].mean(),
                    df["wellbeing"].mean(),
                    df["exercise_minutes"].mean() / 10.0,
                    df["social_support"].mean(),
                ]
                if len(df) > 0
                else [0, 0, 0, 0, 0, 0, 0],
            }
        )

        fig_comp = go.Figure()
        fig_comp.add_trace(
            go.Bar(
                x=comparison_data["Metric"],
                y=comparison_data["Your Scenario"],
                name="Your Scenario",
                marker_color="#3b82f6",
            )
        )
        fig_comp.add_trace(
            go.Bar(
                x=comparison_data["Metric"],
                y=comparison_data["Population Avg"],
                name="Population Average",
                marker_color="#64748b",
            )
        )
        fig_comp.update_layout(
            height=430,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            barmode="group",
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_right:
        st.markdown("#### 🧬 Profile Radar View")

        radar_categories = [
            "Screen Load",
            "Sleep",
            "Stress",
            "Anxiety",
            "Wellbeing",
            "Exercise",
            "Social Support",
        ]

        radar_values = [
            min(sim_screen / 1.8, 10.0),
            min(sim_sleep * 1.2, 10.0),
            sim_stress,
            sim_anxiety,
            sim_wellbeing,
            min(sim_exercise / 12.0, 10.0),
            sim_social_support,
        ]

        radar_categories += radar_categories[:1]
        radar_values += radar_values[:1]

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=radar_values,
                theta=radar_categories,
                fill="toself",
                name="Scenario profile",
                line=dict(color="#3b82f6", width=3),
                fillcolor="rgba(59,130,246,0.35)",
            )
        )
        fig_radar.update_layout(
            height=430,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    gridcolor="rgba(148,163,184,0.4)",
                ),
                angularaxis=dict(gridcolor="rgba(148,163,184,0.4)"),
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Quick intervention scenarios
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🧭 Quick Intervention What-If Scenarios")

    # Digital reset: reduce screen & social, increase exercise a bit
    digital_reset = base_features.copy()
    digital_reset["screen"] = max(1.0, digital_reset["screen"] - 2.0)
    digital_reset["social"] = max(5.0, digital_reset["social"] - 60.0)
    digital_reset["unlocks"] = max(10.0, digital_reset["unlocks"] - 40.0)
    digital_reset["exercise"] = min(200.0, digital_reset["exercise"] + 20.0)
    risk_digital = compute_risk_from_features(digital_reset)

    # Sleep protocol: +1.5h sleep, -1 stress, -0.5 anxiety
    sleep_reset = base_features.copy()
    sleep_reset["sleep"] = min(12.0, sleep_reset["sleep"] + 1.5)
    sleep_reset["stress"] = max(1.0, sleep_reset["stress"] - 1.0)
    sleep_reset["anxiety"] = max(1.0, sleep_reset["anxiety"] - 0.5)
    risk_sleep = compute_risk_from_features(sleep_reset)

    # Holistic plan: combine modest improvements
    holistic = base_features.copy()
    holistic["screen"] = max(1.0, holistic["screen"] - 1.5)
    holistic["sleep"] = min(12.0, holistic["sleep"] + 1.0)
    holistic["exercise"] = min(200.0, holistic["exercise"] + 25.0)
    holistic["outdoor"] = min(280.0, holistic["outdoor"] + 30.0)
    holistic["loneliness"] = max(1.0, holistic["loneliness"] - 1.0)
    holistic["social_support"] = min(10.0, holistic["social_support"] + 1.0)
    risk_holistic = compute_risk_from_features(holistic)

    col1, col2, col3 = st.columns(3)

    def render_intervention_card(title: str, risk_new: float):
        delta = risk_new - scenario_risk
        color = (
            "#10b981"
            if delta < 0
            else "#f59e0b"
            if abs(delta) < 0.01
            else "#ef4444"
        )
        sign = "+" if delta >= 0 else "−"
        st.markdown(
            f"""
        <div class='metric-card'>
            <div class='metric-label'>{title}</div>
            <div class='metric-value' style='font-size: 2.1rem; color: {color};'>
                {risk_new:.3f}
            </div>
            <div class='metric-change'>
                Δ risk: {sign}{abs(delta):.3f}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col1:
        render_intervention_card(
            "Digital Reset (−2h screen, −60min social)", risk_digital
        )
    with col2:
        render_intervention_card("Sleep Protocol (+1.5h sleep, ↓ stress)", risk_sleep)
    with col3:
        render_intervention_card(
            "Holistic Plan (screen, sleep, exercise, social)", risk_holistic
        )

    # Sensitivity curve: risk vs screen time
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📈 Sensitivity: Risk vs Screen Time (other factors fixed)")

    screen_min = max(1.0, sim_screen - 3.0)
    screen_max = min(18.0, sim_screen + 3.0)
    screen_grid = np.linspace(screen_min, screen_max, 40)
    risk_grid = []

    for s_val in screen_grid:
        tmp = base_features.copy()
        tmp["screen"] = float(s_val)
        risk_grid.append(compute_risk_from_features(tmp))

    fig_sens = go.Figure()
    fig_sens.add_trace(
        go.Scatter(
            x=screen_grid,
            y=risk_grid,
            mode="lines",
            name="Risk vs Screen Time",
            line=dict(color="#3b82f6", width=4),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.25)",
        )
    )
    fig_sens.add_vline(
        x=sim_screen,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=3,
        annotation_text=f"Current: {sim_screen:.1f}h",
        annotation_position="top right",
    )
    fig_sens.update_layout(
        height=430,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        xaxis_title="Screen Time (hours/day)",
        yaxis_title="Predicted Risk",
    )
    fig_sens.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.14)")
    fig_sens.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.14)")
    st.plotly_chart(fig_sens, use_container_width=True)

    # Recommendations
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 💡 AI-Generated Recommendations")

    recommendations = []

    if sim_screen > 10:
        recommendations.append(
            (
                "danger",
                "🔴 Screen time is well above healthy limits. Introduce strict boundaries, offline blocks, and app usage limits.",
            )
        )
    elif sim_screen > 8:
        recommendations.append(
            (
                "warning",
                "🟡 High digital exposure detected. Add app time caps and at least one daily offline block of 60–90 minutes.",
            )
        )

    if sim_sleep < 6:
        recommendations.append(
            (
                "danger",
                "🔴 Severe sleep deficit. Establish a consistent sleep schedule, limit screens before bedtime, and consider medical follow-up if problems persist.",
            )
        )
    elif sim_sleep < 7:
        recommendations.append(
            (
                "warning",
                "🟡 Sleep is below the recommended range. Aim for 30–60 additional minutes for the next two weeks and track consistency.",
            )
        )

    if sim_stress > 7 or sim_anxiety > 7:
        recommendations.append(
            (
                "danger",
                "🔴 Elevated psychological stress and anxiety. Structured support (for example, counseling or therapy) may be beneficial.",
            )
        )

    if sim_exercise < 20:
        recommendations.append(
            (
                "warning",
                "🟡 Physical activity is low. Target at least 30 minutes of light-to-moderate movement on most days.",
            )
        )

    if sim_loneliness > 6:
        recommendations.append(
            (
                "warning",
                "🟡 Social isolation signals detected. Strengthening connections through community activities or regular check-ins may help.",
            )
        )

    if (
        sim_wellbeing > 7.5
        and sim_stress < 5.5
        and sim_exercise > 30
        and sim_sleep >= 7
    ):
        recommendations.append(
            (
                "success",
                "🟢 Overall balance looks healthy. Continue tracking digital habits and maintaining the current protective lifestyle factors.",
            )
        )

    if not recommendations:
        recommendations.append(
            (
                "success",
                "🟢 Metrics are balanced. Maintain current routines and review again after any major lifestyle changes.",
            )
        )

    for rec_type, rec_text in recommendations:
        bg_color = {
            "danger": "rgba(239,68,68,0.15)",
            "warning": "rgba(245,158,11,0.15)",
            "success": "rgba(16,185,129,0.15)",
        }[rec_type]
        border_color = {
            "danger": "#ef4444",
            "warning": "#f59e0b",
            "success": "#10b981",
        }[rec_type]
        st.markdown(
            f"""
        <div style='background: {bg_color}; padding: 15px; border-radius: 10px;
                    border-left: 4px solid {border_color}; margin: 8px 0;'>
            <p style='color: white; margin: 0; font-size: 1rem;'>{rec_text}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ========================================================
# TAB 6: Clinical Reports
# ========================================================
with tab6:
    st.markdown(
        """
    <div class='section-header'>
        <div class='section-icon'>🏥</div>
        <div>
            <div class='section-title'>Clinical Intelligence Reports</div>
            <div class='section-subtitle'>High-risk users and summary indicators</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Top risk users
    st.markdown("#### 🚨 Critical Risk Users - Priority List")

    if len(plot_df) > 0:
        top_risk = plot_df.nlargest(
            30,
            "risk_score",
        )[
            [
                "user_id",
                "age",
                "gender",
                "occupation",
                "risk_score",
                "risk_segment",
                "screen_hours",
                "sleep_hours",
                "stress",
                "anxiety",
                "depression",
                "wellbeing",
                "social_support",
                "loneliness",
                "last_active",
            ]
        ].copy()

        top_risk["risk_score"] = top_risk["risk_score"].round(3)
        top_risk["screen_hours"] = top_risk["screen_hours"].round(1)
        top_risk["sleep_hours"] = top_risk["sleep_hours"].round(1)
        top_risk["days_inactive"] = (
            pd.Timestamp.now() - top_risk["last_active"]
        ).dt.days

        st.dataframe(
            top_risk.drop(columns=["last_active"]),
            use_container_width=True,
            height=500,
        )

        # Export high-risk report
        if st.button(
            "📄 Export High-Risk Report (CSV)", use_container_width=False
        ):
            csv_data = top_risk.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Report",
                data=csv_data,
                file_name=f"high_risk_users_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
    else:
        st.info("No users in current filtered view.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Summary statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Risk Distribution Summary")
        if len(plot_df) > 0:
            risk_summary = (
                plot_df["risk_segment"]
                .value_counts()
                .rename_axis("risk_segment")
                .to_frame("count")
            )
            risk_summary["percentage"] = (
                risk_summary["count"] / len(plot_df) * 100
            ).round(1)
            st.dataframe(risk_summary, use_container_width=True)
        else:
            st.write("No data available for summary.")

    with col2:
        st.markdown("#### Mental Health Averages")
        if len(plot_df) > 0:
            mental_health_avg = (
                plot_df[
                    [
                        "stress",
                        "anxiety",
                        "depression",
                        "wellbeing",
                        "mood",
                        "energy",
                    ]
                ]
                .mean()
                .round(2)
                .to_frame(name="Average Score")
            )
            st.dataframe(mental_health_avg, use_container_width=True)
        else:
            st.write("No data available for summary.")

    with col3:
        st.markdown("#### Behavioral Metrics")
        if len(plot_df) > 0:
            behavior_avg = (
                plot_df[
                    [
                        "screen_hours",
                        "phone_unlocks",
                        "notifications",
                        "social_minutes",
                        "gaming_minutes",
                        "exercise_minutes",
                        "outdoor_time",
                        "steps_daily",
                    ]
                ]
                .mean()
                .round(2)
                .to_frame(name="Average Value")
            )
            st.dataframe(behavior_avg, use_container_width=True)
        else:
            st.write("No data available for summary.")
