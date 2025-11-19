# 🧬 Advanced AI Health Intelligence Platform

[![Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-FF4B4B)](https://streamlit.io/)\
[![Made with ❤️ by Tarek Masryo](https://img.shields.io/badge/Made%20by-Tarek%20Masryo-blue)](https://github.com/tarekmasryo)\
[![Data](https://img.shields.io/badge/Data-AI%20Generated-purple)](#)

---

## 📌 Overview

The **Advanced AI Health Intelligence Platform** is an interactive analytics app built with **Streamlit** and **Plotly** to:

- Monitor **digital wellbeing** and **mental health risk** at population level  
- Analyze **behavioral patterns** (screen time, sleep, stress, social usage, activity)  
- Run **what-if scenarios** with an interactive risk simulator  
- Produce **clinical-style summary reports** for high-risk cohorts  

Data is generated programmatically inside the app to enable safe experimentation and scenario design without handling real user records.

---

## 🔑 Key Modules

### 🎯 Executive Dashboard
- Core KPIs: active users, high-risk share, model AUC, behavioral averages  
- Risk score distribution with configurable threshold  
- 90-day trends for screen time, stress, wellbeing, sleep, engagement, high-risk counts  
- Demographic breakdowns by age group, gender, location, occupation  

### 🧠 Risk Analytics
- Correlation heatmap for key risk drivers  
- Risk distributions by segment (Low / Moderate / High)  
- Confusion matrix & ROC curve with AUC  
- Conceptual feature importance and focused scatter plots (screen vs sleep, stress vs wellbeing)  

### 📱 Behavioral Insights
- 24-hour circadian patterns (screen time, notifications, stress, energy)  
- App usage composition and digital interaction metrics across segments  
- Physical vs digital balance and quick indicators (sleep deficit, high stress, inactivity)  

### 🧪 Scenario Simulator
- Single-profile simulator with sliders for digital, health, physical, and social variables  
- Real-time risk score, risk segment, intervention flag, and population percentile  
- Scenario vs population comparison and radar profile view  
- Pre-defined intervention bundles (digital reset, sleep protocol, holistic plan) with AI-style recommendations  

### 🏥 Clinical Reports
- Ranked list of highest-risk users with key attributes  
- One-click CSV export of high-risk cohort  
- Summary tables for risk segments, mental health metrics, and behavioral metrics  

---

## 🧠 Data & Risk Engine

The platform constructs a rich data space including:

- Demographics, digital behavior, lifestyle, mental health scores, social factors, engagement signals  
- Time-based structures: 90-day history and 24-hour patterns  

Risk is computed via a **logistic risk function** over weighted combinations of:

- Screen exposure and digital intensity  
- Stress, anxiety, depression, wellbeing, mood, energy  
- Sleep, activity, outdoor time  
- Social support and loneliness  

This produces:

- `risk_score` ∈ (0, 1)  
- `high_risk` label  
- `risk_segment` ∈ {Low, Moderate, High}  

The design focuses on **interpretability and controllable experimentation** in digital wellbeing analytics. It does not replace professional clinical judgement.

---

## 🧩 Tech Stack

- **Python**
- **Streamlit** – application framework  
- **NumPy** – numerical computations & generators  
- **pandas** – data manipulation & aggregations  
- **Plotly** – interactive visualizations  
- **scikit-learn (metrics)** – AUC, PR, Brier score, ROC, confusion matrix  

---

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
streamlit run app.py
