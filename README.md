# AI Child Growth Advisor — Streamlit Web Application

This repository contains the complete Streamlit frontend web application for the **AI Child Growth Advisor** platform, replacing the Wix website (`https://vishalkannan070.wixstudio.com/my-site`).

## Live Apps & Links

- **Main Streamlit Frontend Application:** Rebuild of Wix site features including Children Directory, Attendance System, Food Recommendations, Weekly Meal Schedule, and Parent Messaging.
- **Deployed AI Growth Assessment App:** [`https://aigrowthchildtracker.streamlit.app/`](https://aigrowthchildtracker.streamlit.app/) (Handles WHO percentiles, PyTorch GrowthNet model predictions, PDF report generation with charts, Wix Media upload, and Gmail SMTP email delivery).

## Features Overview

1. **🏠 Home / Overview:**
   - Platform mission & key stats (30 Enrolled Children, WHO Growth Standards, 6-Day Meal Plans).
   - Direct launch CTA button redirecting to `https://aigrowthchildtracker.streamlit.app/`.

2. **👶 Children Directory:**
   - Enrolled students master list loaded from `Child_Attendance_Data(1).xlsx` (30 children, C001 - C030).
   - Search by Child ID, Name, Parent, or Location.
   - Student profile cards with quick AI assessment launcher buttons.

3. **📋 Daily Attendance System:**
   - Teacher attendance sheet for marking Present/Absent per student.
   - Real-time attendance rate metrics (% Present / % Absent).
   - Local JSON storage (`attendance_records.json`) and history log.

4. **🥗 Food Recommendations:**
   - Sourced from `Food+Recommendations.csv`.
   - Category filtering (*Status-Based* vs *Age-Based Guides*).
   - Detailed cards for Goals, Recommended Foods, How to Eat, Snack Ideas, Avoid List, Activity Tips, and Key Nutrients.

5. **📅 Weekly Meal Schedule:**
   - Sourced from `MealSchedule.csv`.
   - Monday–Saturday meal plans with Calorie counts (kcal), Protein (g), Carbs (g), Fat (g), and notes.

6. **🤖 AI Growth Model Hub:**
   - Explains PyTorch `GrowthNet` architecture and WHO percentile calculation engine.
   - Direct link buttons launching `https://aigrowthchildtracker.streamlit.app/`.

7. **💬 Parent-Teacher Messaging:**
   - Select child ID to view and record parent-teacher communication notes (saved to `messages.json`).

## How to Run Locally

```bash
# 1. Clone/Navigate to workspace
cd "c:\Users\ELCOT\Desktop\final stream"

# 2. Run with uv virtualenv
uv venv
uv pip install streamlit openpyxl pandas reportlab matplotlib torch scikit-learn optuna joblib

# 3. Launch Streamlit app
.venv\Scripts\streamlit run app.py
```
