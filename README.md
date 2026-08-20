# AI Child Growth Advisor & Pediatric Health Portal

An intelligent pediatric health, growth screening, and nutritional care platform tailored for children aged **1 to 5 years (12–60 months)**.

---

## 👨‍💻 Project Development Team

- **SACHIN M** — *Lead AI Engineer & System Architect* (Deep Learning Model Optimization, GrowthNet 2.0 Neural Network, WHO LMS Standards Engine & Backend Systems)
- **VISHAL KANNAN S I** — *Lead Full-Stack Developer & UI/UX Architect* (Interactive AI Meal Planner, Clinical Pediatric Standards Integration, Streamlit Multi-Portal Architecture & UI/UX Design)

---

## 🌟 Platform Capabilities & Features

1. **🧠 PyTorch GrowthNet 2.0 AI Engine**:
   - Deep Multi-Layer Perceptron trained on 29,000+ clinical benchmark cases with 9 domain-engineered features (`[age_months, height, weight, sex, bmi, hfa_p, wfa_p, wfh_p, bfa_p]`).
   - Achieves **99.0% diagnostic test accuracy** across WHO growth classifications.

2. **📏 Multi-Axial WHO & CDC/IAP Pediatric Standards**:
   - Computes Box-Cox Cole & Green LMS Z-scores and exact percentiles for Height-for-Age (HFA), Weight-for-Age (WFA), Weight-for-Height (WFH), and BMI-for-Age (BFA).
   - Classifies 6 pediatric categories: *Underweight / Wasted*, *Healthy*, *Overweight*, *Obese*, *Stunted*, and *Stunted & Underweight*.

3. **🍽️ Interactive AI Daily Meal Planner**:
   - Grounded in **ICMR-NIN RDA 2024**, **AskNestle Meal Plan**, **USDA MyPlate**, and **Harvard Healthy Eating Plate**.
   - Generates age-calibrated 5-meal daily schedules (Breakfast, Mid-Morning Snack, Lunch, Evening Snack, Dinner) with macro/micronutrient breakdowns and Precision Nutrition hand-size portion guides (Palm, Fist, Cupped Hand, Thumb).
   - Generates downloadable printable Daily Meal Plan PDF charts.

4. **🎓 Student Database Management (CRUD)**:
   - Full database interface for teachers to **Add New Students**, **Edit / Update Records**, and **Delete Students** with real-time Excel (`Child_Attendance_Data(1).xlsx`) and JSON synchronization.

5. **📋 Multi-Day Cumulative Attendance System**:
   - Permanent attendance history tracking across all recorded school dates.
   - Computes individual student cumulative attendance percentages with color-coded badges (🟢 Excellent ≥90%, 🟡 Good 75–89%, 🔴 At Risk <75%).
   - Academic Year Attendance Reset feature for clean restarts.

6. **💬 Direct Parent-Teacher Communication**:
   - Real-time messaging stream between teachers and parents for individual progress notes and dietary coordination.

7. **📄 Automated PDF Growth & Nutrition Reports**:
   - Instant high-resolution ReportLab PDF generation featuring growth percentile curves, AI diagnostic summaries, and tailored clinical guidance.

---

## ⚠️ Official Medical & Legal Disclaimer

> **IMPORTANT NOTICE:**  
> The **AI Child Growth Advisor**, its neural network predictions, WHO percentile curves, dietary recommendations, and PDF growth charts are generated for **educational, nutritional guidance, and informational screening support only**.  
>  
> This platform **does not provide medical diagnosis, therapeutic prescription, or clinical treatment**. It is not a substitute for clinical judgment or professional medical consultation with a licensed pediatrician or healthcare professional. Always consult a qualified pediatrician for medical advice regarding a child's development, health conditions, or nutritional deficiencies.

---

## 🚀 How to Run Locally

```bash
# 1. Clone repository
git clone https://github.com/sachin7753/AI_Child_Growth_Model.git
cd AI_Child_Growth_Model

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Run Automated Comprehensive Test Suite
python test_model.py

# 4. Launch Streamlit Application
streamlit run app.py
```
