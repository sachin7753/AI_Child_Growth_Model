# AI Child Growth Advisor — Complete Project Documentation

**Project Name:** AI Child Growth Advisor  
**GitHub Repository:** https://github.com/sachin7753/ai_growth_2.0  
**Deployed URL:** Streamlit Cloud (auto-deploys from GitHub `main` branch)  
**Wix Website:** Connected via Wix REST API (metaSiteId: `d0fa869a-cbdb-4975-8a29-d01f7a7c1245`)  
**Last Updated:** February 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [File & Folder Structure](#4-file--folder-structure)
5. [AI Model (GrowthNet)](#5-ai-model-growthnet)
6. [WHO Growth Standards](#6-who-growth-standards)
7. [Streamlit Web Application (app.py)](#7-streamlit-web-application-apppy)
8. [Model Training Pipeline (train.py)](#8-model-training-pipeline-trainpy)
9. [Wix Integration](#9-wix-integration)
10. [Wix CMS Collections](#10-wix-cms-collections)
11. [Wix Velo Page Code](#11-wix-velo-page-code)
12. [Email System](#12-email-system)
13. [Deployment Guide](#13-deployment-guide)
14. [Security & Secrets Management](#14-security--secrets-management)
15. [API Reference](#15-api-reference)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Project Overview

The **AI Child Growth Advisor** is an intelligent child health monitoring system for children aged 0–5 years. It uses WHO (World Health Organization) growth standards combined with a PyTorch neural network to assess a child's nutritional status and generate personalized reports.

### What It Does

1. **Takes child measurements** — Age, height, weight, sex
2. **Computes WHO percentiles** — Height-for-age (HFA) and Weight-for-height (WFH)
3. **Classifies growth status** — Using a trained AI model backed by WHO rule-based validation
4. **Generates a PDF report** — With charts, percentile data, and food recommendations
5. **Uploads the PDF to Wix** — Automatically to Wix Media Manager + CMS collection
6. **Emails the report** — To the parent via Gmail SMTP
7. **Provides food recommendations** — Stored in Wix CMS, displayed on the Wix website

### Classification Categories

| Class ID | Label        | Criteria                                  |
|----------|--------------|-------------------------------------------|
| 0        | Underweight  | Weight-for-height percentile < 3          |
| 1        | Healthy      | Normal range, no flags                    |
| 2        | Overweight   | WFH percentile > 85 or BMI ≥ 25          |
| 3        | Obese        | WFH percentile > 85 and BMI ≥ 30         |
| 4        | Stunted      | Height-for-age percentile < 3             |
| 5        | Normal Ht    | Height within normal range (fallback)     |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    USER / TEACHER                    │
│           (Browser — Streamlit Cloud)                │
└─────────────┬───────────────────────┬───────────────┘
              │                       │
              ▼                       ▼
┌─────────────────────┐    ┌─────────────────────────┐
│   Streamlit App     │    │   Wix Website            │
│   (app.py)          │    │   (Wix Editor + Velo)    │
│                     │    │                          │
│ • Input form        │    │ • Student List           │
│ • AI prediction     │    │ • Attendance Page        │
│ • PDF generation    │    │ • Food Recommendations   │
│ • WHO percentiles   │    │ • Child Reports (CMS)    │
│ • Email sending     │    │ • Meal Schedule           │
│ • Wix API upload    │    │ • Messages               │
└────────┬────────────┘    └────────┬──────────────────┘
         │                          │
         │    Wix REST API          │
         ├──────────────────────────┤
         │                          │
         ▼                          ▼
┌──────────────────────────────────────────────────────┐
│                   Wix CMS Database                    │
│                                                       │
│  Collections:                                         │
│  • Import1 (Students) — 30 students                   │
│  • AttendanceRecords — daily attendance               │
│  • ChildReports — AI-generated PDF reports            │
│  • FoodRecommendations — nutrition guidance           │
│  • MealSchedule — weekly meal plans                   │
│  • Messages — parent-teacher messages                 │
│  • UsersDetails — registered users                    │
└──────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│   Gmail SMTP         │
│   (smtp.gmail.com)   │
│   App Password auth  │
└──────────────────────┘
```

---

## 3. Technology Stack

### Backend / AI

| Technology | Version | Purpose                                     |
|------------|---------|---------------------------------------------|
| Python     | 3.x     | Primary language                            |
| PyTorch    | Latest  | Neural network (GrowthNet model)            |
| Optuna     | Latest  | Hyperparameter optimization (80 trials)     |
| scikit-learn| Latest | StandardScaler, train/test split, metrics   |
| pandas     | Latest  | Data manipulation, WHO table loading        |
| numpy      | Latest  | Numerical computation                       |

### Web Application

| Technology   | Purpose                                       |
|--------------|-----------------------------------------------|
| Streamlit    | Web UI framework                              |
| Streamlit Cloud | Hosting (auto-deploys from GitHub)         |
| ReportLab    | PDF generation with charts                    |
| Matplotlib   | Growth charts in PDF                          |

### External Services

| Service        | Purpose                                    |
|----------------|---------------------------------------------|
| Wix REST API   | Media upload, CMS data management           |
| Gmail SMTP     | Email delivery with app password            |
| GitHub         | Source code repository                      |

### Wix Frontend

| Technology | Purpose                                       |
|------------|-----------------------------------------------|
| Wix Editor | Visual page builder                           |
| Wix Velo   | JavaScript frontend code for dynamic pages    |
| Wix CMS    | Database collections for all data             |

---

## 4. File & Folder Structure

```
AI_Growth_Report-main/
│
├── app.py                          # Main Streamlit web application (507 lines)
├── train.py                        # Model training pipeline (363 lines)
├── test_model.py                   # Model validation tests (19 WHO test cases)
├── wix_food_setup.py               # Script to populate Wix Food Recommendations
├── check_students.py               # Utility: query Import1 students from Wix
├── check_collections.py            # Utility: list all Wix CMS collections
│
├── growth_model.pth                # Trained PyTorch model weights
├── scaler.joblib                   # Fitted StandardScaler for input normalization
├── best_params.json                # Best hyperparameters from Optuna
│
├── tab_hfa_boys_p_0_5.xlsx         # WHO Height-for-age reference (boys, 24-60 months)
├── tab_hfa_girls_p_0_5.xlsx        # WHO Height-for-age reference (girls, 24-60 months)
├── tab_wfh_boys_p_0_5.xlsx         # WHO Weight-for-height reference (boys, 65-120 cm)
├── tab_wfh_girls_p_0_5.xlsx        # WHO Weight-for-height reference (girls, 65-120 cm)
├── tab_wfa_boys_p_0_5.xlsx         # WHO Weight-for-age reference (boys, 0-60 months)
├── tab_wfa_girls_p_0_5.xlsx        # WHO Weight-for-age reference (girls, 0-60 months)
├── tab_lhfa_boys_p_2_5.xlsx        # WHO Length/height-for-age (boys, 2-5 years)
├── tab_lhfa_girls_p_2_5.xlsx       # WHO Length/height-for-age (girls, 2-5 years)
├── bmi.csv.xlsx                    # Additional BMI reference data
│
├── confusion_matrix.png            # Model evaluation confusion matrix
├── requirements.txt                # Python package dependencies
├── service_account.json            # Google service account (legacy, not used)
│
├── .streamlit/
│   └── secrets.toml                # Wix API key, site ID, email credentials
│
├── .gitignore                      # Protects secrets from being committed
├── .gitattributes                  # Git configuration
│
├── temp_reports/                   # Temporary PDF storage (gitignored)
└── __pycache__/                    # Python cache (gitignored)
```

---

## 5. AI Model (GrowthNet)

### Architecture

The model is a fully-connected (MLP) neural network built with PyTorch:

```
Input (4 features)
    │
    ▼
Linear(4 → 185) → ReLU → Dropout(0.06)      ← Layer 1
    │
    ▼
Linear(185 → 185) → ReLU → Dropout(0.06)    ← Layer 2
    │
    ▼
Linear(185 → 185) → ReLU → Dropout(0.06)    ← Layer 3
    │
    ▼
Linear(185 → 185) → ReLU → Dropout(0.06)    ← Layer 4
    │
    ▼
Linear(185 → 6) → Softmax                   ← Output Layer
    │
    ▼
Output: 6 class probabilities
```

### Model Specifications

| Parameter        | Value                                    |
|------------------|------------------------------------------|
| Input Features   | 4 (age_months, height_cm, weight_kg, sex)|
| Output Classes   | 6 (Underweight, Healthy, Overweight, Obese, Stunted, Normal Ht) |
| Hidden Layers    | 4                                        |
| Units per Layer  | 185                                      |
| Dropout Rate     | 0.06 (6%)                                |
| Learning Rate    | 0.00416                                  |
| Total Parameters | 105,271                                  |
| Optimizer        | Adam                                     |
| Loss Function    | CrossEntropyLoss (class-weighted)        |

### Input Features

| Feature | Type    | Range         | Encoding         |
|---------|---------|---------------|-------------------|
| age     | Float   | 0–60 months   | StandardScaler    |
| height  | Float   | 40–130 cm     | StandardScaler    |
| weight  | Float   | 1–40 kg       | StandardScaler    |
| sex     | Integer | 0 or 1        | 1=Male, 0=Female  |

### Performance Metrics

| Metric      | Score |
|-------------|-------|
| Accuracy    | 88%   |
| Macro Avg F1| 0.88  |

**Per-class performance:**

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Underweight | 0.79      | 0.86   | 0.82     | 1,948   |
| Healthy     | 0.88      | 0.78   | 0.83     | 3,247   |
| Overweight  | 0.90      | 0.93   | 0.91     | 2,525   |
| Obese       | 0.99      | 1.00   | 0.99     | 1,948   |
| Stunted     | 0.85      | 0.88   | 0.86     | 1,949   |

**WHO Test Case Accuracy:** 19/19 (100%) — all WHO-based reference cases classified correctly.

### Hybrid Prediction System

The model's prediction is combined with WHO rule-based overrides for maximum accuracy:

```python
# Step 1: Neural network predicts class
status = model_prediction  

# Step 2: WHO rule-based overrides (in order of priority)
if WFH_percentile < 3:         → status = "Underweight"
elif WFH_percentile > 85:      → status = "Obese" (if BMI ≥ 30) or "Overweight"
elif BMI ≥ 30:                 → status = "Obese"
elif BMI ≥ 25:                 → status = "Overweight"
elif HFA_percentile < 3:       → status = "Stunted"
```

This hybrid approach ensures clinically accurate results even if the neural network makes an error.

---

## 6. WHO Growth Standards

### Reference Tables Used

| File                      | Type              | Primary Key | Rows | Range           |
|---------------------------|-------------------|-------------|------|-----------------|
| tab_hfa_boys_p_0_5.xlsx   | Height-for-age    | Month       | 37   | 24–60 months    |
| tab_hfa_girls_p_0_5.xlsx  | Height-for-age    | Month       | 37   | 24–60 months    |
| tab_wfh_boys_p_0_5.xlsx   | Weight-for-height | Height (cm) | 111  | 65–120 cm       |
| tab_wfh_girls_p_0_5.xlsx  | Weight-for-height | Height (cm) | 111  | 65–120 cm       |
| tab_wfa_boys_p_0_5.xlsx   | Weight-for-age    | Month       | 61   | 0–60 months     |
| tab_wfa_girls_p_0_5.xlsx  | Weight-for-age    | Month       | 61   | 0–60 months     |

### Table Columns

Each WHO table contains:
- **L, M, S** — Lambda, Mu, Sigma parameters for LMS method
- **P01, P1, P3, P5, P10, P15, P25, P50, P75, P85, P90, P95, P97, P99, P999** — Percentile values

### Percentile Computation

The system uses **linear interpolation** between percentile curves:

1. Find the child's age/height in the reference table
2. Interpolate between the two nearest rows
3. Map the child's measurement to the corresponding percentile
4. Use the percentile for classification

```python
# Example: A 36-month boy at 96.1 cm tall, 14.3 kg
# → HFA percentile: ~P50 (normal height)
# → WFH percentile: ~P50 (normal weight for height)
# → Classification: "Healthy"
```

---

## 7. Streamlit Web Application (app.py)

### User Interface

The app has a **sidebar** for inputs and a **main area** for results:

**Sidebar Inputs:**
| Field         | Type           | Range/Default    |
|---------------|----------------|-------------------|
| Child's Name  | Text           | "John Doe"        |
| Sex           | Radio (M/F)    | Male              |
| Age (Years)   | Number (float) | 0.0–5.0, step 0.1|
| Height (cm)   | Number (float) | 40.0–130.0        |
| Weight (kg)   | Number (float) | 1.0–40.0          |
| Child ID      | Text           | For Wix CMS link  |
| Parent Email  | Text           | For email delivery|

**Output (Main Area):**
- Growth status with confidence score
- Height-for-age percentile
- Weight-for-height percentile
- BMI value
- WHO assessment (wasting risk, stunting risk, or healthy)
- AI recommendations (food & lifestyle)
- Downloadable PDF report
- Auto-upload to Wix Media Manager
- Auto-save to Wix CMS collection
- Email sent to parent

### Function Flow

```
User clicks "Generate & Send Report"
    │
    ├── generate_report()
    │   ├── load_ref()          — Load WHO tables
    │   ├── interp_curve()      — Interpolate percentile curves
    │   ├── est_percentile()    — Calculate HFA & WFH percentiles
    │   ├── ai_predict()        — Neural network + WHO rule overrides
    │   └── get_ai_recommendations() — Status-specific advice
    │
    ├── create_pdf_report()     — Generate PDF with charts
    │
    ├── upload_to_wix_media()   — Upload PDF to Wix Media Manager
    │   ├── POST /site-media/v1/files/generate-upload-url
    │   └── POST (multipart file upload to generated URL)
    │
    ├── save_to_wix_collection() — Insert record in ChildReports CMS
    │   └── POST /wix-data/v2/items
    │
    └── send_email_report()     — Email PDF via Gmail SMTP
        └── SMTP SSL to smtp.gmail.com:465
```

### PDF Report Contents

The generated PDF includes:
1. **Page 1:**
   - Child's name and age
   - Height percentile, Weight-for-height percentile, BMI
   - WHO Assessment (wasting/stunting risk with color coding)
   - AI Recommendations (status-specific food and lifestyle advice)
2. **Page 2:**
   - Height-for-age percentile chart (matplotlib graph)
   - Weight-for-height percentile chart (matplotlib graph)

---

## 8. Model Training Pipeline (train.py)

### Training Configuration

| Parameter       | Value |
|-----------------|-------|
| Epochs          | 300   |
| Patience        | 25    |
| Optuna Trials   | 80    |
| Train/Val/Test  | 60/20/20 split |
| Seed            | 42    |

### Dataset Generation (build_dataset)

The training data is **synthetically generated** from WHO reference tables:

#### Part 1: Ages 24–60 months (full WHO coverage)
- For each (age, sex) in HFA table:
  - For each height percentile (P01–P999):
    - Look up the height value from HFA table
    - For each weight percentile in the WFH table at that height:
      - Compute BMI
      - Classify using `classify_child()` (same rules as app.py)
      - Add jittered copies (±1.2%)
      - More copies (4x) for minority classes (Underweight, Stunted)

#### Part 2: Ages 0–23 months (infant estimates)
- For each (age, sex) in WFA table:
  - Estimate height using WHO-aligned growth formula
  - Try 6 height multipliers (0.90–1.06) to cover stunted → tall
  - For each weight percentile:
    - Compute WFH percentile if height is in range
    - Classify and add jittered sample

#### Oversampling
- Minority classes are oversampled to reach at least 60% of the majority class count

### Final Dataset Statistics

| Class       | Samples |
|-------------|---------|
| Underweight | 9,741   |
| Healthy     | 16,235  |
| Overweight  | 12,626  |
| Obese       | 9,741   |
| Stunted     | 9,741   |
| **Total**   | **58,084** |

### Optuna Hyperparameter Search

| Hyperparameter | Search Range     | Best Value |
|----------------|------------------|------------|
| n_layers       | 2–4              | 4          |
| n_units        | 48–192 (log)     | 185        |
| dropout_rate   | 0.05–0.45        | 0.06       |
| learning_rate  | 5e-5 – 5e-3 (log)| 0.00416   |

### Training Output Files

| File             | Contents                           |
|------------------|------------------------------------|
| growth_model.pth | PyTorch state dict (trained weights)|
| scaler.joblib    | Fitted StandardScaler object       |
| best_params.json | Optuna best hyperparameters        |
| confusion_matrix.png | Evaluation confusion matrix     |

---

## 9. Wix Integration

### API Configuration

| Setting         | Value                                          |
|-----------------|------------------------------------------------|
| API Key         | Stored in `.streamlit/secrets.toml` → `[wix].api_key` |
| Site ID         | `d0fa869a-cbdb-4975-8a29-d01f7a7c1245` (metaSiteId) |
| Base URL        | `https://www.wixapis.com`                      |

**Important Notes:**
- The `wix-site-id` header must use the **metaSiteId**, NOT the dashboard site ID
- The `wix-account-id` header must be **OMITTED** (causes 403 errors if included)
- File uploads use **multipart POST** format

### API Endpoints Used

#### 1. Upload PDF to Media Manager
```
POST https://www.wixapis.com/site-media/v1/files/generate-upload-url
Headers:
  Authorization: {API_KEY}
  wix-site-id: {META_SITE_ID}
  Content-Type: application/json
Body:
  {"mimeType": "application/pdf", "fileName": "..."}

→ Returns: {"uploadUrl": "https://upload.wixmp.com/..."}

POST {uploadUrl}
  Content-Type: multipart/form-data
  file: (binary PDF data)

→ Returns: file info with URL
```

#### 2. Insert CMS Data Item
```
POST https://www.wixapis.com/wix-data/v2/items
Headers:
  Authorization: {API_KEY}
  wix-site-id: {META_SITE_ID}
  Content-Type: application/json
Body:
  {
    "dataCollectionId": "ChildReports",
    "dataItem": {
      "data": {
        "title": "...",
        "childId": "...",
        "reportFileUrl": "..."
      }
    }
  }
```

#### 3. Query CMS Items
```
POST https://www.wixapis.com/wix-data/v2/items/query
Body:
  {
    "dataCollectionId": "Import1",
    "query": {"paging": {"limit": 100}}
  }
```

---

## 10. Wix CMS Collections

### 10.1 Import1 (Students)

**Purpose:** Master list of all enrolled children (30 students).

| Field        | Type   | Description                   |
|--------------|--------|-------------------------------|
| childId      | TEXT   | Unique ID (C001, C002, ...)   |
| childName    | TEXT   | Child's full name             |
| parentName   | TEXT   | Parent's full name            |
| place        | TEXT   | Location/address              |
| phoneNumber  | NUMBER | Parent's phone number         |
| attendance   | TEXT   | Attendance status (legacy)    |
| title        | TEXT   | Title field                   |

**Students (30 total):** Arun (C001), Kavya (C002), Prasad (C003), Keerthana (C004), Karthik (C005), Nirupa (C006), Harish (C007), Divya (C008), Vijay (C009), Nandhini (C010), Varun (C011), Lavanya (C012), Suresh (C013), Meena (C014), Rohit (C015), Janaki (C016), Aditya (C017), Hema (C018), Mohan (C019), Devika (C020), Srinivas (C021), Abirami (C022), Aravind (C023), Mathavi (C024), Dinesh (C025), Pavithra (C026), Yuvaraj (C027), Harini (C028), Santhosh (C029), Anitha (C030).

---

### 10.2 AttendanceRecords

**Purpose:** Daily attendance tracking for each child.

| Field      | Type     | Description                     |
|------------|----------|---------------------------------|
| childId    | OBJECT   | Linked child's ID               |
| childName  | TEXT     | Child's name                    |
| date       | DATE     | Date of attendance              |
| status     | TEXT     | "Present" or "Absent"           |
| markedBy   | TEXT     | Teacher's email who marked it   |

---

### 10.3 ChildReports

**Purpose:** AI-generated growth reports (auto-inserted by Streamlit app).

| Field          | Type | Description                       |
|----------------|------|-----------------------------------|
| title          | TEXT | "{Child Name} Growth Report - {timestamp}" |
| childId        | TEXT | Child's unique ID                 |
| reportFileUrl  | URL  | Wix Media URL of the PDF          |

---

### 10.4 FoodRecommendations

**Purpose:** Nutrition guidance displayed on the Wix Food Recommendations page.

| Field            | Type   | Description                        |
|------------------|--------|------------------------------------|
| status           | TEXT   | Status name or age range           |
| goal             | TEXT   | Nutritional goal                   |
| recommendedFoods | TEXT   | List of recommended foods          |
| mealPattern      | TEXT   | Meal frequency/pattern             |
| snackIdeas       | TEXT   | Healthy snack suggestions          |
| avoid            | TEXT   | Foods to avoid                     |
| activityTip      | TEXT   | Activity and health tips           |
| keyNutrients     | TEXT   | Important nutrients to focus on    |
| category         | TEXT   | "Status-Based" or "Age-Based Guide"|
| sortOrder        | NUMBER | Display order (1-14)               |

**10 Items:**

| Status-Based (5)  | Age-Based Guide (5)  |
|--------------------|-----------------------|
| 1. Underweight     | 10. 0-6 Months       |
| 2. Healthy         | 11. 6-8 Months       |
| 3. Overweight      | 12. 9-12 Months      |
| 4. Obese           | 13. 1-2 Years        |
| 5. Stunted         | 14. 2-5 Years        |

---

### 10.5 MealSchedule

**Purpose:** Weekly meal plan with nutritional details.

| Field         | Type | Description              |
|---------------|------|--------------------------|
| day           | TEXT | Day of the week          |
| meal          | TEXT | Meal description         |
| caloriesKcal  | TEXT | Calorie count            |
| proteinG      | TEXT | Protein in grams         |
| carbsG        | TEXT | Carbohydrates in grams   |
| fatG          | TEXT | Fat in grams             |
| notes         | TEXT | Additional notes         |

---

### 10.6 Messages

**Purpose:** Parent-teacher communication system.

| Field     | Type | Description               |
|-----------|------|---------------------------|
| childId   | TEXT | Associated child ID       |
| from      | TEXT | Sender name               |
| message   | TEXT | Message content           |
| timestamp | DATE | When the message was sent |

---

### 10.7 UsersDetails

**Purpose:** Registered user profiles (parents/teachers).

| Field        | Type    | Description              |
|--------------|---------|--------------------------|
| fullName     | TEXT    | User's full name         |
| phoneNumber  | TEXT    | Contact phone            |
| status       | BOOLEAN | Active/inactive          |
| loginType    | TEXT    | Login method             |

---

### 10.8 Attendence (Legacy)

**Purpose:** Older attendance system (superseded by AttendanceRecords).

| Field     | Type      | Description            |
|-----------|-----------|------------------------|
| title     | TEXT      | Status label           |
| reference | REFERENCE | Link to Students       |
| date      | DATE      | Date                   |

---

## 11. Wix Velo Page Code

### 11.1 Attendance Page (Teacher Marking)

This page lets teachers mark Present/Absent for all students using radio buttons.

```javascript
import wixData from "wix-data";
import wixUsers from "wix-users";

$w.onReady(async function () {
    $w("#text9").hide();

    try {
        const studentResults = await wixData.query("Import1")
            .ascending("childName")
            .limit(1000)            // <-- IMPORTANT: default is only 50
            .find();

        // Auto-assign Child IDs (C001, C002, ...) if missing
        let highestNum = 0;
        for (const student of studentResults.items) {
            if (student.childId && /^C\d{3}$/.test(student.childId)) {
                const num = parseInt(student.childId.substring(1));
                if (num > highestNum) highestNum = num;
            }
        }
        for (let i = 0; i < studentResults.items.length; i++) {
            const student = studentResults.items[i];
            if (!student.childId || !/^C\d{3}$/.test(student.childId)) {
                highestNum++;
                student.childId = "C" + highestNum.toString().padStart(3, "0");
                await wixData.update("Import1", student);
            }
        }

        $w("#repeater1").data = [];
        $w("#repeater1").data = studentResults.items;
    } catch (err) {
        showMessage("❌ Error loading students.");
    }

    // Repeater: show name + radio button
    $w("#repeater1").onItemReady(($item, itemData) => {
        $item("#childName").text = `${itemData.childName} (${itemData.childId})`;
        $item("#checkboxPresent").value = "Absent";
    });

    // Submit attendance
    $w("#submitButton").onClick(async () => {
        showMessage("Saving attendance...");
        const currentUser = wixUsers.currentUser;
        const userEmail = await currentUser.getEmail();
        const today = new Date();
        today.setHours(0, 0, 0, 0);

        const studentIdsOnPage = $w("#repeater1").data.map(s => s.childId);
        const existingRecords = await wixData.query("AttendanceRecords")
            .ge("date", today)
            .hasSome("childId", studentIdsOnPage)
            .limit(1000)
            .find();

        const existingMap = new Map(existingRecords.items.map(r => [r.childId, r]));
        const toInsert = [], toUpdate = [];

        $w("#repeater1").forEachItem(($item, itemData) => {
            const selectedValue = $item("#checkboxPresent").value;
            const newStatus = selectedValue === "Present" ? "Present" : "Absent";

            if (existingMap.has(itemData.childId)) {
                const rec = existingMap.get(itemData.childId);
                rec.status = newStatus;
                rec.markedBy = userEmail;
                toUpdate.push(rec);
            } else {
                toInsert.push({
                    childId: itemData.childId,
                    childName: itemData.childName,
                    date: new Date(),
                    status: newStatus,
                    markedBy: userEmail
                });
            }
        });

        const promises = [];
        if (toInsert.length > 0) promises.push(wixData.bulkInsert("AttendanceRecords", toInsert));
        if (toUpdate.length > 0) promises.push(wixData.bulkUpdate("AttendanceRecords", toUpdate));
        await Promise.all(promises);

        showMessage(`✅ Attendance saved for ${studentIdsOnPage.length} students!`);
    });
});

function showMessage(msg) {
    $w("#text9").text = msg;
    $w("#text9").show();
    setTimeout(() => $w("#text9").hide(), 4000);
}
```

**Required Repeater Elements:**
| Element ID         | Type          | Purpose               |
|--------------------|---------------|------------------------|
| #repeater1         | Repeater      | Student list container |
| #childName         | Text          | Displays "Name (C001)"|
| #checkboxPresent   | Radio Buttons | Present / Absent       |
| #submitButton      | Button        | Save attendance        |
| #text9             | Text          | Status messages        |

**Known Issue:** The Wix repeater may show only 24 items if a Dataset element is connected in the editor. Solution: Delete the dataset element from the page, or set its "Number of items to display" to 50+.

---

### 11.2 Food Recommendations Page (Enhanced Status Filter)

This page shows comprehensive nutrition guidance based on the selected growth status, with structured "What to eat," "How to eat," meal ideas, and links.

```javascript
import wixData from 'wix-data';

$w.onReady(function () {
    hideAllDetails();

    $w("#statusFilter").onChange((event) => {
        const selected = event.target.value;
        if (!selected || selected === "All") {
            hideAllDetails();
            return;
        }

        wixData.query("FoodRecommendations")
            .eq("category", "Status-Based")
            .eq("status", selected)
            .find()
            .then((results) => {
                if (results.items.length > 0) {
                    const item = results.items[0];
                    
                    // Format the display with clear sections
                    $w("#goalText").text = "🎯 Goal:\n" + (item.goal || "");
                    
                    $w("#foodsText").text = "🍽️ RECOMMENDED FOODS:\n" + (item.recommendedFoods || "");
                    
                    // Split mealPattern to show "How to eat" and meal ideas separately
                    const mealInfo = item.mealPattern || "";
                    $w("#mealText").text = "🥄 " + mealInfo;
                    
                    $w("#snackText").text = "🥗 SNACK IDEAS:\n" + (item.snackIdeas || "");
                    
                    $w("#avoidText").text = "🚫 FOODS TO AVOID:\n" + (item.avoid || "");
                    
                    $w("#tipText").text = "💡 ACTIVITY & TIPS:\n" + (item.activityTip || "");
                    
                    $w("#nutrientsText").text = "⚡ KEY NUTRIENTS:\n" + (item.keyNutrients || "");
                    
                    showAllDetails();
                }
            });
    });
});

function hideAllDetails() {
    $w("#goalText").hide();
    $w("#foodsText").hide();
    $w("#mealText").hide();
    $w("#snackText").hide();
    $w("#avoidText").hide();
    $w("#tipText").hide();
    $w("#nutrientsText").hide();
}

function showAllDetails() {
    $w("#goalText").show("fade");
    $w("#foodsText").show("fade");
    $w("#mealText").show("fade");
    $w("#snackText").show("fade");
    $w("#avoidText").show("fade");
    $w("#tipText").show("fade");
    $w("#nutrientsText").show("fade");
}
```

**Data Structure in Wix CMS (FoodRecommendations collection):**

Each item now contains comprehensive guidance:
- **recommendedFoods** — Emoji-coded list of what to eat
- **mealPattern** — Contains both "HOW TO EAT" practical steps AND "MEAL IDEAS"
- **snackIdeas** — Quick snack options separated by pipes (|)
- **keyNutrients** — Important nutrients to focus on
- **activityTip** — Activity recommendations and health tips
- **avoid** — Foods to avoid
- **goal** — The nutritional objective for this status

**Example display for "Underweight" status:**

```
🎯 Goal:
Healthy weight gain through calorie-dense, nutrient-rich foods

🍽️ RECOMMENDED FOODS:
🥛 Whole milk | 🧀 Cheese & Paneer | 🥚 Eggs | 🥜 Peanut butter
🍌 Bananas | 🥑 Avocado | 🍠 Sweet potato | 🌰 Nuts (crushed)

🥄 HOW TO EAT:
Offer 5-6 small meals/snacks instead of 3 large meals.
Add one calorie booster per meal (ghee, nut powder, cheese).
Do not give excess water before meals.
Track weight monthly.

MEAL IDEAS:
Banana milkshake with full-fat milk | Dry fruits ladoo with ghee
Cheese toast with butter | Sooji halwa with ghee | Makhana with ghee

🥗 SNACK IDEAS:
Dry fruits ladoo | Banana milkshake | Cheese toast | Sooji halwa...

🚫 FOODS TO AVOID:
Junk food that fills stomach without nutrition; excessive water before meals

💡 ACTIVITY & TIPS:
Monitor weight monthly; consult pediatrician if no weight gain in 2 months

⚡ KEY NUTRIENTS:
Calories, healthy fats, protein, iron, calcium
```

**Required Page Elements:**
| Element ID      | Type          | Purpose                     |
|-----------------|---------------|-----------------------------|
| #statusFilter   | Radio Buttons | All, Underweight, Healthy, Overweight, Obese, Stunted |
| #goalText       | Text          | Nutritional goal            |
| #foodsText      | Text          | Recommended foods list      |
| #mealText       | Text          | Meal pattern                |
| #snackText      | Text          | Snack ideas                 |
| #avoidText      | Text          | Foods to avoid              |
| #tipText        | Text          | Activity/health tip         |
| #nutrientsText  | Text          | Key nutrients               |

All 7 text elements should have **"Hidden on Load"** set to ON in Properties.

---

## 12. Email System

### Configuration

| Setting       | Value                         |
|---------------|-------------------------------|
| SMTP Server   | smtp.gmail.com                |
| Port          | 465 (SSL)                     |
| Sender Email  | mystuntman009@gmail.com       |
| Auth Method   | Gmail App Password            |

### How It Works

1. After generating a report, the PDF is attached to an email
2. Email is sent via Gmail SMTP with SSL encryption
3. Uses a Gmail **App Password** (not the regular Gmail password)
4. App password is stored in `.streamlit/secrets.toml` under `[email]`

### Generating a Gmail App Password

1. Go to https://myaccount.google.com/apppasswords
2. Sign in with the Gmail account
3. Select "Mail" and "Windows Computer"
4. Click "Generate"
5. Copy the 16-character password
6. Add it to secrets.toml: `app_password = "xxxxxxxxxxxx"`

---

## 13. Deployment Guide

### Streamlit Cloud Deployment

1. **Push code to GitHub:**
   ```bash
   git add .
   git commit -m "Update"
   git push origin main
   ```

2. **Go to** https://share.streamlit.io

3. **Deploy from repo:**
   - Repository: `sachin7753/ai_growth_2.0`
   - Branch: `main`
   - Main file: `app.py`

4. **Add Secrets** (Settings → Secrets):
   ```toml
   [wix]
   api_key = "IST.eyJraWQ..."
   site_id = "d0fa869a-cbdb-4975-8a29-d01f7a7c1245"
   
   [email]
   user = "mystuntman009@gmail.com"
   app_password = "kkxcvbveuaxbenrx"
   ```

5. **Reboot** the app after adding secrets

### Local Development

```bash
# Clone the repo
git clone https://github.com/sachin7753/ai_growth_2.0.git
cd ai_growth_2.0

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create secrets file
mkdir .streamlit
# Add secrets.toml with API keys

# Run the app
streamlit run app.py

# Retrain model (optional)
python train.py
```

---

## 14. Security & Secrets Management

### Protected Files (.gitignore)

```
.streamlit/secrets.toml      # API keys, email passwords
service_account.json          # Google service account
__pycache__/                  # Python cache
temp_reports/                 # Temporary PDFs
*.pyc                         # Compiled Python
```

### Secrets Location

| Environment      | Location                             |
|------------------|--------------------------------------|
| Local Dev        | `.streamlit/secrets.toml`            |
| Streamlit Cloud  | Settings → Secrets (web UI)          |

### Secrets Format

```toml
[wix]
api_key = "IST.eyJraWQ..."              # Wix API key (starts with IST.)
site_id = "d0fa869a-cbdb-..."           # Wix metaSiteId (NOT dashboard ID)

[email]
user = "mystuntman009@gmail.com"        # Gmail sender address
app_password = "kkxcvbveuaxbenrx"       # 16-char Gmail app password
```

---

## 15. API Reference

### Wix API Headers (Required for ALL calls)

```
Authorization: IST.eyJraWQ...          (full API key)
wix-site-id: d0fa869a-cbdb-...         (metaSiteId)
Content-Type: application/json
```

**DO NOT include:** `wix-account-id` header (causes 403 Forbidden)

### Endpoints Summary

| Action                    | Method | Endpoint                                             |
|---------------------------|--------|------------------------------------------------------|
| List Collections          | GET    | `/wix-data/v2/collections`                           |
| Create Collection         | POST   | `/wix-data/v2/collections`                           |
| Query Items               | POST   | `/wix-data/v2/items/query`                           |
| Insert Item               | POST   | `/wix-data/v2/items`                                 |
| Update Item               | PATCH  | `/wix-data/v2/items/{id}`                            |
| Bulk Insert               | POST   | `/wix-data/v2/items/bulk-insert`                     |
| Generate Upload URL       | POST   | `/site-media/v1/files/generate-upload-url`           |
| Upload File               | POST   | `{generated uploadUrl}` (multipart)                  |

---

## 16. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Wix 404 error | Using dashboard site ID instead of metaSiteId | Use `d0fa869a-cbdb-4975-8a29-d01f7a7c1245` |
| Wix 403 error | Including `wix-account-id` header | Remove the header entirely |
| Gmail auth fail | Wrong app password or 2FA not enabled | Generate new app password at Google account settings |
| Model predicts wrong class | Old broken model | Run `python train.py` to retrain |
| Repeater shows 24 of 30 students | Dataset element capping display | Delete the Dataset element or set items to 50+ |
| Streamlit "missing secrets" | Secrets not configured on Cloud | Add secrets in Streamlit Cloud → Settings → Secrets |
| PDF not uploading | Wix API key expired | Generate new API key in Wix dashboard |
| Import error on Streamlit | Missing packages | Check `requirements.txt` includes all needed packages |

### Retraining the Model

If you need to retrain for any reason:

```bash
cd AI_Growth_Report-main
python train.py
# Takes ~2-6 hours (80 Optuna trials)
# Outputs: growth_model.pth, scaler.joblib, best_params.json
```

Then push to GitHub:
```bash
git add growth_model.pth scaler.joblib best_params.json train.py
git commit -m "Retrained model"
git push origin main
```

Streamlit Cloud will auto-redeploy.

### Validating the Model

```bash
python test_model.py
# Runs 19 WHO-based test cases
# Expected: 19/19 (100%) accuracy
```

---

*Document generated on February 21, 2026*  
*Project by Sachin — AI Child Growth Advisor v2.0*
