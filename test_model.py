"""
AI Child Growth Advisor — Comprehensive Test Suite (1–5 Years)
Tests PyTorch GrowthNet 2.0 model, WHO percentiles, clinical test cases, and AI Meal Planner.
"""

import os
import sys
import json
import streamlit as st

# Setup mock session state for headless test suite
if "current_user" not in st.session_state:
    st.session_state.current_user = {
        "username": "tester",
        "fullName": "System Test Runner",
        "role": "Teacher",
        "childId": "",
        "email": "test@domain.com"
    }

import torch
import joblib
import pandas as pd
import numpy as np

from app import (
    GrowthNet,
    load_model_and_scaler,
    generate_report,
    create_pdf_report,
    HFA_BOYS_FILE,
    HFA_GIRLS_FILE,
    WFA_BOYS_FILE,
    WFA_GIRLS_FILE,
    WFH_BOYS_FILE,
    WFH_GIRLS_FILE,
    MODEL_PATH,
    SCALER_PATH,
    PARAMS_PATH,
    CLASS_LABELS,
    load_students_data,
    load_food_recommendations,
    load_meal_schedule
)
from meal_planner import (
    generate_custom_meal_plan,
    create_meal_plan_pdf,
    RDA_TABLE
)
from who_standards import (
    calc_hfa_percentile,
    calc_bfa_percentile,
    classify_pediatric_status
)

def run_tests():
    print("=" * 70)
    print("STARTING AI CHILD GROWTH MODEL 2.0 COMPREHENSIVE TEST SUITE (1-5 YEARS)")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. FILE & ASSET INTEGRITY TESTS
    # -------------------------------------------------------------
    print("\n[TEST 1] Verifying Data & Model Files...")
    required_files = [
        HFA_BOYS_FILE, HFA_GIRLS_FILE, WFA_BOYS_FILE, WFA_GIRLS_FILE,
        WFH_BOYS_FILE, WFH_GIRLS_FILE,
        MODEL_PATH, SCALER_PATH, PARAMS_PATH, "Child_Attendance_Data(1).xlsx",
        "Food+Recommendations.csv", "MealSchedule.csv"
    ]
    for rf in required_files:
        assert os.path.exists(rf), f"Missing required file: {rf}"
        print(f"  [OK] Found {rf} ({os.path.getsize(rf)} bytes)")
    print(">>> Test 1 Passed: All model and WHO dataset files verified.")

    # -------------------------------------------------------------
    # 2. MODEL & SCALER LOADING TEST
    # -------------------------------------------------------------
    print("\n[TEST 2] Loading PyTorch Model & Feature Scaler...")
    growth_model, model_scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH, PARAMS_PATH)
    assert growth_model is not None, "Failed to load GrowthNet PyTorch model"
    assert model_scaler is not None, "Failed to load Scaler"
    
    with open(PARAMS_PATH, 'r') as f:
        best_params = json.load(f)
    print(f"  [OK] Model Hyperparameters: {best_params}")
    print(">>> Test 2 Passed: PyTorch model & StandardScaler loaded successfully.")

    # -------------------------------------------------------------
    # 3. CLINICAL TEST CASES (1–5 YEARS / 12–60 MONTHS)
    # -------------------------------------------------------------
    print("\n[TEST 3] Running Clinical Pediatric Test Cases...")
    test_cases = [
        {
            "name": "Case 1: Typical Healthy 2-Year-Old Boy (24m)",
            "age_months": 24,
            "sex": "M",
            "height": 87.1,
            "weight": 12.2,
            "expected_hfa_range": (40, 60),
            "expected_wfh_range": (40, 60),
            "expected_category": "Healthy"
        },
        {
            "name": "Case 2: Underweight / Wasted 3-Year-Old Girl (36m)",
            "age_months": 36,
            "sex": "F",
            "height": 95.0,
            "weight": 9.5,
            "expected_hfa_range": (30, 60),
            "expected_wfh_range": (0, 10),
            "expected_category": "Underweight"
        },
        {
            "name": "Case 3: Childhood Obesity 4.5-Year-Old Boy (54m)",
            "age_months": 54,
            "sex": "M",
            "height": 105.0,
            "weight": 24.5,
            "expected_hfa_range": (30, 70),
            "expected_wfh_range": (95, 100),
            "expected_category": ["Overweight", "Obese"]
        },
        {
            "name": "Case 4: Stunted Height 2.5-Year-Old Boy (30m)",
            "age_months": 30,
            "sex": "M",
            "height": 81.0,
            "weight": 11.0,
            "expected_hfa_range": (0, 3.5),
            "expected_wfh_range": (20, 80),
            "expected_category": ["Stunted", "Stunted & Underweight"]
        },
        {
            "name": "Case 5: 14-Month Toddler Girl Transition (14m)",
            "age_months": 14,
            "sex": "F",
            "height": 76.4,
            "weight": 9.4,
            "expected_hfa_range": (40, 60),
            "expected_wfh_range": (40, 60),
            "expected_category": "Healthy"
        },
        {
            "name": "Case 6: Upper Boundary 5-Year-Old Healthy Girl (60m)",
            "age_months": 60,
            "sex": "F",
            "height": 109.4,
            "weight": 18.2,
            "expected_hfa_range": (40, 60),
            "expected_wfh_range": (40, 60),
            "expected_category": "Healthy"
        }
    ]

    for tc in test_cases:
        print(f"\n  >> Testing: {tc['name']}")
        res = generate_report(
            age_m=tc["age_months"],
            ht=tc["height"],
            wt=tc["weight"],
            sex=tc["sex"],
            model=growth_model,
            scaler=model_scaler
        )

        hfa_p = res["hfa_p"]
        wfh_p = res["wfh_p"]
        bmi = res["bmi"]
        status = res["ai_status"]
        confs = res["confidence"]

        print(f"     - Height-for-Age: P{hfa_p:.1f}")
        print(f"     - Weight-for-Height: P{wfh_p:.1f}")
        print(f"     - Calculated BMI: {bmi:.2f} kg/m^2")
        print(f"     - AI Predicted Status: {status} (Confidence: {confs*100:.1f}%)")

        min_hfa, max_hfa = tc["expected_hfa_range"]
        min_wfh, max_wfh = tc["expected_wfh_range"]
        assert min_hfa <= hfa_p <= max_hfa, f"HFA percentile {hfa_p} out of range ({min_hfa}, {max_hfa})"
        assert min_wfh <= wfh_p <= max_wfh, f"WFH percentile {wfh_p} out of range ({min_wfh}, {max_wfh})"

        if isinstance(tc["expected_category"], list):
            assert status in tc["expected_category"], f"Expected one of {tc['expected_category']}, got {status}"
        else:
            assert status == tc["expected_category"], f"Expected {tc['expected_category']}, got {status}"
        print(f"     [PASS] Clinical validation passed.")

    print("\n>>> Test 3 Passed: All 6 clinical test cases validated successfully.")

    # -------------------------------------------------------------
    # 4. AI MEAL PLANNER ENGINE & PDF TEST
    # -------------------------------------------------------------
    print("\n[TEST 4] Testing Interactive AI Meal Planner Engine & PDF...")
    
    # Test for 2-year old vegetarian (1-3y RDA)
    mp_1_3 = generate_custom_meal_plan(24, "M", "Vegetarian", "Healthy", "Moderate")
    assert mp_1_3["target_rda"]["calories"] == 1010, f"Expected 1010 kcal, got {mp_1_3['target_rda']['calories']}"
    assert len(mp_1_3["meals"]) == 5, f"Expected 5 meals, got {len(mp_1_3['meals'])}"
    print(f"  [OK] 1-3y Meal Plan: {mp_1_3['plan_totals']['calories']} kcal, {mp_1_3['plan_totals']['protein_g']}g Protein across 5 meals")

    # Test for 4-year old non-vegetarian underweight (3-5y RDA with booster)
    mp_3_5 = generate_custom_meal_plan(48, "F", "Non-Vegetarian", "Underweight", "Active")
    assert mp_3_5["target_rda"]["calories"] > 1400, "Underweight active preschooler should receive calorie boost"
    print(f"  [OK] 3-5y Underweight Meal Plan: {mp_3_5['target_rda']['calories']} kcal target")

    # Test PDF Generation for Meal Plan
    pdf_mp_buf = create_meal_plan_pdf("Aarav", mp_1_3)
    pdf_mp_bytes = pdf_mp_buf.getvalue()
    assert len(pdf_mp_bytes) > 2000, f"Meal Plan PDF too small ({len(pdf_mp_bytes)} bytes)"
    print(f"  [OK] Meal Plan PDF generated successfully ({len(pdf_mp_bytes)} bytes)")
    print(">>> Test 4 Passed: AI Meal Planner engine & PDF exporter verified.")

    # -------------------------------------------------------------
    # 5. GROWTH REPORT PDF GENERATION TEST
    # -------------------------------------------------------------
    print("\n[TEST 5] Testing End-to-End Growth PDF Report Generation...")
    sample_res = generate_report(24, 87.1, 12.2, "M", growth_model, model_scaler)
    pdf_buf = create_pdf_report("Test Child", 24, sample_res)
    pdf_bytes = pdf_buf.getvalue()
    assert len(pdf_bytes) > 5000, f"Generated PDF too small ({len(pdf_bytes)} bytes)"
    print(f"  [OK] Generated PDF report successfully ({len(pdf_bytes)} bytes)")
    print(">>> Test 5 Passed: Growth PDF Generation pipeline verified.")

    # -------------------------------------------------------------
    # 6. DATA TABLES & RECOMMENDATIONS INTEGRITY
    # -------------------------------------------------------------
    print("\n[TEST 6] Verifying Attendance, Food Recommendations, and Meal Schedule...")
    st_df = load_students_data()
    food_df = load_food_recommendations()
    meal_df = load_meal_schedule()

    assert len(st_df) == 30, f"Expected 30 students, got {len(st_df)}"
    assert not food_df.empty, "Food recommendations dataframe is empty"
    assert len(meal_df) >= 6, f"Expected at least 6 daily meal schedules, got {len(meal_df)}"
    print(f"  [OK] Enrolled Students: {len(st_df)} children")
    print(f"  [OK] Food Recommendation entries: {len(food_df)} records")
    print(f"  [OK] Meal Schedule: {len(meal_df)} days mapped")
    print(">>> Test 6 Passed: Data tables and recommendations verified.")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! AI CHILD GROWTH MODEL 2.0 & MEAL PLANNER OPERATIONAL.")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
