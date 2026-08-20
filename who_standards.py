"""
WHO Child Growth Standards & Pediatric Anthropometrics Engine (1–5 Years / 12–60 Months)
Reference standards: WHO Child Growth Standards (2006), CDC Pediatric Growth Charts, IAP Guidelines.
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm

# Standard WHO percentiles mapping
WHO_PCOLS = ["P01", "P1", "P3", "P5", "P10", "P15", "P25", "P50", "P75", "P85", "P90", "P95", "P97", "P99", "P999"]
WHO_PERC_VALS = [0.1, 1.0, 3.0, 5.0, 10.0, 15.0, 25.0, 50.0, 75.0, 85.0, 90.0, 95.0, 97.0, 99.0, 99.9]

# --- WHO Length/Height-for-Age (LHFA) LMS Data (12 to 60 Months) ---
# Format: month: (L, M, S)
LHFA_BOYS_LMS = {
    12: (1.0, 75.75, 0.0337), 13: (1.0, 76.92, 0.0338), 14: (1.0, 78.02, 0.0339),
    15: (1.0, 79.10, 0.0340), 16: (1.0, 80.15, 0.0341), 17: (1.0, 81.18, 0.0342),
    18: (1.0, 82.30, 0.0343), 19: (1.0, 83.20, 0.0344), 20: (1.0, 84.20, 0.0345),
    21: (1.0, 85.10, 0.0346), 22: (1.0, 86.00, 0.0347), 23: (1.0, 86.90, 0.0348),
    24: (1.0, 87.12, 0.0351), 25: (1.0, 87.97, 0.0354), 26: (1.0, 88.80, 0.0357),
    27: (1.0, 89.60, 0.0360), 28: (1.0, 90.40, 0.0363), 29: (1.0, 91.20, 0.0366),
    30: (1.0, 91.90, 0.0369), 31: (1.0, 92.70, 0.0372), 32: (1.0, 93.40, 0.0375),
    33: (1.0, 94.10, 0.0378), 34: (1.0, 94.80, 0.0381), 35: (1.0, 95.40, 0.0384),
    36: (1.0, 96.10, 0.0387), 37: (1.0, 96.70, 0.0390), 38: (1.0, 97.40, 0.0393),
    39: (1.0, 98.00, 0.0396), 40: (1.0, 98.60, 0.0399), 41: (1.0, 99.20, 0.0402),
    42: (1.0, 99.90, 0.0405), 43: (1.0, 100.4, 0.0408), 44: (1.0, 101.0, 0.0411),
    45: (1.0, 101.6, 0.0414), 46: (1.0, 102.2, 0.0417), 47: (1.0, 102.8, 0.0420),
    48: (1.0, 103.3, 0.0423), 49: (1.0, 103.9, 0.0426), 50: (1.0, 104.4, 0.0429),
    51: (1.0, 105.0, 0.0432), 52: (1.0, 105.6, 0.0435), 53: (1.0, 106.1, 0.0438),
    54: (1.0, 106.7, 0.0441), 55: (1.0, 107.2, 0.0444), 56: (1.0, 107.8, 0.0447),
    57: (1.0, 108.3, 0.0450), 58: (1.0, 108.9, 0.0453), 59: (1.0, 109.4, 0.0456),
    60: (1.0, 110.0, 0.0459)
}

LHFA_GIRLS_LMS = {
    12: (1.0, 74.02, 0.0352), 13: (1.0, 75.20, 0.0354), 14: (1.0, 76.40, 0.0356),
    15: (1.0, 77.50, 0.0358), 16: (1.0, 78.60, 0.0360), 17: (1.0, 79.70, 0.0362),
    18: (1.0, 80.70, 0.0364), 19: (1.0, 81.70, 0.0366), 20: (1.0, 82.70, 0.0368),
    21: (1.0, 83.70, 0.0370), 22: (1.0, 84.60, 0.0372), 23: (1.0, 85.50, 0.0374),
    24: (1.0, 85.72, 0.0376), 25: (1.0, 86.59, 0.0379), 26: (1.0, 87.40, 0.0382),
    27: (1.0, 88.30, 0.0385), 28: (1.0, 89.10, 0.0388), 29: (1.0, 89.90, 0.0391),
    30: (1.0, 90.70, 0.0394), 31: (1.0, 91.40, 0.0397), 32: (1.0, 92.20, 0.0400),
    33: (1.0, 92.90, 0.0403), 34: (1.0, 93.60, 0.0406), 35: (1.0, 94.40, 0.0409),
    36: (1.0, 95.10, 0.0412), 37: (1.0, 95.70, 0.0415), 38: (1.0, 96.40, 0.0418),
    39: (1.0, 97.10, 0.0421), 40: (1.0, 97.70, 0.0424), 41: (1.0, 98.40, 0.0427),
    42: (1.0, 99.00, 0.0430), 43: (1.0, 99.70, 0.0433), 44: (1.0, 100.3, 0.0436),
    45: (1.0, 100.9, 0.0439), 46: (1.0, 101.5, 0.0442), 47: (1.0, 102.1, 0.0445),
    48: (1.0, 102.7, 0.0448), 49: (1.0, 103.3, 0.0451), 50: (1.0, 103.9, 0.0454),
    51: (1.0, 104.5, 0.0457), 52: (1.0, 105.0, 0.0460), 53: (1.0, 105.6, 0.0463),
    54: (1.0, 106.2, 0.0466), 55: (1.0, 106.7, 0.0469), 56: (1.0, 107.3, 0.0472),
    57: (1.0, 107.8, 0.0475), 58: (1.0, 108.4, 0.0478), 59: (1.0, 108.9, 0.0481),
    60: (1.0, 109.4, 0.0484)
}

# --- WHO BMI-for-Age (BFA) LMS Data (12 to 60 Months) ---
BFA_BOYS_LMS = {
    12: (-0.0827, 16.84, 0.0768), 15: (-0.1171, 16.36, 0.0765), 18: (-0.1554, 16.03, 0.0768),
    21: (-0.1973, 15.80, 0.0776), 24: (-0.2423, 16.02, 0.0790), 27: (-0.2899, 15.82, 0.0805),
    30: (-0.3397, 15.65, 0.0823), 33: (-0.3912, 15.52, 0.0841), 36: (-0.4438, 15.42, 0.0861),
    42: (-0.5513, 15.28, 0.0902), 48: (-0.6601, 15.21, 0.0945), 54: (-0.7681, 15.21, 0.0988),
    60: (-0.8732, 15.25, 0.1030)
}

BFA_GIRLS_LMS = {
    12: (-0.2520, 16.42, 0.0818), 15: (-0.2882, 15.98, 0.0818), 18: (-0.3275, 15.67, 0.0823),
    21: (-0.3695, 15.45, 0.0833), 24: (-0.4138, 15.71, 0.0848), 27: (-0.4597, 15.52, 0.0865),
    30: (-0.5069, 15.37, 0.0884), 33: (-0.5549, 15.26, 0.0904), 36: (-0.6033, 15.18, 0.0925),
    42: (-0.6997, 15.08, 0.0968), 48: (-0.7937, 15.04, 0.1011), 54: (-0.8837, 15.06, 0.1054),
    60: (-0.9680, 15.13, 0.1096)
}

CLASS_LABELS = {
    0: "Underweight",
    1: "Healthy",
    2: "Overweight",
    3: "Obese",
    4: "Stunted",
    5: "Stunted & Underweight"
}

def lms_to_zscore(val: float, l: float, m: float, s: float) -> float:
    """Calculate Z-score using standard Box-Cox Cole & Green (LMS) formula."""
    if val <= 0 or m <= 0 or s <= 0:
        return 0.0
    if abs(l) < 0.01:
        return math.log(val / m) / s
    else:
        return ((val / m) ** l - 1.0) / (l * s)

def zscore_to_percentile(z: float) -> float:
    """Convert Z-score to percentile (0.1 to 99.9%)."""
    p = float(norm.cdf(z) * 100.0)
    return max(0.1, min(99.9, p))

def interpolate_lms(age_m: float, lms_table: dict):
    """Interpolate L, M, S parameters for exact fractional age in months."""
    months = sorted(lms_table.keys())
    if age_m <= months[0]:
        return lms_table[months[0]]
    if age_m >= months[-1]:
        return lms_table[months[-1]]
    
    idx = np.searchsorted(months, age_m, side="right")
    m0, m1 = months[idx - 1], months[idx]
    frac = (age_m - m0) / (m1 - m0)
    l0, mean0, s0 = lms_table[m0]
    l1, mean1, s1 = lms_table[m1]
    
    l_int = l0 + frac * (l1 - l0)
    m_int = mean0 + frac * (mean1 - mean0)
    s_int = s0 + frac * (s1 - s0)
    return (l_int, m_int, s_int)

def calc_hfa_percentile(age_m: float, height: float, sex: str) -> float:
    """Compute Height-for-Age percentile for child aged 12-60 months."""
    lms_table = LHFA_BOYS_LMS if sex in ["M", "Male", 1] else LHFA_GIRLS_LMS
    l, m, s = interpolate_lms(age_m, lms_table)
    z = lms_to_zscore(height, l, m, s)
    return zscore_to_percentile(z)

def calc_bfa_percentile(age_m: float, bmi: float, sex: str) -> float:
    """Compute BMI-for-Age percentile using pediatric WHO LMS curves (12-60 months)."""
    lms_table = BFA_BOYS_LMS if sex in ["M", "Male", 1] else BFA_GIRLS_LMS
    l, m, s = interpolate_lms(age_m, lms_table)
    z = lms_to_zscore(bmi, l, m, s)
    return zscore_to_percentile(z)

def classify_pediatric_status(hfa_p: float, wfa_p: float, wfh_p: float, bfa_p: float, bmi: float) -> int:
    """
    Standard Pediatric Diagnostic Classification for Children Aged 1-5 Years:
    - 0: Underweight / Wasted (WFH < 3rd percentile or BFA < 5th percentile)
    - 1: Healthy (Normal linear and ponderal growth)
    - 2: Overweight (WFH > 85th percentile or BFA > 85th percentile, but not obese)
    - 3: Obese (WFH > 97th percentile or BFA > 97th percentile)
    - 4: Stunted (HFA < 3rd percentile with normal weight-for-height)
    - 5: Stunted & Underweight (HFA < 3rd percentile and WFH < 3rd percentile)
    """
    is_wasted_or_underweight = (wfh_p < 3.0) or (bfa_p < 5.0)
    is_stunted = (hfa_p < 3.0)
    is_obese = (wfh_p > 97.0) or (bfa_p > 97.0)
    is_overweight = ((wfh_p > 85.0) or (bfa_p > 85.0)) and not is_obese

    if is_stunted and is_wasted_or_underweight:
        return 5  # Stunted & Underweight
    if is_stunted:
        return 4  # Stunted
    if is_obese:
        return 3  # Obese
    if is_overweight:
        return 2  # Overweight
    if is_wasted_or_underweight:
        return 0  # Underweight
    return 1  # Healthy
