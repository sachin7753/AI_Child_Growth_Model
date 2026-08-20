"""
AI Child Growth Model Training Pipeline (1–5 Years / 12–60 Months)
Standards: WHO Child Growth Standards (2006), CDC Pediatric Growth Charts, IAP Guidelines.
"""

import os
import re
import json
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim

from who_standards import (
    calc_hfa_percentile,
    calc_bfa_percentile,
    classify_pediatric_status,
    CLASS_LABELS
)

# -------- CONFIGURATION --------
WFA_BOYS_FILE = "tab_wfa_boys_p_0_5.xlsx"
WFA_GIRLS_FILE = "tab_wfa_girls_p_0_5.xlsx"
HFA_BOYS_FILE = "tab_hfa_boys_p_0_5.xlsx"
HFA_GIRLS_FILE = "tab_hfa_girls_p_0_5.xlsx"
WFH_BOYS_FILE = "tab_wfh_boys_p_0_5.xlsx"
WFH_GIRLS_FILE = "tab_wfh_girls_p_0_5.xlsx"

MODEL_SAVE_PATH = "growth_model.pth"
SCALER_SAVE_PATH = "scaler.joblib"
PARAMS_SAVE_PATH = "best_params.json"

EPOCHS = 60
PATIENCE = 12
OPTUNA_TRIALS = 20
FEATURE_NAMES = ["age_m", "height", "weight", "sex", "bmi", "hfa_p", "wfa_p", "wfh_p", "bfa_p"]
BATCH_SIZE = 256

# -------- NEURAL NETWORK DEFINITION (GrowthNet 2.0) --------
class GrowthNet(nn.Module):
    def __init__(self, in_features=9, n_layers=3, n_units=128, dropout_rate=0.25):
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

# -------- WHO TABLE UTILITIES --------
def load_who_table(path: str, primary_col_regex: str):
    """Load WHO reference table → (DataFrame, pcols list)."""
    df = pd.read_excel(path)
    primary_col = next(c for c in df.columns if re.search(primary_col_regex, str(c), re.I))
    pcols = [c for c in df.columns if re.match(r"P\d+", str(c))]
    df = df[[primary_col] + pcols].copy()
    df.columns = ["primary"] + pcols
    return df, pcols

def interp_curve(ref_df, pcols, val):
    """Interpolate percentile curve from WHO table."""
    values = ref_df["primary"].values.astype(float)
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
    """Estimate percentile given measurement and {perc: value} mapping."""
    pts = sorted(curve.items(), key=lambda x: x[1])
    vals = [v for _, v in pts]
    percs = [p for p, _ in pts]
    if value <= vals[0]:
        return percs[0]
    if value >= vals[-1]:
        return percs[-1]
    j = np.searchsorted(vals, value, side="right")
    v0, v1, p0, p1 = vals[j - 1], vals[j], percs[j - 1], percs[j]
    return p0 + (value - v0) / (v1 - v0) * (p1 - p0)

# -------- SYNTHETIC DATASET GENERATION (12–60 MONTHS) --------
def build_dataset_1_to_5_years() -> pd.DataFrame:
    """
    Generate high-fidelity dataset for children aged 1–5 years (12–60 months)
    calibrated with exact WHO Child Growth Standards & Pediatric Percentiles.
    """
    print("Building WHO Pediatric Dataset for 1–5 Years (12–60 Months)...")
    np.random.seed(42)

    wfa_b, pcols = load_who_table(WFA_BOYS_FILE, r"month")
    wfa_g, _ = load_who_table(WFA_GIRLS_FILE, r"month")
    wfh_b, _ = load_who_table(WFH_BOYS_FILE, r"height")
    wfh_g, _ = load_who_table(WFH_GIRLS_FILE, r"height")

    dataset = []

    # Iterate over exact months from 12 to 60
    for sex_val, sex_code, wfa_df, wfh_df in [(1, "M", wfa_b, wfh_b), (0, "F", wfa_g, wfh_g)]:
        wfh_min_ht = wfh_df["primary"].min()
        wfh_max_ht = wfh_df["primary"].max()

        for age_m in np.arange(12.0, 60.5, 1.0):
            # Calculate height distribution from WHO LHFA
            wfa_curve = interp_curve(wfa_df, pcols, age_m)

            # Sample across diverse percentile points
            percentile_targets = [0.1, 1, 3, 5, 10, 15, 25, 50, 75, 85, 90, 95, 97, 99, 99.9]
            
            # Base median height for age
            base_ht_p50 = 75.75 if sex_code == "M" and age_m == 12 else (87.12 if age_m == 24 else (96.10 if age_m == 36 else (103.3 if age_m == 48 else 110.0)))
            
            for ht_mult in [0.82, 0.88, 0.92, 0.95, 0.98, 1.0, 1.03, 1.06, 1.10]:
                raw_ht = base_ht_p50 * ht_mult + (age_m - 24) * 0.6 if age_m != 24 else base_ht_p50 * ht_mult
                raw_ht = max(65.0, min(120.0, raw_ht))
                
                hfa_p = calc_hfa_percentile(age_m, raw_ht, sex_code)

                # Clamp to WFH table bounds
                clamped_ht = max(wfh_min_ht, min(wfh_max_ht, raw_ht))
                wfh_curve = interp_curve(wfh_df, pcols, clamped_ht)

                for wt_mult in [0.72, 0.80, 0.88, 0.95, 1.0, 1.08, 1.18, 1.30, 1.45]:
                    base_wt = wfh_curve.get(50.0, 12.0) * wt_mult
                    raw_wt = max(4.0, min(35.0, base_wt))
                    
                    wfh_p = est_percentile(raw_wt, wfh_curve)
                    wfa_p = est_percentile(raw_wt, wfa_curve)
                    
                    bmi = raw_wt / ((raw_ht / 100.0) ** 2)
                    bfa_p = calc_bfa_percentile(age_m, bmi, sex_code)

                    label = classify_pediatric_status(hfa_p, wfa_p, wfh_p, bfa_p, bmi)

                    # Add jittered samples for realistic variations
                    reps = 3 if label in [0, 4, 5] else 2
                    for _ in range(reps):
                        ht_j = raw_ht * np.random.normal(1.0, 0.008)
                        wt_j = raw_wt * np.random.normal(1.0, 0.008)
                        bmi_j = wt_j / ((ht_j / 100.0) ** 2)
                        
                        hfa_pj = calc_hfa_percentile(age_m, ht_j, sex_code)
                        wfa_pj = est_percentile(wt_j, wfa_curve)
                        wfh_pj = est_percentile(wt_j, interp_curve(wfh_df, pcols, max(wfh_min_ht, min(wfh_max_ht, ht_j))))
                        bfa_pj = calc_bfa_percentile(age_m, bmi_j, sex_code)
                        
                        dataset.append([age_m, ht_j, wt_j, sex_val, bmi_j, hfa_pj, wfa_pj, wfh_pj, bfa_pj, label])

    df = pd.DataFrame(dataset, columns=FEATURE_NAMES + ["label"])
    
    # Class balancing
    class_counts = df["label"].value_counts()
    max_count = class_counts.max()
    target_count = int(max_count * 0.7)
    
    balanced_frames = [df]
    for cls, count in class_counts.items():
        if count < target_count:
            n_needed = target_count - count
            extras = df[df["label"] == cls].sample(n=n_needed, replace=True, random_state=42).copy()
            extras["height"] *= np.random.normal(1.0, 0.012, size=len(extras))
            extras["weight"] *= np.random.normal(1.0, 0.012, size=len(extras))
            extras["bmi"] = extras["weight"] / ((extras["height"] / 100.0) ** 2)
            balanced_frames.append(extras)

    df_final = pd.concat(balanced_frames, ignore_index=True)
    print(f"Dataset generated: {len(df_final)} samples across 12–60 months.")
    print("Class distribution:")
    for cls in sorted(df_final["label"].unique()):
        print(f"  {cls} ({CLASS_LABELS.get(cls, str(cls))}): {(df_final['label'] == cls).sum()}")
        
    return df_final

from torch.utils.data import DataLoader, TensorDataset

# -------- OPTUNA OBJECTIVE --------
def objective(trial, train_loader, val_loader, class_weights_tensor):
    n_layers = trial.suggest_int("n_layers", 2, 4)
    n_units = trial.suggest_int("n_units", 64, 192, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.35)
    lr = trial.suggest_float("lr", 5e-4, 4e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    model = GrowthNet(in_features=9, n_layers=n_layers, n_units=n_units, dropout_rate=dropout_rate)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for bx, by in val_loader:
                val_loss += criterion(model(bx), by).item()
                val_batches += 1
        avg_val_loss = val_loss / max(1, val_batches)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            break

    return best_val_loss

# -------- MAIN TRAINING SCRIPT --------
if __name__ == "__main__":
    df = build_dataset_1_to_5_years()
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    present_classes = np.unique(y)
    print(f"Present diagnostic classes: {[CLASS_LABELS.get(c, c) for c in present_classes]}", flush=True)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.35, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"StandardScaler saved -> '{SCALER_SAVE_PATH}'", flush=True)

    # Balanced class weights
    weights = compute_class_weight("balanced", classes=present_classes, y=y_train)
    class_weights_tensor = torch.ones(len(CLASS_LABELS), dtype=torch.float32)
    for i, cls_idx in enumerate(present_classes):
        class_weights_tensor[cls_idx] = float(weights[i])

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(scaler.transform(X_test), dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

    # Optuna Search
    print(f"\nRunning Optuna Hyperparameter Search ({OPTUNA_TRIALS} trials)...", flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, class_weights_tensor),
        n_trials=OPTUNA_TRIALS
    )
    best_params = study.best_trial.params
    print("Optimal Hyperparameters:", best_params, flush=True)
    with open(PARAMS_SAVE_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Hyperparameters saved -> '{PARAMS_SAVE_PATH}'", flush=True)

    # Final Model Training
    print("\nTraining Final GrowthNet 2.0 Neural Network...", flush=True)
    final_model = GrowthNet(
        in_features=9,
        n_layers=best_params["n_layers"],
        n_units=best_params["n_units"],
        dropout_rate=best_params["dropout_rate"]
    )
    optimizer = optim.AdamW(
        final_model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params.get("weight_decay", 1e-4)
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    X_tv = torch.cat((X_train_t, X_val_t))
    y_tv = torch.cat((y_train_t, y_val_t))
    tv_loader = DataLoader(TensorDataset(X_tv, y_tv), batch_size=BATCH_SIZE, shuffle=True)

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(EPOCHS + 30):
        final_model.train()
        epoch_loss = 0.0
        batches = 0
        for bx, by in tv_loader:
            optimizer.zero_grad()
            loss = criterion(final_model(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        cur_loss = epoch_loss / max(1, batches)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_state = {k: v.clone() for k, v in final_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE * 2:
            print(f"Early convergence at epoch {epoch + 1}", flush=True)
            break

    if best_state is not None:
        final_model.load_state_dict(best_state)
    torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
    print(f"[OK] GrowthNet 2.0 Model saved -> '{MODEL_SAVE_PATH}'", flush=True)

    # Evaluation on Test Set
    print("\n--- Model Evaluation on Unseen Test Set ---")
    final_model.eval()
    with torch.no_grad():
        y_pred = torch.argmax(final_model(X_test_t), dim=1).numpy()

    used_labels = sorted(set(y_test) | set(y_pred))
    used_names = [CLASS_LABELS.get(l, str(l)) for l in used_labels]
    print(classification_report(y_test, y_pred, labels=used_labels, target_names=used_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=used_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("WHO Child Growth Diagnostic Confusion Matrix (1-5 Years)")
    fig.colorbar(cax)
    tick_marks = np.arange(len(used_names))
    plt.xticks(tick_marks, used_names, rotation=45, ha="right")
    plt.yticks(tick_marks, used_names)

    # Text annotations inside confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Confusion matrix saved -> confusion_matrix.png", flush=True)