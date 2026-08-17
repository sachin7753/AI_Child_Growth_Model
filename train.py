import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import joblib
import json

# -------- CONFIG --------
WFA_BOYS_FILE = "tab_wfa_boys_p_0_5.xlsx"
WFA_GIRLS_FILE = "tab_wfa_girls_p_0_5.xlsx"
HFA_BOYS_FILE = "tab_hfa_boys_p_0_5.xlsx"
HFA_GIRLS_FILE = "tab_hfa_girls_p_0_5.xlsx"
WFH_BOYS_FILE = "tab_wfh_boys_p_0_5.xlsx"
WFH_GIRLS_FILE = "tab_wfh_girls_p_0_5.xlsx"
BFA_FILE = "bmi.csv.xlsx"

MODEL_SAVE_PATH = "growth_model.pth"
SCALER_SAVE_PATH = "scaler.joblib"
PARAMS_SAVE_PATH = "best_params.json"

# Training parameters
EPOCHS = 300
PATIENCE = 25
OPTUNA_TRIALS = 80
CLASS_LABELS = {0: "Underweight", 1: "Healthy", 2: "Overweight", 3: "Obese", 4: "Stunted", 5: "Normal Ht"}

# -------- AI MODEL DEFINITION for Optuna --------
class GrowthNet(nn.Module):
    def __init__(self, n_layers=2, n_units=64, dropout_rate=0.3):
        super().__init__()
        layers = []
        in_features = 4
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, n_units)); layers.append(nn.ReLU()); layers.append(nn.Dropout(dropout_rate))
            in_features = n_units
        layers.append(nn.Linear(in_features, len(CLASS_LABELS)))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

# -------- DATA UTILITIES --------
def load_who_table(path: str, primary_col_regex: str):
    """Load WHO reference table → (DataFrame with 'primary' col + P-columns, pcols list)."""
    df = pd.read_excel(path)
    primary_col = next(c for c in df.columns if re.search(primary_col_regex, str(c), re.I))
    pcols = [c for c in df.columns if re.match(r"P\d+", str(c))]
    df = df[[primary_col] + pcols].copy()
    df.columns = ["primary"] + pcols
    return df, pcols


def interp_curve(ref_df, pcols, val):
    """Interpolate percentile curve from WHO table for a given primary value."""
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
    """Estimate percentile given a measurement value and a {perc: ref_value} dict."""
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


def classify_child(wfh_p, hfa_p, bmi):
    """
    WHO-based classification matching app.py rule-based logic.
    Priority: Underweight > Obese > Overweight > Stunted > Healthy.
    """
    if wfh_p < 3:
        return 0  # Underweight
    if wfh_p > 85:
        return 3 if bmi >= 30 else 2  # Obese or Overweight
    if bmi >= 30:
        return 3  # Obese
    if bmi >= 25:
        return 2  # Overweight
    if hfa_p < 3:
        return 4  # Stunted
    return 1  # Healthy


def build_dataset() -> pd.DataFrame:
    """
    Generate synthetic children using WHO reference tables.
    For each (age, sex): sample heights from HFA percentiles, weights from WFH percentiles.
    Label using the same rules as app.py's ai_predict().
    """
    print("Building WHO-based synthetic dataset...")

    wfh_boys, pcols = load_who_table(WFH_BOYS_FILE, r"height")
    wfh_girls, _ = load_who_table(WFH_GIRLS_FILE, r"height")
    hfa_boys, _ = load_who_table(HFA_BOYS_FILE, r"month")
    hfa_girls, _ = load_who_table(HFA_GIRLS_FILE, r"month")
    wfa_boys, _ = load_who_table(WFA_BOYS_FILE, r"month")
    wfa_girls, _ = load_who_table(WFA_GIRLS_FILE, r"month")

    np.random.seed(42)
    dataset = []

    # ---- Part 1: Ages 24–60 months (HFA + WFH fully available) ----
    for sex_val, wfh, hfa in [(1, wfh_boys, hfa_boys), (0, wfh_girls, hfa_girls)]:
        wfh_lo, wfh_hi = wfh["primary"].min(), wfh["primary"].max()

        for _, hfa_row in hfa.iterrows():
            age_m = float(hfa_row["primary"])

            for ht_pcol in pcols:
                ht = float(hfa_row[ht_pcol])
                ht_p = float(re.findall(r"\d+", ht_pcol)[0])

                if ht < wfh_lo or ht > wfh_hi:
                    continue

                wfh_curve = interp_curve(wfh, pcols, ht)

                for wt_p_val, wt in wfh_curve.items():
                    bmi = wt / ((ht / 100) ** 2)
                    label = classify_child(wt_p_val, ht_p, bmi)

                    # More jitter copies for minority classes
                    n_jitter = 4 if label in [0, 4] else 2
                    for _ in range(n_jitter):
                        ht_j = ht * np.random.normal(1, 0.012)
                        wt_j = wt * np.random.normal(1, 0.012)
                        dataset.append([age_m, ht_j, wt_j, sex_val, label])

    # ---- Part 2: Ages 0–23 months (WFA + estimated heights) ----
    for sex_val, wfa, wfh in [(1, wfa_boys, wfh_boys), (0, wfa_girls, wfh_girls)]:
        wfh_lo, wfh_hi = wfh["primary"].min(), wfh["primary"].max()

        for _, wfa_row in wfa.iterrows():
            age_m = float(wfa_row["primary"])
            if age_m > 23:
                continue

            # Rough WHO-aligned height estimate for infants
            if age_m <= 12:
                est_ht = 50 + 2.0 * age_m  # ~50 cm at birth → ~74 cm at 12 mo
            else:
                est_ht = 74 + 1.0 * (age_m - 12)  # ~74 → ~85 cm at 23 mo

            for wt_pcol in pcols:
                wt = float(wfa_row[wt_pcol])
                wt_p = float(re.findall(r"\d+", wt_pcol)[0])

                # Try multiple height multipliers to cover stunted / normal / tall
                for ht_mult in [0.90, 0.94, 0.97, 1.0, 1.03, 1.06]:
                    ht = est_ht * ht_mult

                    # Map height multiplier to approximate HFA percentile
                    if ht_mult <= 0.92:
                        hfa_p = 0.5  # deeply stunted
                    elif ht_mult <= 0.95:
                        hfa_p = 2.0  # stunted
                    elif ht_mult <= 0.98:
                        hfa_p = 5.0  # borderline
                    else:
                        hfa_p = 50.0  # normal

                    # Compute WFH percentile if the height is in WFH range
                    if wfh_lo <= ht <= wfh_hi:
                        wfh_curve = interp_curve(wfh, pcols, ht)
                        wfh_p = est_percentile(wt, wfh_curve)
                    else:
                        wfh_p = wt_p  # fall back to WFA percentile

                    bmi = wt / ((ht / 100) ** 2)
                    label = classify_child(wfh_p, hfa_p, bmi)

                    ht_j = ht * np.random.normal(1, 0.01)
                    wt_j = wt * np.random.normal(1, 0.01)
                    dataset.append([age_m, ht_j, wt_j, sex_val, label])

    df = pd.DataFrame(dataset, columns=["age", "height", "weight", "sex", "label"])

    # ---- Oversample minority classes to reduce imbalance ----
    class_counts = df["label"].value_counts()
    target_count = int(class_counts.max() * 0.6)  # minority classes → at least 60 % of majority
    oversampled = [df]
    for cls, count in class_counts.items():
        if count < target_count:
            n_needed = target_count - count
            extras = df[df["label"] == cls].sample(n=n_needed, replace=True, random_state=42).copy()
            extras["height"] *= np.random.normal(1, 0.02, size=len(extras))
            extras["weight"] *= np.random.normal(1, 0.02, size=len(extras))
            oversampled.append(extras)
    df = pd.concat(oversampled, ignore_index=True)

    print(f"Dataset built: {len(df)} samples")
    print("Class distribution:")
    for cls in sorted(df["label"].unique()):
        print(f"  {CLASS_LABELS.get(cls, cls)}: {(df['label'] == cls).sum()}")
    return df


# -------- OPTUNA OBJECTIVE --------
def objective(trial, X_train, y_train, X_val, y_val, class_weights_tensor):
    n_layers = trial.suggest_int("n_layers", 2, 4)
    n_units = trial.suggest_int("n_units", 48, 192, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.45)
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)

    model = GrowthNet(n_layers=n_layers, n_units=n_units, dropout_rate=dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            break
    return best_val_loss


# -------- MAIN TRAINING SCRIPT --------
if __name__ == "__main__":
    data = build_dataset()
    X = data.drop("label", axis=1).values.astype(np.float32)
    y = data["label"].values.astype(np.int64)

    # Only keep classes that actually exist in dataset
    present_classes = np.unique(y)
    print(f"Present classes: {[CLASS_LABELS.get(c, c) for c in present_classes]}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Scaler saved → '{SCALER_SAVE_PATH}'")

    # Balanced class weights
    weights_array = compute_class_weight("balanced", classes=present_classes, y=y_train)
    class_weights_tensor = torch.ones(len(CLASS_LABELS), dtype=torch.float32)
    for i, cls_idx in enumerate(present_classes):
        class_weights_tensor[cls_idx] = torch.tensor(weights_array[i], dtype=torch.float32)
    print("Class weights:", {CLASS_LABELS.get(int(c), c): f"{class_weights_tensor[c]:.2f}" for c in present_classes})

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(scaler.transform(X_test), dtype=torch.float32)

    # ---- Optuna hyperparameter search ----
    print(f"\nStarting Optuna search ({OPTUNA_TRIALS} trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X_train_t, y_train_t, X_val_t, y_val_t, class_weights_tensor),
        n_trials=OPTUNA_TRIALS,
    )
    best_params = study.best_trial.params
    print("Best trial:", best_params)
    with open(PARAMS_SAVE_PATH, "w") as f:
        json.dump(best_params, f)
    print(f"Hyperparameters saved → '{PARAMS_SAVE_PATH}'")

    # ---- Final training on train+val with best params ----
    print("\nTraining final model...")
    final_model = GrowthNet(
        n_layers=best_params["n_layers"],
        n_units=best_params["n_units"],
        dropout_rate=best_params["dropout_rate"],
    )
    optimizer = optim.Adam(final_model.parameters(), lr=best_params["lr"])
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    X_tv = torch.cat((X_train_t, X_val_t))
    y_tv = torch.cat((y_train_t, y_val_t))

    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    for epoch in range(EPOCHS):
        final_model.train()
        optimizer.zero_grad()
        loss = criterion(final_model(X_tv), y_tv)
        loss.backward()
        optimizer.step()
        cur_loss = loss.item()
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_state = {k: v.clone() for k, v in final_model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE * 2:
            print(f"Early stop at epoch {epoch + 1}")
            break
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{EPOCHS}  loss={cur_loss:.4f}")

    if best_state is not None:
        final_model.load_state_dict(best_state)
    torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved → '{MODEL_SAVE_PATH}'")

    # ---- Evaluation ----
    print("\n--- Evaluation on held-out test set ---")
    final_model.eval()
    with torch.no_grad():
        y_pred = torch.argmax(final_model(X_test_t), dim=1).numpy()

    used_labels = sorted(set(y_test) | set(y_pred))
    used_names = [CLASS_LABELS.get(l, str(l)) for l in used_labels]
    print(classification_report(y_test, y_pred, labels=used_labels, target_names=used_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=used_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=used_names, yticklabels=used_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Confusion matrix saved → confusion_matrix.png")
    plt.show()