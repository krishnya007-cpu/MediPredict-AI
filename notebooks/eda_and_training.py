# ============================================================
# notebooks/eda_and_training.py
# STANDALONE TRAINING SCRIPT — Python 3.7+ Compatible
# Run: python notebooks/eda_and_training.py
# ============================================================

import os, sys, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, roc_curve,
)
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

# ── Resolve paths ──────────────────────────────────────────────
THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(THIS_DIR)
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

FEATURE_COLS = [
    "age", "gender", "region", "marital_status", "num_dependents",
    "bmi", "smoker", "chronic_disease", "prev_hospitalizations",
    "num_medications", "exercise_frequency", "coverage_type",
    "policy_duration", "deductible", "num_prev_claims",
    "annual_income", "credit_score",
]


# ==============================================================
# 1. LOAD DATA
# ==============================================================
def load_data():
    path = os.path.join(DATA_DIR, "insurance_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run: python data/generate_dataset.py"
        )
    df = pd.read_csv(path)
    print(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ==============================================================
# 2. PREPROCESS
# ==============================================================
def preprocess(df):
    print("\n[STEP 2] Preprocessing...")
    df = df.copy()

    # Fill missing numeric → median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Fill missing categorical → mode
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Encode categorical feature columns
    cat_cols = ["gender", "region", "marital_status", "coverage_type", "smoker"]
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[cat_cols] = oe.fit_transform(df[cat_cols])

    # Encode target: risk_level → integer classes
    le = LabelEncoder()
    df["risk_level_encoded"] = le.fit_transform(df["risk_level"])
    print(f"   Risk classes: {list(le.classes_)}")

    # Scale numeric features
    scale_cols = [
        "age", "bmi", "num_dependents", "prev_hospitalizations",
        "num_medications", "exercise_frequency", "policy_duration",
        "deductible", "num_prev_claims", "annual_income", "credit_score",
    ]
    sc = StandardScaler()
    df[scale_cols] = sc.fit_transform(df[scale_cols])

    encoders = {
        "ordinal_encoder":    oe,
        "risk_label_encoder": le,
        "scaler":             sc,
        "cat_feature_cols":   cat_cols,
        "scale_cols":         scale_cols,
    }
    print("   ✅ Preprocessing complete")
    return df, encoders


# ==============================================================
# 3. EDA CHARTS
# ==============================================================
def run_eda(df):
    print("\n[STEP 3] Generating EDA charts...")

    def save_fig(fig, name):
        fig.savefig(os.path.join(REPORTS_DIR, name),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

    # Age histogram
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(df["age"], bins=30, color="#4C72B0", edgecolor="white")
    ax.set_title("Age Distribution"); ax.set_xlabel("Age")
    save_fig(fig, "eda_age.png")

    # Risk level bar
    rc = df["risk_level"].value_counts()
    col_map = {"Low":"#2ECC71","Medium":"#F39C12","High":"#E74C3C"}
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(rc.index, rc.values,
           color=[col_map.get(k,"#777") for k in rc.index],
           edgecolor="white")
    ax.set_title("Risk Level Distribution")
    save_fig(fig, "eda_risk_level.png")

    # Premium histogram
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(df["annual_premium"], bins=40,
            color="#8E44AD", edgecolor="white")
    ax.set_title("Annual Premium Distribution ($)")
    save_fig(fig, "eda_premium.png")

    # Fraud bar
    fc = df["fraud_flag"].value_counts()
    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(["Legitimate","Fraud"], fc.values,
           color=["#27AE60","#C0392B"], edgecolor="white")
    ax.set_title("Fraud Flag Distribution")
    save_fig(fig, "eda_fraud.png")

    # Correlation heatmap
    num_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12,9))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.4, ax=ax,
                annot_kws={"size":7})
    ax.set_title("Correlation Heatmap")
    save_fig(fig, "eda_correlation.png")

    print(f"   ✅ Charts saved to {REPORTS_DIR}/")


# ==============================================================
# 4. TRAIN RISK MODEL
# ==============================================================
def train_risk(df, encoders):
    print("\n" + "="*55)
    print("  [MODEL 1] Risk Level Classification")
    print("="*55)

    X = df[FEATURE_COLS]
    y = df["risk_level_encoded"]
    Xtr,Xte,ytr,yte = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(Xtr):,}  Test: {len(Xte):,}")

    m = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        min_samples_leaf=5, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    print("  Training RandomForest...")
    m.fit(Xtr, ytr)

    yp   = m.predict(Xte)
    acc  = accuracy_score(yte, yp)
    f1   = f1_score(yte, yp, average="weighted")
    lbls = encoders["risk_label_encoder"].classes_

    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(classification_report(yte, yp, target_names=lbls))

    # Confusion matrix chart
    cm = confusion_matrix(yte, yp)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=lbls, yticklabels=lbls, ax=ax)
    ax.set_title("Risk — Confusion Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR,"model_risk_cm.png"), dpi=100)
    plt.close(fig)

    out = os.path.join(MODELS_DIR, "risk_model.pkl")
    joblib.dump(m, out)
    print(f"  ✅ Saved → {out}")
    return m


# ==============================================================
# 5. TRAIN PREMIUM MODEL
# ==============================================================
def train_premium(df):
    print("\n" + "="*55)
    print("  [MODEL 2] Annual Premium Regression")
    print("="*55)

    X = df[FEATURE_COLS]
    y = df["annual_premium"]
    Xtr,Xte,ytr,yte = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(Xtr):,}  Test: {len(Xte):,}")

    m = XGBRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=6, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1.0, random_state=42, verbosity=0,
    )
    print("  Training XGBRegressor...")
    m.fit(Xtr, ytr, eval_set=[(Xte,yte)], verbose=False)

    yp   = m.predict(Xte)
    rmse = float(np.sqrt(mean_squared_error(yte, yp)))
    r2   = float(r2_score(yte, yp))
    mae  = float(np.mean(np.abs(np.array(yte) - yp)))

    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  R²   : {r2:.4f}")

    out = os.path.join(MODELS_DIR, "premium_model.pkl")
    joblib.dump(m, out)
    print(f"  ✅ Saved → {out}")
    return m


# ==============================================================
# 6. TRAIN FRAUD MODEL
# ==============================================================
def train_fraud(df):
    print("\n" + "="*55)
    print("  [MODEL 3] Fraud Detection Classification")
    print("="*55)

    X = df[FEATURE_COLS]
    y = df["fraud_flag"]
    Xtr,Xte,ytr,yte = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(Xtr):,}  Test: {len(Xte):,}")

    neg   = int((ytr == 0).sum())
    pos   = int((ytr == 1).sum())
    scale = neg / pos
    print(f"  Fraud rate: {pos/(neg+pos):.2%}  scale_pos_weight={scale:.1f}")

    m = XGBClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=5, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=scale,
        eval_metric="logloss", random_state=42,
        verbosity=0, use_label_encoder=False,
    )
    print("  Training XGBClassifier...")
    m.fit(Xtr, ytr, eval_set=[(Xte,yte)], verbose=False)

    yp   = m.predict(Xte)
    yprb = m.predict_proba(Xte)[:,1]
    print(f"  Accuracy  : {accuracy_score(yte,yp):.4f}")
    print(f"  Precision : {precision_score(yte,yp,zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(yte,yp,zero_division=0):.4f}")
    print(f"  F1        : {f1_score(yte,yp,zero_division=0):.4f}")
    print(f"  AUC-ROC   : {roc_auc_score(yte,yprb):.4f}")
    print(classification_report(yte,yp,target_names=["Legit","Fraud"]))

    out = os.path.join(MODELS_DIR, "fraud_model.pkl")
    joblib.dump(m, out)
    print(f"  ✅ Saved → {out}")
    return m


# ==============================================================
# 7. TRAIN CLAIM MODEL
# ==============================================================
def train_claim(df):
    print("\n" + "="*55)
    print("  [MODEL 4] Claim Amount Regression")
    print("="*55)

    X = df[FEATURE_COLS]
    y = df["claim_amount"]
    Xtr,Xte,ytr,yte = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(Xtr):,}  Test: {len(Xte):,}")
    print(f"  Non-zero claims: {int((ytr>0).sum()):,} ({float((ytr>0).mean()):.1%})")

    m = XGBRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=6, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1.0, random_state=42, verbosity=0,
    )
    print("  Training XGBRegressor...")
    m.fit(Xtr, ytr, eval_set=[(Xte,yte)], verbose=False)

    yp   = np.clip(m.predict(Xte), 0, None)
    rmse = float(np.sqrt(mean_squared_error(yte, yp)))
    r2   = float(r2_score(yte, yp))
    mae  = float(np.mean(np.abs(np.array(yte) - yp)))

    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  R²   : {r2:.4f}")

    out = os.path.join(MODELS_DIR, "claim_model.pkl")
    joblib.dump(m, out)
    print(f"  ✅ Saved → {out}")
    return m


# ==============================================================
# 8. SAVE ENCODERS
# ==============================================================
def save_encoders(encoders):
    out = os.path.join(MODELS_DIR, "encoders.pkl")
    joblib.dump(encoders, out)
    print(f"\n  ✅ Encoders saved → {out}")


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("\n" + "="*55)
    print("  Actuarial AI — Training Pipeline")
    print("="*55)

    # 1. Load raw data
    print("\n[STEP 1] Loading data...")
    df_raw = load_data()

    # 2. EDA on raw data
    run_eda(df_raw)

    # 3. Preprocess
    df_clean, encoders = preprocess(df_raw)

    # 4. Train all 4 models
    train_risk(df_clean, encoders)
    train_premium(df_clean)
    train_fraud(df_clean)
    train_claim(df_clean)

    # 5. Save encoders
    save_encoders(encoders)

    # 6. Final summary
    print("\n" + "="*55)
    print("  ✅ ALL DONE! Files in models/:")
    for f in sorted(os.listdir(MODELS_DIR)):
        sz = os.path.getsize(os.path.join(MODELS_DIR, f))
        print(f"     ✅ {f:<25} ({sz/1024:.1f} KB)")
    print()
    print("  🚀 Refresh your browser now!")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()