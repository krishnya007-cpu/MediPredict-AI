# ============================================================
# data/generate_dataset.py
#
# PURPOSE:
#   This script generates a realistic, synthetic insurance dataset
#   with 10,000 records. We use NumPy's random functions to create
#   data that mimics real insurance policyholder information.
#
# WHY SYNTHETIC DATA?
#   Real insurance data is private and regulated. Synthetic data
#   lets us build and test our models without privacy concerns,
#   while still being realistic enough to learn from.
#
# OUTPUT:
#   Saves 'insurance_data.csv' into the data/ folder.
# ============================================================

import numpy as np
import pandas as pd
import os
import sys

# ── make sure we can always find the project root ──────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def generate_insurance_dataset(n_samples: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic health/insurance dataset.

    Parameters
    ----------
    n_samples : int
        Number of rows (policyholders) to create.
    seed : int
        Random seed for reproducibility — same seed → same data every time.

    Returns
    -------
    pd.DataFrame
        A DataFrame with 22 columns covering demographics, health,
        policy details, financial info, and four target variables.
    """

    # Fix the random seed so results are reproducible
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. DEMOGRAPHIC FEATURES
    # ------------------------------------------------------------------

    # Age: most policyholders are between 18 and 75
    age = rng.integers(18, 76, size=n_samples)

    # Gender: roughly 50/50 split
    gender = rng.choice(["Male", "Female"], size=n_samples, p=[0.50, 0.50])

    # Region: four geographic zones
    region = rng.choice(
        ["Northeast", "Southeast", "Midwest", "West"],
        size=n_samples,
        p=[0.25, 0.30, 0.25, 0.20],
    )

    # Marital status
    marital_status = rng.choice(
        ["Single", "Married", "Divorced", "Widowed"],
        size=n_samples,
        p=[0.30, 0.50, 0.15, 0.05],
    )

    # Number of dependents (children / family members covered)
    num_dependents = rng.integers(0, 6, size=n_samples)

    # ------------------------------------------------------------------
    # 2. HEALTH FEATURES
    # ------------------------------------------------------------------

    # BMI: Body Mass Index — normal distribution centred at 27
    bmi = np.clip(rng.normal(loc=27.0, scale=5.5, size=n_samples), 15.0, 55.0).round(1)

    # Smoker status: ~20 % of policyholders smoke (higher risk)
    smoker = rng.choice(["Yes", "No"], size=n_samples, p=[0.20, 0.80])

    # Chronic disease: ~30 % have at least one chronic condition
    chronic_disease = rng.choice([1, 0], size=n_samples, p=[0.30, 0.70])

    # Previous hospitalisations in the last 5 years
    prev_hospitalizations = rng.integers(0, 6, size=n_samples)

    # Number of medications currently taken
    num_medications = rng.integers(0, 11, size=n_samples)

    # Exercise frequency (days per week)
    exercise_frequency = rng.integers(0, 8, size=n_samples)

    # ------------------------------------------------------------------
    # 3. POLICY / COVERAGE FEATURES
    # ------------------------------------------------------------------

    # Coverage type
    coverage_type = rng.choice(
        ["Basic", "Standard", "Premium", "Comprehensive"],
        size=n_samples,
        p=[0.20, 0.35, 0.30, 0.15],
    )

    # Policy duration in years (how long they have been insured)
    policy_duration = rng.integers(1, 21, size=n_samples)

    # Annual deductible amount in USD
    deductible = rng.choice([500, 1000, 1500, 2000, 3000, 5000], size=n_samples)

    # Number of previous claims filed
    num_prev_claims = rng.integers(0, 8, size=n_samples)

    # ------------------------------------------------------------------
    # 4. FINANCIAL FEATURES
    # ------------------------------------------------------------------

    # Annual income in USD — log-normal to mimic real income distribution
    annual_income = np.clip(
        rng.lognormal(mean=10.8, sigma=0.6, size=n_samples), 20_000, 500_000
    ).round(-2)  # round to nearest 100

    # Credit score: 300 – 850
    credit_score = np.clip(
        rng.normal(loc=680, scale=80, size=n_samples), 300, 850
    ).astype(int)

    # ------------------------------------------------------------------
    # 5. TARGET VARIABLES  (what we want to predict)
    # ------------------------------------------------------------------

    # ── 5a. RISK SCORE (continuous, 0–100) ──────────────────────────
    # Higher age, BMI, smoker status, chronic disease → higher risk
    risk_score = (
        0.25 * (age / 75 * 100)                         # age contribution
        + 0.20 * (bmi / 55 * 100)                       # BMI contribution
        + 0.20 * (np.where(smoker == "Yes", 1, 0) * 100) # smoking
        + 0.15 * (chronic_disease * 100)                 # chronic illness
        + 0.10 * (prev_hospitalizations / 5 * 100)       # past hospitalisations
        + 0.05 * (num_prev_claims / 7 * 100)             # claims history
        + rng.normal(0, 3, n_samples)                    # noise
    )
    risk_score = np.clip(risk_score, 0, 100)

    # Map continuous risk score to categorical Risk Level
    risk_level = np.where(
        risk_score < 35, "Low",
        np.where(risk_score < 65, "Medium", "High")
    )

    # ── 5b. ANNUAL PREMIUM (USD) ─────────────────────────────────────
    # Premium is driven by risk factors plus coverage type
    coverage_multiplier = {
        "Basic": 0.7,
        "Standard": 1.0,
        "Premium": 1.4,
        "Comprehensive": 1.8,
    }
    cov_mult = np.array([coverage_multiplier[c] for c in coverage_type])

    base_premium = (
        500
        + 30 * age
        + 25 * bmi
        + np.where(smoker == "Yes", 2000, 0)
        + 800 * chronic_disease
        + 150 * prev_hospitalizations
        + 100 * num_prev_claims
        - 0.002 * annual_income        # higher income → slight discount
        - 0.5 * credit_score           # better credit → lower premium
        + rng.normal(0, 200, n_samples) # noise
    )
    annual_premium = np.clip(base_premium * cov_mult, 500, 50_000).round(2)

    # ── 5c. FRAUD FLAG (binary: 1 = fraud, 0 = legitimate) ──────────
    # ~8 % fraud rate; more likely with many claims, short policy duration
    fraud_prob = (
        0.02
        + 0.04 * (num_prev_claims / 7)
        + 0.03 * (1 - policy_duration / 20)
        + 0.02 * (np.where(smoker == "Yes", 1, 0))
        + rng.uniform(0, 0.02, n_samples)
    )
    fraud_prob = np.clip(fraud_prob, 0, 1)
    fraud_flag = (rng.uniform(0, 1, n_samples) < fraud_prob).astype(int)

    # ── 5d. CLAIM AMOUNT (USD) ───────────────────────────────────────
    # Only policyholders with at least 1 claim have a positive amount
    has_claim = (num_prev_claims > 0).astype(int)
    base_claim = (
        500 * num_prev_claims
        + 200 * age
        + 300 * bmi
        + np.where(smoker == "Yes", 3000, 0)
        + 1500 * chronic_disease
        + 400 * prev_hospitalizations
        + rng.normal(0, 500, n_samples)
    )
    claim_amount = np.clip(base_claim * has_claim, 0, 200_000).round(2)

    # ------------------------------------------------------------------
    # 6. ASSEMBLE THE DATAFRAME
    # ------------------------------------------------------------------

    df = pd.DataFrame({
        # Demographics
        "age":                 age,
        "gender":              gender,
        "region":              region,
        "marital_status":      marital_status,
        "num_dependents":      num_dependents,

        # Health
        "bmi":                 bmi,
        "smoker":              smoker,
        "chronic_disease":     chronic_disease,
        "prev_hospitalizations": prev_hospitalizations,
        "num_medications":     num_medications,
        "exercise_frequency":  exercise_frequency,

        # Policy
        "coverage_type":       coverage_type,
        "policy_duration":     policy_duration,
        "deductible":          deductible,
        "num_prev_claims":     num_prev_claims,

        # Financial
        "annual_income":       annual_income,
        "credit_score":        credit_score,

        # Targets
        "risk_score":          risk_score.round(2),
        "risk_level":          risk_level,
        "annual_premium":      annual_premium,
        "fraud_flag":          fraud_flag,
        "claim_amount":        claim_amount,
    })

    # ------------------------------------------------------------------
    # 7. INJECT REALISTIC MISSING VALUES (~2 % per column)
    # ------------------------------------------------------------------
    # In real data, some fields are not always filled in.
    # We randomly set ~2 % of certain columns to NaN.

    cols_with_missing = ["bmi", "credit_score", "annual_income", "num_medications"]
    for col in cols_with_missing:
        missing_idx = rng.choice(n_samples, size=int(0.02 * n_samples), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


def main():
    """Entry point: generate dataset and save to CSV."""

    # Make sure the data/ folder exists
    data_dir = os.path.join(ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, "insurance_data.csv")

    print("=" * 60)
    print("  Generating Synthetic Insurance Dataset")
    print("=" * 60)

    df = generate_insurance_dataset(n_samples=10_000, seed=42)

    # Save to CSV
    df.to_csv(output_path, index=False)

    # Print a summary
    print(f"\n✅ Dataset saved → {output_path}")
    print(f"\n📊 Shape       : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\n📋 Columns     :\n{list(df.columns)}")
    print(f"\n🔍 Missing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string())
    print(f"\n📈 Target variable distributions:")
    print("\n  Risk Level counts:")
    print(df["risk_level"].value_counts().to_string())
    print("\n  Fraud Flag counts:")
    print(df["fraud_flag"].value_counts().to_string())
    print(f"\n  Annual Premium  — mean: ${df['annual_premium'].mean():,.0f}  "
          f"std: ${df['annual_premium'].std():,.0f}")
    print(f"  Claim Amount    — mean: ${df['claim_amount'].mean():,.0f}  "
          f"std: ${df['claim_amount'].std():,.0f}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()