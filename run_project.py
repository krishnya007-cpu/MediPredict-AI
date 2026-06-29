# ============================================================
# run_project.py
#
# PURPOSE:
#   One-click project setup script.
#   Run this ONCE before launching the Streamlit app:
#       python run_project.py
#
# WHAT IT DOES (in order):
#   1. Creates all required project directories.
#   2. Generates the synthetic insurance dataset (10,000 rows).
#   3. Runs the EDA and trains all four machine learning models.
#   4. Saves trained models (.pkl) and EDA charts.
#   5. Prints a final success message with the launch command.
#
# ESTIMATED TIME: 1–3 minutes depending on your machine.
# ============================================================

import os
import sys
import time

# ── Ensure the project root is on the Python path ──────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def create_directories():
    """Create all required project folders if they don't exist."""
    dirs = [
        os.path.join(ROOT, "data"),
        os.path.join(ROOT, "models"),
        os.path.join(ROOT, "reports"),
        os.path.join(ROOT, "notebooks"),
        os.path.join(ROOT, "app"),
        os.path.join(ROOT, "app", "pages"),
        os.path.join(ROOT, "app", "utils"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # Create __init__.py files so Python treats them as packages
    init_files = [
        os.path.join(ROOT, "app", "__init__.py"),
        os.path.join(ROOT, "app", "pages", "__init__.py"),
        os.path.join(ROOT, "app", "utils", "__init__.py"),
    ]
    for f in init_files:
        if not os.path.exists(f):
            open(f, "w").close()

    print("✅ Project directories created.")


def main():
    print("\n" + "=" * 65)
    print("  🏥  Emerging AI Applications in Actuarial Work")
    print("  Project Setup & Training Pipeline")
    print("=" * 65)

    # ── Step 1: Directories ────────────────────────────────────
    print("\n[1/3] Creating project directories...")
    create_directories()

    # ── Step 2: Generate dataset ───────────────────────────────
    data_path = os.path.join(ROOT, "data", "insurance_data.csv")
    if os.path.exists(data_path):
        print("\n[2/3] Dataset already exists — skipping generation.")
        print(f"      ({data_path})")
        print("      Delete the file and re-run to regenerate.")
    else:
        print("\n[2/3] Generating synthetic insurance dataset...")
        t0 = time.time()
        from data.generate_dataset import main as gen_main
        gen_main()
        print(f"      Done in {time.time() - t0:.1f}s")

    # ── Step 3: EDA + Model training ───────────────────────────
    models_exist = all(
        os.path.exists(os.path.join(ROOT, "models", f))
        for f in ["risk_model.pkl", "premium_model.pkl",
                  "fraud_model.pkl", "claim_model.pkl", "encoders.pkl"]
    )

    if models_exist:
        print("\n[3/3] All model files already exist — skipping training.")
        print("      Delete files in models/ and re-run to retrain.")
    else:
        print("\n[3/3] Running EDA and training models...")
        print("      This may take 1–3 minutes...")
        t0 = time.time()
        from notebooks.eda_and_training import main as train_main
        train_main()
        print(f"      Done in {time.time() - t0:.1f}s")

    # ── Final message ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ✅  Setup complete!")
    print()
    print("  📂 Files created:")
    print(f"     • data/insurance_data.csv")
    print(f"     • models/risk_model.pkl")
    print(f"     • models/premium_model.pkl")
    print(f"     • models/fraud_model.pkl")
    print(f"     • models/claim_model.pkl")
    print(f"     • models/encoders.pkl")
    print(f"     • reports/*.png  (EDA and model charts)")
    print()
    print("  🚀  Launch the app with:")
    print()
    print("      streamlit run app/app.py")
    print()
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()