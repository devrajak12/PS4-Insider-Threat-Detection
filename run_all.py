"""
PS4 — Data Access Audit & Insider Threat Detection
Run the full pipeline with a single command.

Usage:
    python run_all.py
    streamlit run ps4_step4_dashboard.py
"""

import subprocess
import sys
import time
from pathlib import Path

def run_step(name, script):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n❌  {name} FAILED — stopping pipeline.")
        sys.exit(1)
    print(f"\n✅  {name} done in {elapsed:.1f}s")

print("\n🔐  PS4 — Insider Threat Detection Pipeline")
print("    Societe Generale Hackathon\n")

# Check data files exist
for f in ["sample_data/data_access_logs.csv", "sample_data/user_profiles.csv"]:
    if not Path(f).exists():
        print(f"❌  Missing: {f}")
        print("    Place your CSV files in the sample_data/ folder and retry.")
        sys.exit(1)

Path("outputs").mkdir(exist_ok=True)

run_step("Step 1 — Feature Engineering",         "ps4_step1_features.py")
run_step("Step 2 — Model 1: Event Anomaly Detection", "ps4_step2_model1.py")
run_step("Step 3 — Model 2: User Risk Scoring",  "ps4_step3_model2.py")

print(f"\n{'='*60}")
print("  PIPELINE COMPLETE")
print(f"{'='*60}")
print("""
  Outputs generated:
    outputs/features_events.csv
    outputs/features_users.csv
    outputs/model1_predictions.csv
    outputs/model1_feature_importance.csv
    outputs/model2_user_risk.csv
    outputs/model2_top10_report.txt

  Launch dashboard:
    streamlit run ps4_step4_dashboard.py

  Then open: http://localhost:8501
""")


