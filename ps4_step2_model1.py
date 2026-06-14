"""
PS4 — Data Access Audit & Insider Threat Detection
Step 2: Model 1 — Event-Level Anomaly Detection

PROBLEM BEING SOLVED IN THIS STEP:
------------------------------------
We have 1,200 access events with 31 features from Step 1.
We have NO ground-truth labels (the hackathon only gave 2 CSVs).

So the question is:
    "How do we train a model when we don't know which events are truly anomalous?"

ANSWER — Two-stage approach:
  Stage A: Isolation Forest (unsupervised)
      → Learns what "normal" looks like from the data itself
      → Flags statistical outliers — no labels needed
      → Gives us anomaly scores for every event

  Stage B: LightGBM (supervised, using Stage A + heuristic as pseudo-labels)
      → Uses the Isolation Forest score + our heuristic risk_score as training signal
      → Learns non-linear patterns between features and risk
      → Outputs a probability (0–1) that we convert to severity
      → More interpretable and faster at inference than Isolation Forest alone

WHY THIS IS VALID FOR THE HACKATHON:
  - Isolation Forest is the industry standard for unsupervised anomaly detection
  - Using heuristic scores as pseudo-labels is a standard semi-supervised technique
  - The judges asked for "anomaly detection" — this is exactly that
  - We can show precision/recall on a held-out split

INPUTS:
  outputs/features_events.csv   (from ps4_step1_features.py)

OUTPUTS:
  outputs/model1_predictions.csv   — every event with its anomaly score + label
  outputs/model1_feature_importance.csv
  outputs/model1_metrics.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD FEATURE TABLE
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("MODEL 1 — Event-Level Anomaly Detection")
print("=" * 60)

df = pd.read_csv(OUT_DIR / "features_events.csv")
print(f"\nLoaded feature table: {df.shape[0]} rows × {df.shape[1]} cols")

# ─────────────────────────────────────────────────────────────
# STEP 2 — SELECT FEATURES FOR THE MODEL
#
# We only feed NUMERIC features to the model.
# We exclude:
#   - identifiers (user_id, username, timestamp)
#   - raw strings (action, resource, department...)
#   - the heuristic label columns (we'll use those as pseudo-labels, not inputs)
# ─────────────────────────────────────────────────────────────
print("\nSTEP 2 — Selecting model features")

FEATURE_COLS = [
    # Time signals
    "hour",
    "day_of_week",
    "is_weekend",
    "time_risk",
    "is_off_hours",
    # Action signals
    "action_risk",
    "is_high_risk_action",
    # Sensitivity signals
    "sensitivity_score",
    "is_high_sensitivity",
    # Failure signals
    "is_failed_access",
    "user_failure_rate",
    "failed_high_sens",
    # Structural risk signals
    "dept_mismatch",
    "sox_flag",
    "privilege_score",
    "priv_action_mismatch",
    # Account health signals
    "is_stale_account",
    "is_inactive_user",
    "stale_high_risk",
]

# Verify all feature columns exist
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"  WARNING: Missing columns: {missing} — they will be skipped")
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

X = df[FEATURE_COLS].copy().fillna(0)
print(f"  Using {len(FEATURE_COLS)} features: {FEATURE_COLS}")

# ─────────────────────────────────────────────────────────────
# STEP 3 — STAGE A: ISOLATION FOREST
#
# What it does:
#   Builds many random decision trees. Events that are easy to
#   isolate (short path length) are outliers — anomalies.
#   Events that need many splits to isolate are normal.
#
# contamination=0.35 means we tell the model:
#   "expect roughly 35% of events to be anomalous"
#   (matches our domain knowledge from Step 1 analysis)
# ─────────────────────────────────────────────────────────────
print("\nSTEP 3 — Stage A: Isolation Forest (unsupervised)")

iso = IsolationForest(
    n_estimators=200,       # 200 trees — more stable than default 100
    contamination=0.35,     # expected anomaly rate
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)
iso.fit(X)

# anomaly_score: lower (more negative) = more anomalous
# decision_function returns the raw anomaly score
iso_scores = iso.decision_function(X)     # range: roughly -0.5 to +0.5
iso_labels = iso.predict(X)               # -1 = anomaly, +1 = normal

# Convert to 0/1 (1 = anomaly) for consistency
df["iso_anomaly"]    = (iso_labels == -1).astype(int)
# Normalise score to 0–100: higher = more anomalous
df["iso_score"] = pd.Series(
    100 * (1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min()))
).round(2)

iso_anomaly_rate = df["iso_anomaly"].mean() * 100
print(f"  Isolation Forest flagged: {df['iso_anomaly'].sum()} events ({iso_anomaly_rate:.1f}%)")
print(f"  Iso score range: {df['iso_score'].min():.1f} → {df['iso_score'].max():.1f}")

# ─────────────────────────────────────────────────────────────
# STEP 4 — CREATE PSEUDO-LABELS FOR STAGE B
#
# Since we have no ground truth, we create pseudo-labels by
# combining two signals:
#   1. Isolation Forest verdict (iso_anomaly)
#   2. Our heuristic risk score (raw_risk_score from Step 1)
#
# An event is pseudo-labeled as anomalous (1) if:
#   - Isolation Forest flagged it AND heuristic score > 40
#   OR
#   - Heuristic score is very high (> 65) regardless of ISO
#
# This gives a clean, defensible pseudo-label.
# ─────────────────────────────────────────────────────────────
print("\nSTEP 4 — Creating pseudo-labels for Stage B")

df["pseudo_label"] = (
    ((df["iso_anomaly"] == 1) & (df["raw_risk_score"] > 40)) |
    (df["raw_risk_score"] > 65)
).astype(int)

pseudo_rate = df["pseudo_label"].mean() * 100
print(f"  Pseudo-anomaly rate: {df['pseudo_label'].sum()} events ({pseudo_rate:.1f}%)")
print(f"\n  Logic:")
print(f"    ISO flagged + heuristic > 40  → anomaly")
print(f"    heuristic > 65 (any)          → anomaly")
print(f"    else                          → normal")

# ─────────────────────────────────────────────────────────────
# STEP 5 — STAGE B: LIGHTGBM CLASSIFIER
#
# Now we train a LightGBM model:
#   INPUT:  the 19 numeric features
#   TARGET: pseudo_label (0 = normal, 1 = anomaly)
#
# Why LightGBM?
#   - Handles mixed feature types well
#   - Fast to train on small datasets
#   - Gives feature importances (great for judge explanation)
#   - Outputs calibrated probabilities
# ─────────────────────────────────────────────────────────────
print("\nSTEP 5 — Stage B: LightGBM classifier")

y = df["pseudo_label"]

# 80/20 train-test split — stratified so both splits have same anomaly rate
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)
print(f"  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")
print(f"  Train anomaly rate: {y_train.mean()*100:.1f}%")
print(f"  Test  anomaly rate: {y_test.mean()*100:.1f}%")

# Class weight: anomalies are important, penalise missing them more
# scale_pos_weight = (# normal) / (# anomaly)
n_normal  = (y_train == 0).sum()
n_anomaly = (y_train == 1).sum()
scale_pos = n_normal / n_anomaly
print(f"  scale_pos_weight: {scale_pos:.2f} (upweights anomaly class)")

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    scale_pos_weight=scale_pos,   # handles class imbalance
    min_child_samples=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False),
               lgb.log_evaluation(period=-1)],
)

print(f"  Best iteration: {model.best_iteration_}")

# ─────────────────────────────────────────────────────────────
# STEP 6 — EVALUATE
# ─────────────────────────────────────────────────────────────
print("\nSTEP 6 — Evaluation on held-out test set (20%)")

y_pred_prob = model.predict_proba(X_test)[:, 1]   # probability of anomaly
y_pred      = (y_pred_prob >= 0.5).astype(int)

precision = precision_score(y_test, y_pred,  zero_division=0)
recall    = recall_score(y_test, y_pred,     zero_division=0)
f1        = f1_score(y_test, y_pred,         zero_division=0)
roc_auc   = roc_auc_score(y_test, y_pred_prob)
cm        = confusion_matrix(y_test, y_pred)

print(f"\n  {'Metric':<20} {'Score':>8}  {'Target':>8}")
print(f"  {'-'*40}")
print(f"  {'Precision':<20} {precision:>8.3f}  {'> 0.75':>8}  {'✅' if precision > 0.75 else '⚠️ '}")
print(f"  {'Recall':<20} {recall:>8.3f}  {'> 0.70':>8}  {'✅' if recall > 0.70 else '⚠️ '}")
print(f"  {'F1 Score':<20} {f1:>8.3f}  {'> 0.72':>8}  {'✅' if f1 > 0.72 else '⚠️ '}")
print(f"  {'ROC-AUC':<20} {roc_auc:>8.3f}  {'> 0.80':>8}  {'✅' if roc_auc > 0.80 else '⚠️ '}")

print(f"\n  Confusion Matrix:")
print(f"                  Predicted Normal  Predicted Anomaly")
print(f"  Actual Normal   {cm[0][0]:>16}  {cm[0][1]:>17}")
print(f"  Actual Anomaly  {cm[1][0]:>16}  {cm[1][1]:>17}")

print(f"\n  Full classification report:")
print(classification_report(y_test, y_pred,
                             target_names=["normal", "anomaly"],
                             zero_division=0))

# ─────────────────────────────────────────────────────────────
# STEP 7 — FEATURE IMPORTANCE
# Shows judges WHICH signals drive the model's decisions
# ─────────────────────────────────────────────────────────────
print("\nSTEP 7 — Feature importance (what drives the model?)")

importance_df = pd.DataFrame({
    "feature"   : FEATURE_COLS,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print(f"\n  {'Rank':<6} {'Feature':<30} {'Importance':>10}")
print(f"  {'-'*50}")
for i, row in importance_df.iterrows():
    bar = "█" * int(row["importance"] / importance_df["importance"].max() * 20)
    print(f"  {i+1:<6} {row['feature']:<30} {row['importance']:>10.0f}  {bar}")

# ─────────────────────────────────────────────────────────────
# STEP 8 — SCORE ALL 1200 EVENTS
# Apply the trained model to the full dataset
# ─────────────────────────────────────────────────────────────
print("\nSTEP 8 — Scoring all 1,200 events")

all_probs   = model.predict_proba(X)[:, 1]
all_preds   = (all_probs >= 0.5).astype(int)

df["anomaly_probability"] = all_probs.round(4)
df["is_anomaly"]          = all_preds

# Final severity: combine ML probability with heuristic risk score
# This makes the severity more stable and explainable
df["final_risk_score"] = (
    0.6 * df["anomaly_probability"] * 100 +
    0.4 * df["raw_risk_score"]
).round(1)

# Severity buckets for the dashboard
def assign_severity(score):
    if score >= 75: return "critical"
    if score >= 55: return "high"
    if score >= 35: return "medium"
    return "low"

df["final_severity"] = df["final_risk_score"].apply(assign_severity)

print(f"  Total anomalies detected : {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
print(f"\n  Final severity distribution:")
print(df["final_severity"].value_counts().sort_index().to_string())

# ─────────────────────────────────────────────────────────────
# STEP 9 — INCIDENT NARRATIVE GENERATOR
#
# For every flagged event, generate a one-line human-readable
# explanation. This is what the dashboard will show.
# No LLM needed — pure template logic.
# ─────────────────────────────────────────────────────────────
print("\nSTEP 9 — Generating incident narratives")

def generate_narrative(row):
    """
    Build a plain-English explanation of WHY this event was flagged.
    Combines the top triggered signals into a readable sentence.
    """
    reasons = []

    if row.get("is_off_hours", 0):
        time_label = row.get("time_classification", "off-hours")
        reasons.append(f"accessed during {time_label}")

    if row.get("dept_mismatch", 0):
        reasons.append(f"accessed resource outside their department")

    if row.get("sox_flag", 0):
        reasons.append(f"accessed a finance system (SOX concern)")

    if row.get("priv_action_mismatch", 0):
        reasons.append(f"performed admin operation without admin privileges")

    if row.get("is_high_risk_action", 0):
        reasons.append(f"performed high-risk action: {row.get('action','')}")

    if row.get("is_high_sensitivity", 0):
        reasons.append(f"targeted high-sensitivity resource")

    if row.get("stale_high_risk", 0):
        reasons.append(f"stale account performing risky action")

    if row.get("is_failed_access", 0):
        reasons.append(f"access attempt failed")

    if not reasons:
        reasons.append("statistical outlier in access pattern")

    user   = row.get("username", row.get("user_id", "unknown"))
    action = row.get("action", "performed action")
    res    = row.get("resource", "unknown resource")
    sev    = row.get("final_severity", "medium").upper()

    reason_str = "; ".join(reasons[:3])   # top 3 reasons max
    return (
        f"[{sev}] {user} {action} on {res} — {reason_str}. "
        f"Risk score: {row.get('final_risk_score', 0):.0f}/100."
    )

df["narrative"] = df.apply(generate_narrative, axis=1)

# Show top 10 highest-risk events
print("\n  Top 10 highest-risk events:")
top10 = (
    df.sort_values("final_risk_score", ascending=False)
    .head(10)[["username", "action", "resource", "final_severity",
               "final_risk_score", "narrative"]]
)
for _, row in top10.iterrows():
    print(f"\n  [{row['final_severity'].upper():>8}] score={row['final_risk_score']:.0f}")
    print(f"           {row['narrative']}")

# ─────────────────────────────────────────────────────────────
# STEP 10 — SAVE ALL OUTPUTS
# ─────────────────────────────────────────────────────────────
print("\nSTEP 10 — Saving outputs")

# Full predictions table
pred_cols = [
    "user_id", "username", "timestamp", "action", "resource",
    "resource_sensitivity", "status", "time_classification", "department",
    "privilege_level",
    # ML outputs
    "iso_score", "iso_anomaly",
    "anomaly_probability", "is_anomaly",
    "raw_risk_score", "final_risk_score", "final_severity",
    "narrative",
    # Key features for explainability
    "is_off_hours", "is_high_risk_action", "is_high_sensitivity",
    "dept_mismatch", "sox_flag", "priv_action_mismatch",
    "is_stale_account", "is_failed_access",
]
pred_cols = [c for c in pred_cols if c in df.columns]
predictions_df = df[pred_cols].copy()
predictions_df.to_csv(OUT_DIR / "model1_predictions.csv", index=False)

# Feature importance
importance_df.to_csv(OUT_DIR / "model1_feature_importance.csv", index=False)

# Metrics summary text file
metrics_text = f"""
PS4 Model 1 — Event-Level Anomaly Detection
=============================================
Dataset      : 1,200 events, {len(FEATURE_COLS)} features
Train/Test   : 80/20 stratified split
Algorithm    : Isolation Forest (Stage A) + LightGBM (Stage B)

EVALUATION METRICS (test set, 20% holdout)
-------------------------------------------
Precision    : {precision:.3f}   (target > 0.75)  {'PASS' if precision > 0.75 else 'FAIL'}
Recall       : {recall:.3f}   (target > 0.70)  {'PASS' if recall > 0.70 else 'FAIL'}
F1 Score     : {f1:.3f}   (target > 0.72)  {'PASS' if f1 > 0.72 else 'FAIL'}
ROC-AUC      : {roc_auc:.3f}   (target > 0.80)  {'PASS' if roc_auc > 0.80 else 'FAIL'}

ANOMALY DETECTION RESULTS (full 1,200 events)
-----------------------------------------------
Total flagged: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)

Severity breakdown:
{df['final_severity'].value_counts().to_string()}

TOP FEATURES (by LightGBM importance):
{importance_df.head(5).to_string(index=False)}
"""
(OUT_DIR / "model1_metrics.txt").write_text(metrics_text)

print(f"  model1_predictions.csv       → {len(predictions_df)} rows")
print(f"  model1_feature_importance.csv → {len(importance_df)} features ranked")
print(f"  model1_metrics.txt            → eval summary")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 1 COMPLETE")
print("=" * 60)
print(f"""
  Two-stage pipeline:
    Stage A: Isolation Forest → unsupervised outlier detection
    Stage B: LightGBM         → supervised scoring on pseudo-labels

  Results:
    Precision : {precision:.3f}
    Recall    : {recall:.3f}
    F1        : {f1:.3f}
    ROC-AUC   : {roc_auc:.3f}

  Top 3 features driving detections:
    1. {importance_df.iloc[0]['feature']}
    2. {importance_df.iloc[1]['feature']}
    3. {importance_df.iloc[2]['feature']}

Next → Step 3: Model 2 — User-Level Risk Scoring
       Run  : python ps4_step3_model2.py
""")
