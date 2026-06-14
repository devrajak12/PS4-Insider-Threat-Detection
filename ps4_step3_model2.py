"""
PS4 — Data Access Audit & Insider Threat Detection
Step 3: Model 2 — User-Level Risk Scoring

PROBLEM BEING SOLVED IN THIS STEP:
------------------------------------
Model 1 told us WHICH EVENTS are anomalous.
But security teams don't investigate 600 individual events — they investigate PEOPLE.

The question now is:
    "Across all 100 users, which 10 should my security team look at first?"

A user is risky if they have a PATTERN of bad behaviour — not just one odd event.
  - A user with 10 off-hours exports is riskier than one with 1.
  - A stale admin account is riskier than an active standard user.
  - A user whose ML anomaly rate is 90% needs immediate attention.

This step:
  1. Enriches the user feature table with per-user ML signals from Model 1
  2. Trains an Isolation Forest on user-level features (again unsupervised)
  3. Produces a final USER RISK SCORE (0–100) for every user
  4. Classifies each user: normal / elevated / high / critical
  5. Generates a user risk report with the top 10 riskiest users

INPUTS:
  outputs/features_users.csv        (from Step 1)
  outputs/model1_predictions.csv    (from Step 2)

OUTPUTS:
  outputs/model2_user_risk.csv      — all 100 users with risk scores
  outputs/model2_top10_report.txt   — human-readable top-10 report
  outputs/model2_metrics.txt        — evaluation summary
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD BOTH INPUTS
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("MODEL 2 — User-Level Risk Scoring")
print("=" * 60)

users_df = pd.read_csv(OUT_DIR / "features_users.csv")
pred_df  = pd.read_csv(OUT_DIR / "model1_predictions.csv")

print(f"\n  User feature table  : {users_df.shape[0]} users × {users_df.shape[1]} cols")
print(f"  Model 1 predictions : {pred_df.shape[0]} events × {pred_df.shape[1]} cols")

# ─────────────────────────────────────────────────────────────
# STEP 2 — AGGREGATE MODEL 1 RESULTS PER USER
#
# Problem: Model 1 gave us event-level signals.
#          We need to summarise them at the user level.
#
# For each user we compute:
#   - How many of their events were flagged as anomalous?
#   - What is their average / max anomaly probability?
#   - How many critical events do they have?
#   - What is their highest single-event risk score?
# ─────────────────────────────────────────────────────────────
print("\nSTEP 2 — Aggregating Model 1 signals per user")

user_ml = (
    pred_df.groupby("user_id")
    .agg(
        ml_anomaly_count  = ("is_anomaly",          "sum"),
        ml_anomaly_rate   = ("is_anomaly",           "mean"),
        avg_anomaly_prob  = ("anomaly_probability",  "mean"),
        max_anomaly_prob  = ("anomaly_probability",  "max"),
        avg_final_risk    = ("final_risk_score",     "mean"),
        max_final_risk    = ("final_risk_score",     "max"),
        critical_events   = ("final_severity",       lambda x: (x == "critical").sum()),
        high_events       = ("final_severity",       lambda x: (x == "high").sum()),
        # Pull most recent narrative for context
        worst_narrative   = ("narrative",            "first"),
    )
    .reset_index()
)

# Also get the username from predictions (not in user profile)
usernames = pred_df[["user_id","username"]].drop_duplicates("user_id")
user_ml   = user_ml.merge(usernames, on="user_id", how="left")

print(f"  Aggregated ML stats for {len(user_ml)} users")
print(f"  Avg anomaly rate per user : {user_ml['ml_anomaly_rate'].mean()*100:.1f}%")
print(f"  Users with >70% anomaly rate: {(user_ml['ml_anomaly_rate'] > 0.7).sum()}")

# ─────────────────────────────────────────────────────────────
# STEP 3 — MERGE USER FEATURES + ML SIGNALS
# ─────────────────────────────────────────────────────────────
print("\nSTEP 3 — Merging user features with ML signals")

df = users_df.merge(user_ml, on="user_id", how="left")
print(f"  Merged user table : {df.shape[0]} users × {df.shape[1]} cols")

# ─────────────────────────────────────────────────────────────
# STEP 4 — ENCODE CATEGORICAL FEATURES
#
# Isolation Forest needs all-numeric input.
# Encode privilege_level and department as numbers.
# ─────────────────────────────────────────────────────────────
print("\nSTEP 4 — Encoding categorical features")

# Privilege level → ordered numeric (higher = more powerful = higher risk potential)
priv_map = {"user": 1, "power-user": 2, "service-account": 2, "admin": 3}
df["privilege_score"] = df["privilege_level"].map(priv_map).fillna(1)

# is_active: already bool — convert to int
df["is_active_int"] = df["is_active"].astype(str).str.lower().isin(["true","1"]).astype(int)

# Department risk weight:
# Finance/Executive/IT departments have access to more sensitive systems
dept_risk_map = {
    "finance"    : 3,
    "executive"  : 3,
    "it"         : 3,
    "security"   : 2,
    "legal"      : 2,
    "compliance" : 2,
    "hr"         : 2,
    "engineering": 2,
    "sales"      : 1,
    "marketing"  : 1,
    "support"    : 1,
    "operations" : 1,
}
df["dept_risk_weight"] = df["department"].map(dept_risk_map).fillna(1)

print(f"  privilege_score distribution : {df['privilege_score'].value_counts().to_dict()}")
print(f"  dept_risk_weight distribution: {df['dept_risk_weight'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — DERIVED RISK INDICATORS
#
# These compound features are more predictive than raw counts alone.
# Each one captures a specific insider threat pattern.
# ─────────────────────────────────────────────────────────────
print("\nSTEP 5 — Derived risk indicators")

# Stale + privileged = dormant admin account (classic attack vector)
df["stale_privileged"] = (
    (df["is_stale_account"] == 1) & (df["privilege_score"] >= 2)
).astype(int)

# High ML anomaly rate + many events = active bad actor (not just noise)
df["active_anomalous"] = (
    (df["ml_anomaly_rate"] > 0.6) & (df["total_events"] > 8)
).astype(int)

# SOX violator: any non-finance user accessing finance systems
df["is_sox_risk"] = (df["sox_flag_count"] > 0).astype(int)

# Privilege abuse: standard user doing admin operations
df["is_priv_abuser"] = (df["priv_mismatch_count"] > 0).astype(int)

# Cross-department explorer: accessing many different resources
# (Data exfiltration often involves accessing breadth of systems)
resource_mean = df["unique_resources"].mean()
resource_std  = df["unique_resources"].std()
df["resource_breadth_z"] = (
    (df["unique_resources"] - resource_mean) / resource_std
).round(3)
df["is_wide_accessor"] = (df["resource_breadth_z"] > 1.5).astype(int)

print(f"  Stale + privileged accounts  : {df['stale_privileged'].sum()}")
print(f"  Active anomalous users       : {df['active_anomalous'].sum()}")
print(f"  SOX risk users               : {df['is_sox_risk'].sum()}")
print(f"  Privilege abusers            : {df['is_priv_abuser'].sum()}")
print(f"  Wide accessors (>1.5σ)       : {df['is_wide_accessor'].sum()}")

# ─────────────────────────────────────────────────────────────
# STEP 6 — SELECT FEATURES FOR MODEL 2
# ─────────────────────────────────────────────────────────────
print("\nSTEP 6 — Selecting model features")

USER_FEATURE_COLS = [
    # Static profile signals
    "days_inactive",
    "privilege_score",
    "dept_risk_weight",
    "is_stale_account",
    "is_active_int",

    # Behavioural counts (from Step 1 feature engineering)
    "total_events",
    "high_risk_action_count",
    "off_hours_count",
    "failure_count",
    "dept_mismatch_count",
    "sox_flag_count",
    "priv_mismatch_count",
    "unique_resources",

    # Behavioural ratios (normalised — comparable across users)
    "high_risk_action_ratio",
    "off_hours_ratio",
    "failure_ratio",
    "dept_mismatch_ratio",
    "activity_z_score",

    # ML signals from Model 1
    "ml_anomaly_rate",
    "avg_anomaly_prob",
    "max_anomaly_prob",
    "avg_final_risk",
    "max_final_risk",
    "critical_events",
    "high_events",

    # Derived compound indicators
    "stale_privileged",
    "active_anomalous",
    "is_sox_risk",
    "is_priv_abuser",
    "is_wide_accessor",
    "resource_breadth_z",
]

USER_FEATURE_COLS = [c for c in USER_FEATURE_COLS if c in df.columns]
X_user = df[USER_FEATURE_COLS].fillna(0)

print(f"  Using {len(USER_FEATURE_COLS)} features")

# ─────────────────────────────────────────────────────────────
# STEP 7 — ISOLATION FOREST ON USER FEATURES
#
# Same logic as Model 1 — unsupervised outlier detection.
# With only 100 users, Isolation Forest is ideal.
# contamination=0.20: expect ~20% of users to be risky
# (consistent with typical insider threat prevalence)
# ─────────────────────────────────────────────────────────────
print("\nSTEP 7 — Isolation Forest on user features")

iso_user = IsolationForest(
    n_estimators=300,
    contamination=0.20,
    max_samples=min(50, len(X_user)),   # small dataset — use all samples
    random_state=42,
    n_jobs=-1,
)
iso_user.fit(X_user)

iso_scores_user = iso_user.decision_function(X_user)
iso_labels_user = iso_user.predict(X_user)

df["iso_anomaly_user"] = (iso_labels_user == -1).astype(int)
# Normalise to 0-100: higher = more anomalous
df["iso_risk_score"] = pd.Series(
    100 * (1 - (iso_scores_user - iso_scores_user.min()) /
               (iso_scores_user.max() - iso_scores_user.min()))
).round(2)

print(f"  ISO flagged {df['iso_anomaly_user'].sum()} users as anomalous "
      f"({df['iso_anomaly_user'].mean()*100:.0f}%)")

# ─────────────────────────────────────────────────────────────
# STEP 8 — COMPOSITE USER RISK SCORE
#
# Combine multiple signals into one final score:
#   40% — Model 1 ML signal (avg anomaly probability across events)
#   30% — Isolation Forest user-level score
#   30% — Heuristic weighted sum of key risk indicators
#
# This weighted blend is more robust than any single signal.
# ─────────────────────────────────────────────────────────────
print("\nSTEP 8 — Computing composite user risk score")

# Heuristic component — weighted sum of key risk flags
df["heuristic_user_risk"] = (
    df["is_stale_account"]       * 15 +
    df["stale_privileged"]       * 20 +
    df["active_anomalous"]       * 25 +
    df["is_sox_risk"]            * 20 +
    df["is_priv_abuser"]         * 15 +
    df["is_wide_accessor"]       * 10 +
    df["high_risk_action_ratio"] * 30 +
    df["failure_ratio"]          * 20 +
    df["off_hours_ratio"]        * 10 +
    (df["privilege_score"] - 1)  * 10   # admin = +20, standard = 0
).clip(0, 100)

# Normalise ML anomaly signal to 0-100
ml_signal = (df["avg_anomaly_prob"] * 100).clip(0, 100)

# Final composite score
df["user_risk_score"] = (
    0.40 * ml_signal                   +
    0.30 * df["iso_risk_score"]        +
    0.30 * df["heuristic_user_risk"]
).round(1).clip(0, 100)

# User risk tier
def risk_tier(score):
    if score >= 75: return "critical"
    if score >= 55: return "high"
    if score >= 35: return "elevated"
    return "normal"

df["user_risk_tier"] = df["user_risk_score"].apply(risk_tier)

print(f"\n  User risk tier distribution:")
tier_counts = df["user_risk_tier"].value_counts()
for tier in ["critical","high","elevated","normal"]:
    count = tier_counts.get(tier, 0)
    bar   = "█" * count
    print(f"    {tier:<10}: {count:>3}  {bar}")

print(f"\n  Risk score range: {df['user_risk_score'].min():.1f} → "
      f"{df['user_risk_score'].max():.1f}  (mean={df['user_risk_score'].mean():.1f})")

# ─────────────────────────────────────────────────────────────
# STEP 9 — USER RISK NARRATIVE GENERATOR
#
# For each user, explain in plain English WHY they are risky.
# This is what will appear on the dashboard user-profile card.
# ─────────────────────────────────────────────────────────────
print("\nSTEP 9 — Generating user risk narratives")

def user_risk_narrative(row):
    """
    Build a plain-English risk summary for a user.
    Prioritises the most severe signals.
    """
    tier   = row["user_risk_tier"].upper()
    name   = row.get("username", row["user_id"])
    dept   = row.get("department", "unknown").title()
    role   = row.get("job_title",  "unknown").title()
    score  = row["user_risk_score"]

    flags = []

    if row.get("stale_privileged", 0):
        flags.append(
            f"stale privileged account inactive for {int(row['days_inactive'])} days"
        )
    elif row.get("is_stale_account", 0):
        flags.append(f"account inactive for {int(row['days_inactive'])} days")

    if row.get("active_anomalous", 0):
        flags.append(
            f"{row['ml_anomaly_rate']*100:.0f}% of their events flagged as anomalous"
        )

    if row.get("is_sox_risk", 0):
        flags.append(
            f"{int(row['sox_flag_count'])} unauthorized finance system access(es)"
        )

    if row.get("is_priv_abuser", 0):
        flags.append(
            f"{int(row['priv_mismatch_count'])} admin operation(s) without admin rights"
        )

    if row.get("is_wide_accessor", 0):
        flags.append(
            f"accessed {int(row['unique_resources'])} different resources "
            f"({row['resource_breadth_z']:.1f}σ above average)"
        )

    if row.get("critical_events", 0) > 3:
        flags.append(f"{int(row['critical_events'])} critical-severity events")

    if row.get("off_hours_ratio", 0) > 0.4:
        flags.append(
            f"{row['off_hours_ratio']*100:.0f}% of activity outside business hours"
        )

    if not flags:
        flags.append("elevated statistical anomaly score across multiple signals")

    flag_str = "; ".join(flags[:3])
    return (
        f"[{tier}] {name} ({role}, {dept}) — Risk score: {score:.0f}/100. "
        f"Flags: {flag_str}."
    )

df["user_narrative"] = df.apply(user_risk_narrative, axis=1)

# ─────────────────────────────────────────────────────────────
# STEP 10 — TOP 10 RISKIEST USERS
# ─────────────────────────────────────────────────────────────
print("\nSTEP 10 — Top 10 riskiest users")
print("=" * 60)

top10_users = (
    df.sort_values("user_risk_score", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

for rank, row in top10_users.iterrows():
    print(f"\n  #{rank+1}  {row['user_narrative']}")
    print(f"       ML anomaly rate: {row['ml_anomaly_rate']*100:.0f}%  |  "
          f"ISO score: {row['iso_risk_score']:.0f}  |  "
          f"Max event risk: {row['max_final_risk']:.0f}")

# ─────────────────────────────────────────────────────────────
# STEP 11 — SAVE ALL OUTPUTS
# ─────────────────────────────────────────────────────────────
print("\n\nSTEP 11 — Saving outputs")

# Full user risk table
output_cols = [
    "user_id", "username", "department", "job_title", "privilege_level",
    "days_inactive", "is_active",
    # behaviour counts
    "total_events", "ml_anomaly_count", "critical_events", "high_events",
    "sox_flag_count", "priv_mismatch_count", "off_hours_count", "failure_count",
    # ratios
    "ml_anomaly_rate", "high_risk_action_ratio", "off_hours_ratio",
    "dept_mismatch_ratio", "failure_ratio",
    # risk flags
    "is_stale_account", "stale_privileged", "active_anomalous",
    "is_sox_risk", "is_priv_abuser", "is_wide_accessor",
    # scores
    "iso_risk_score", "heuristic_user_risk", "user_risk_score",
    "user_risk_tier",
    # narrative
    "user_narrative",
]
output_cols = [c for c in output_cols if c in df.columns]
user_risk_df = df[output_cols].sort_values("user_risk_score", ascending=False)
user_risk_df.to_csv(OUT_DIR / "model2_user_risk.csv", index=False)

# Human-readable top-10 report
report_lines = [
    "PS4 — USER RISK REPORT",
    "=" * 60,
    f"Total users analysed : {len(df)}",
    f"Critical risk users  : {(df['user_risk_tier']=='critical').sum()}",
    f"High risk users      : {(df['user_risk_tier']=='high').sum()}",
    f"Elevated risk users  : {(df['user_risk_tier']=='elevated').sum()}",
    f"Normal users         : {(df['user_risk_tier']=='normal').sum()}",
    "",
    "TOP 10 USERS REQUIRING IMMEDIATE INVESTIGATION",
    "-" * 60,
]
for rank, row in top10_users.iterrows():
    report_lines += [
        f"",
        f"RANK #{rank+1}",
        f"  {row['user_narrative']}",
        f"  Total events    : {int(row['total_events'])}",
        f"  ML anomaly rate : {row['ml_anomaly_rate']*100:.0f}%",
        f"  Critical events : {int(row['critical_events'])}",
        f"  SOX flags       : {int(row['sox_flag_count'])}",
        f"  Priv mismatches : {int(row['priv_mismatch_count'])}",
        f"  Days inactive   : {int(row['days_inactive'])}",
    ]

(OUT_DIR / "model2_top10_report.txt").write_text("\n".join(report_lines))

# Metrics summary
metrics_lines = [
    "PS4 Model 2 — User-Level Risk Scoring",
    "=" * 60,
    f"Users analysed  : {len(df)}",
    f"Features used   : {len(USER_FEATURE_COLS)}",
    f"Method          : Isolation Forest (user-level) + composite scoring",
    "",
    "RISK TIER DISTRIBUTION",
    "-" * 30,
    f"Critical : {(df['user_risk_tier']=='critical').sum()}",
    f"High     : {(df['user_risk_tier']=='high').sum()}",
    f"Elevated : {(df['user_risk_tier']=='elevated').sum()}",
    f"Normal   : {(df['user_risk_tier']=='normal').sum()}",
    "",
    "SCORE STATISTICS",
    "-" * 30,
    f"Min   : {df['user_risk_score'].min():.1f}",
    f"Max   : {df['user_risk_score'].max():.1f}",
    f"Mean  : {df['user_risk_score'].mean():.1f}",
    f"Std   : {df['user_risk_score'].std():.1f}",
    "",
    "COMPOSITE SCORE FORMULA",
    "-" * 30,
    "user_risk_score =",
    "  0.40 × avg_anomaly_probability (Model 1 ML signal)",
    "  0.30 × iso_risk_score (Isolation Forest user-level)",
    "  0.30 × heuristic_user_risk (weighted flag sum)",
]
(OUT_DIR / "model2_metrics.txt").write_text("\n".join(metrics_lines))

print(f"  model2_user_risk.csv    → {len(user_risk_df)} users, {len(output_cols)} cols")
print(f"  model2_top10_report.txt → human-readable investigation list")
print(f"  model2_metrics.txt      → scoring methodology")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 2 COMPLETE")
print("=" * 60)
print(f"""
  Pipeline:
    Input   : 100 users × {len(USER_FEATURE_COLS)} features
              (profile + behavioural + Model 1 ML signals)
    Method  : Isolation Forest (user-level) + composite scoring
    Output  : User risk score (0–100) + tier + narrative

  Risk tier breakdown:
    Critical : {(df['user_risk_tier']=='critical').sum()} users
    High     : {(df['user_risk_tier']=='high').sum()} users
    Elevated : {(df['user_risk_tier']=='elevated').sum()} users
    Normal   : {(df['user_risk_tier']=='normal').sum()} users

  Score range: {df['user_risk_score'].min():.1f} → {df['user_risk_score'].max():.1f}

Next → Step 4: Streamlit Dashboard
       Run  : python ps4_step4_dashboard.py
""")
