"""
PS4 — Data Access Audit & Insider Threat Detection
Step 1: Load → Merge → Feature Engineering → Save clean feature table

REAL DATA COLUMNS (from actual provided CSVs):
-----------------------------------------------
data_access_logs.csv:
    timestamp, user_id, username, action, resource,
    resource_sensitivity, status, source_ip, time_classification

user_profiles.csv:
    user_id, username, email, department, job_title,
    privilege_level, systems_access, last_login,
    days_inactive, is_active, hire_date

PROBLEM WE ARE SOLVING IN THIS STEP:
--------------------------------------
Raw logs tell us WHAT happened — not WHETHER it is suspicious.
To judge suspicion we need context about the user:
  - Is this action normal for their role/department?
  - Is their account stale or over-privileged?
  - Is this resource outside their usual scope?
This script builds all those contextual features.

OUTPUT:
  outputs/features_events.csv  — 1 row per event,  ready for Model 1
  outputs/features_users.csv   — 1 row per user,   ready for Model 2
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DATA_DIR = Path("sample_data")
OUT_DIR  = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading real data")
print("=" * 60)

logs  = pd.read_csv(DATA_DIR / "data_access_logs.csv")
users = pd.read_csv(DATA_DIR / "user_profiles.csv")

print(f"  Access logs   : {logs.shape[0]} rows × {logs.shape[1]} cols")
print(f"  User profiles : {users.shape[0]} rows × {users.shape[1]} cols")

# ─────────────────────────────────────────────────────────────
# STEP 2 — CLEAN
# ─────────────────────────────────────────────────────────────
print("\nSTEP 2 — Cleaning")

logs["timestamp"]  = pd.to_datetime(logs["timestamp"], errors="coerce")
users["last_login"]= pd.to_datetime(users["last_login"], errors="coerce")
users["hire_date"] = pd.to_datetime(users["hire_date"],  errors="coerce")

# Standardise all string columns: lowercase + strip whitespace
for df in [logs, users]:
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip().str.lower()

# Drop rows where timestamp is missing — time features won't work without it
before = len(logs)
logs   = logs.dropna(subset=["timestamp"]).reset_index(drop=True)
print(f"  Dropped {before - len(logs)} rows with unparseable timestamps")
print(f"  Log date range: {logs['timestamp'].min().date()} → {logs['timestamp'].max().date()}")
print(f"  Unique users in logs    : {logs['user_id'].nunique()}")
print(f"  Unique users in profiles: {users['user_id'].nunique()}")

# ─────────────────────────────────────────────────────────────
# STEP 3 — MERGE
# Problem: log rows have no department, privilege, or tenure info.
# Solution: join user_profiles onto every log event via user_id.
# ─────────────────────────────────────────────────────────────
print("\nSTEP 3 — Merging logs ← user profiles")

df = logs.merge(
    users[["user_id","department","job_title","privilege_level",
           "systems_access","last_login","days_inactive",
           "is_active","hire_date"]],
    on="user_id",
    how="left"
)

no_profile = df["department"].isna().sum()
print(f"  Events with no matching profile : {no_profile} ({100*no_profile/len(df):.1f}%)")

df["department"]     = df["department"].fillna("unknown")
df["privilege_level"]= df["privilege_level"].fillna("user")
df["days_inactive"]  = df["days_inactive"].fillna(0)
df["is_active"]      = df["is_active"].fillna(True)

print(f"  Merged shape: {df.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 4 — TIME FEATURES
# Problem: Is this access outside the user's normal working hours?
# We use time_classification directly (already in the data) +
# extract hour and day for extra signal.
# ─────────────────────────────────────────────────────────────
print("\nSTEP 4 — Time features")

df["hour"]        = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Mon, 6=Sun
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

# time_classification column already exists in the data:
#   business_hours | unusual_hours | night | weekend
# Encode it as a risk score (higher = more suspicious timing)
time_risk_map = {
    "business_hours": 0,
    "unusual_hours" : 1,
    "weekend"       : 1,
    "night"         : 2,
}
df["time_risk"] = df["time_classification"].map(time_risk_map).fillna(1)

# Binary flag: anything that is NOT business_hours
df["is_off_hours"] = (df["time_classification"] != "business_hours").astype(int)

print(f"  Off-hours events : {df['is_off_hours'].sum()} ({100*df['is_off_hours'].mean():.1f}%)")
print(f"  Night events     : {(df['time_classification']=='night').sum()}")
print(f"  Weekend events   : {(df['time_classification']=='weekend').sum()}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — ACTION RISK FEATURES
# Problem: Not all actions are equal — export_data is riskier than login.
# Solution: Assign a risk weight per action type.
# ─────────────────────────────────────────────────────────────
print("\nSTEP 5 — Action risk features")

action_risk_map = {
    "login"           : 0,   # normal, baseline
    "api_call"        : 1,   # programmatic — slightly elevated
    "sql_query"       : 1,   # data read
    "file_access"     : 2,   # direct file access
    "export_data"     : 3,   # data leaving the system — high risk
    "admin_operation" : 3,   # privilege action — high risk
}
df["action_risk"] = df["action"].map(action_risk_map).fillna(1)

# High-risk action flag: export or admin
df["is_high_risk_action"] = (df["action"].isin(["export_data","admin_operation"])).astype(int)

print(f"  Action distribution:\n{df['action'].value_counts().to_string()}")
print(f"  High-risk actions : {df['is_high_risk_action'].sum()} ({100*df['is_high_risk_action'].mean():.1f}%)")

# ─────────────────────────────────────────────────────────────
# STEP 6 — SENSITIVITY FEATURES
# Problem: Accessing 'high' sensitivity data needs more scrutiny.
# Solution: Encode resource_sensitivity as a numeric score 1–3.
# This directly maps to GDPR Article 32 — protecting personal data.
# ─────────────────────────────────────────────────────────────
print("\nSTEP 6 — Sensitivity features")

sensitivity_map = {"low": 1, "medium": 2, "high": 3}
df["sensitivity_score"] = df["resource_sensitivity"].map(sensitivity_map).fillna(2).astype(int)
df["is_high_sensitivity"] = (df["sensitivity_score"] == 3).astype(int)

print(f"  High-sensitivity events : {df['is_high_sensitivity'].sum()} ({100*df['is_high_sensitivity'].mean():.1f}%)")

# ─────────────────────────────────────────────────────────────
# STEP 7 — FAILED ACCESS FEATURES
# Problem: Repeated failures before a success = credential stuffing / brute force.
# Solution: Flag failed access attempts, especially on sensitive resources.
# ─────────────────────────────────────────────────────────────
print("\nSTEP 7 — Failed access features")

df["is_failed_access"] = (df["status"] == "failure").astype(int)

# Per-user failure rate (how often does this user fail?)
user_failure = (
    df.groupby("user_id")["is_failed_access"]
    .mean()
    .rename("user_failure_rate")
    .reset_index()
)
df = df.merge(user_failure, on="user_id", how="left")

# Suspicious: failed access on a high-sensitivity resource
df["failed_high_sens"] = (
    (df["is_failed_access"] == 1) & (df["is_high_sensitivity"] == 1)
).astype(int)

print(f"  Total failures          : {df['is_failed_access'].sum()}")
print(f"  Failed on high-sens     : {df['failed_high_sens'].sum()}")

# ─────────────────────────────────────────────────────────────
# STEP 8 — RESOURCE-DEPARTMENT MISMATCH
# Problem: A Sales user accessing HR or Finance systems = suspicious.
# Solution: Map resources to their owning department, compare with user dept.
# This is the SOX 302 signal — unauthorized GL/AR/AP access.
# ─────────────────────────────────────────────────────────────
print("\nSTEP 8 — Resource-department mismatch (SOX signal)")

# Map each resource name to its owning department
# Based on the actual resource values in the data
resource_dept_map = {
    "gl_system"        : "finance",
    "payroll_db"       : "finance",
    "ar_system"        : "finance",
    "ap_system"        : "finance",
    "hris"             : "hr",
    "hr_portal"        : "hr",
    "employee_records" : "hr",
    "admin_console"    : "it",
    "server_logs"      : "it",
    "network_config"   : "it",
    "customer_vault"   : "sales",
    "crm"              : "sales",
    "sales_db"         : "sales",
    "bi_tool"          : "analytics",
    "file_share"       : "general",     # accessible to all
    "email"            : "general",
    "sharepoint"       : "general",
}

df["resource_lower"]  = df["resource"].str.lower().str.replace(" ", "_")
df["resource_dept"]   = df["resource_lower"].map(resource_dept_map).fillna("general")

# Mismatch: user's dept ≠ resource's dept AND resource is not 'general'
df["dept_mismatch"] = (
    (df["resource_dept"] != "general") &
    (df["resource_dept"] != "unknown") &
    (df["department"].str.lower() != df["resource_dept"])
).astype(int)

# SOX flag: specifically accessing finance systems from a non-finance user
df["sox_flag"] = (
    (df["resource_dept"] == "finance") &
    (df["department"].str.lower() != "finance")
).astype(int)

print(f"  Dept mismatch events : {df['dept_mismatch'].sum()} ({100*df['dept_mismatch'].mean():.1f}%)")
print(f"  SOX flags (non-finance → finance resource): {df['sox_flag'].sum()}")

# ─────────────────────────────────────────────────────────────
# STEP 9 — PRIVILEGE FEATURES
# Problem: An 'admin' user doing bulk exports at night = very suspicious.
#          A 'user' with admin_operations = privilege escalation attempt.
# ─────────────────────────────────────────────────────────────
print("\nSTEP 9 — Privilege features")

priv_score_map = {"user": 1, "power-user": 2, "service-account": 2, "admin": 3}
df["privilege_score"] = df["privilege_level"].map(priv_score_map).fillna(1)

# Privilege-action mismatch: low-privilege user doing admin operations
df["priv_action_mismatch"] = (
    (df["privilege_score"] == 1) & (df["action"] == "admin_operation")
).astype(int)

print(f"  Privilege-action mismatch : {df['priv_action_mismatch'].sum()}")

# ─────────────────────────────────────────────────────────────
# STEP 10 — USER ACTIVITY BASELINE (per-user event counts)
# Problem: Unusual volume of activity for a specific user is a red flag.
# Solution: Count events per user and compare each user to the group mean.
# ─────────────────────────────────────────────────────────────
print("\nSTEP 10 — User activity baseline")

user_activity = (
    df.groupby("user_id")
    .agg(
        total_events          = ("user_id",           "count"),
        high_risk_action_count= ("is_high_risk_action","sum"),
        off_hours_count       = ("is_off_hours",       "sum"),
        failure_count         = ("is_failed_access",   "sum"),
        dept_mismatch_count   = ("dept_mismatch",      "sum"),
        high_sens_count       = ("is_high_sensitivity","sum"),
        sox_flag_count        = ("sox_flag",           "sum"),
        priv_mismatch_count   = ("priv_action_mismatch","sum"),
        unique_resources      = ("resource",           "nunique"),
    )
    .reset_index()
)

# Z-score of total events: how active is this user vs others?
mean_ev = user_activity["total_events"].mean()
std_ev  = user_activity["total_events"].std()
user_activity["activity_z_score"] = (
    (user_activity["total_events"] - mean_ev) / std_ev
).round(3)

# Ratio features (normalised by total events — comparable across users)
user_activity["high_risk_action_ratio"] = (
    user_activity["high_risk_action_count"] / user_activity["total_events"]
).round(3)
user_activity["off_hours_ratio"] = (
    user_activity["off_hours_count"] / user_activity["total_events"]
).round(3)
user_activity["failure_ratio"] = (
    user_activity["failure_count"] / user_activity["total_events"]
).round(3)
user_activity["dept_mismatch_ratio"] = (
    user_activity["dept_mismatch_count"] / user_activity["total_events"]
).round(3)

print(f"  User activity table shape : {user_activity.shape}")
print(f"  Avg events per user       : {mean_ev:.1f}  (std={std_ev:.1f})")

# ─────────────────────────────────────────────────────────────
# STEP 11 — STALE / INACTIVE ACCOUNT FEATURES
# Problem: Dormant accounts being used = orphaned / compromised credential.
# Solution: Use days_inactive and is_active from user_profiles.
# ─────────────────────────────────────────────────────────────
print("\nSTEP 11 — Stale account features")

# days_inactive threshold: >30 days = stale, >60 = very stale
df["is_stale_account"] = (df["days_inactive"] > 30).astype(int)
df["is_inactive_user"] = (~df["is_active"].astype(str).str.lower().isin(["true","1"])).astype(int)

# Stale + high-risk action = the most dangerous combo
df["stale_high_risk"] = (
    (df["is_stale_account"] == 1) & (df["is_high_risk_action"] == 1)
).astype(int)

print(f"  Stale account events     : {df['is_stale_account'].sum()} ({100*df['is_stale_account'].mean():.1f}%)")
print(f"  Inactive user events     : {df['is_inactive_user'].sum()}")
print(f"  Stale + high-risk combo  : {df['stale_high_risk'].sum()}")

# ─────────────────────────────────────────────────────────────
# STEP 12 — COMPOSITE RISK SCORE (heuristic, pre-ML)
# This gives every event a 0–100 suspicion score using weighted signals.
# Useful for: explaining alerts, sanity-checking the ML model,
#             and showing in the dashboard without ML.
# ─────────────────────────────────────────────────────────────
print("\nSTEP 12 — Composite heuristic risk score")

df["raw_risk_score"] = (
    df["time_risk"]           * 10 +   # off-hours timing
    df["action_risk"]         * 15 +   # risky action type
    df["sensitivity_score"]   * 10 +   # sensitive resource
    df["dept_mismatch"]       * 20 +   # wrong department
    df["sox_flag"]            * 25 +   # finance system access
    df["priv_action_mismatch"]* 20 +   # privilege escalation
    df["is_failed_access"]    * 10 +   # failed attempt
    df["failed_high_sens"]    * 15 +   # failed on sensitive resource
    df["stale_high_risk"]     * 20     # stale account doing risky thing
).clip(0, 100)

severity_bins   = [  0,  20,  40,  65, 100]
severity_labels = ["low","medium","high","critical"]
df["severity_heuristic"] = pd.cut(
    df["raw_risk_score"],
    bins=severity_bins,
    labels=severity_labels,
    include_lowest=True,
)

print(f"  Severity distribution:")
print(df["severity_heuristic"].value_counts().sort_index().to_string())

# ─────────────────────────────────────────────────────────────
# STEP 13 — ASSEMBLE FINAL FEATURE TABLES
# ─────────────────────────────────────────────────────────────
print("\nSTEP 13 — Assembling final feature tables")

# ── Event-level feature table ──
event_feature_cols = [
    # identifiers
    "user_id", "username", "timestamp", "action", "resource",
    # raw context
    "resource_sensitivity", "status", "time_classification", "department",
    "privilege_level",
    # engineered features
    "hour", "day_of_week", "is_weekend",
    "time_risk", "is_off_hours",
    "action_risk", "is_high_risk_action",
    "sensitivity_score", "is_high_sensitivity",
    "is_failed_access", "user_failure_rate", "failed_high_sens",
    "dept_mismatch", "sox_flag",
    "privilege_score", "priv_action_mismatch",
    "is_stale_account", "is_inactive_user", "stale_high_risk",
    "raw_risk_score", "severity_heuristic",
]
event_feature_cols = [c for c in event_feature_cols if c in df.columns]
events_out = df[event_feature_cols].copy()

# ── User-level feature table ──
user_feature_cols = [
    "user_id", "department", "job_title", "privilege_level",
    "days_inactive", "is_active",
    "total_events", "high_risk_action_count", "off_hours_count",
    "failure_count", "dept_mismatch_count", "high_sens_count",
    "sox_flag_count", "priv_mismatch_count", "unique_resources",
    "activity_z_score", "high_risk_action_ratio", "off_hours_ratio",
    "failure_ratio", "dept_mismatch_ratio",
]
users_full = users.merge(user_activity, on="user_id", how="left")
# Add stale account info from user profiles
users_full["is_stale_account"] = (users_full["days_inactive"] > 30).astype(int)
user_feature_cols = [c for c in user_feature_cols if c in users_full.columns]
users_out = users_full[user_feature_cols + ["is_stale_account"]].copy()

# ─────────────────────────────────────────────────────────────
# STEP 14 — SAVE
# ─────────────────────────────────────────────────────────────
print("\nSTEP 14 — Saving")

events_path = OUT_DIR / "features_events.csv"
users_path  = OUT_DIR / "features_users.csv"

events_out.to_csv(events_path, index=False)
users_out.to_csv(users_path,   index=False)

print(f"  features_events.csv → {events_out.shape[0]} rows × {events_out.shape[1]} cols")
print(f"  features_users.csv  → {users_out.shape[0]} rows × {users_out.shape[1]} cols")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 COMPLETE — Feature Engineering Summary")
print("=" * 60)

features_built = {
    "Time"       : ["hour","day_of_week","is_weekend","time_risk","is_off_hours"],
    "Action"     : ["action_risk","is_high_risk_action"],
    "Sensitivity": ["sensitivity_score","is_high_sensitivity"],
    "Failure"    : ["is_failed_access","user_failure_rate","failed_high_sens"],
    "Dept/SOX"   : ["dept_mismatch","sox_flag"],
    "Privilege"  : ["privilege_score","priv_action_mismatch"],
    "Stale acct" : ["is_stale_account","is_inactive_user","stale_high_risk"],
    "Risk score" : ["raw_risk_score","severity_heuristic"],
}

for category, feats in features_built.items():
    print(f"  {category:<12}: {', '.join(feats)}")

print(f"\n  Total event features  : {events_out.shape[1]}")
print(f"  Total user features   : {users_out.shape[1]}")
print(f"\nNext → Step 2: Train Model 1 (event-level anomaly detector)")
print("       Run  : python ps4_step2_model1.py")
