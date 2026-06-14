# PS4 — Data Access Audit & Insider Threat Detection
### Societe Generale Hackathon Submission

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your CSV files in sample_data/
#    sample_data/data_access_logs.csv
#    sample_data/user_profiles.csv

# 3. Run full pipeline
python run_all.py

# 4. Launch dashboard
streamlit run ps4_step4_dashboard.py
```

Open **http://localhost:8501** in your browser.

---

## What This System Does

Detects insider threats and data access anomalies from raw access logs using a two-stage ML pipeline:

- **Stage A:** Isolation Forest (unsupervised) — learns what normal looks like
- **Stage B:** LightGBM (supervised) — scores every event 0–100

## Results

| Metric    | Score | Target | Status  |
|-----------|-------|--------|---------|
| Precision | 0.967 | >0.75  | ✅ PASS |
| Recall    | 0.975 | >0.70  | ✅ PASS |
| F1 Score  | 0.971 | >0.72  | ✅ PASS |
| ROC-AUC   | 0.996 | >0.80  | ✅ PASS |

## Project Structure

```
ps4_project/
├── run_all.py                  ← Run everything in one command
├── ps4_step1_features.py       ← Feature engineering (19 features)
├── ps4_step2_model1.py         ← Event-level anomaly detection
├── ps4_step3_model2.py         ← User-level risk scoring
├── ps4_step4_dashboard.py      ← 4-page Streamlit dashboard
├── requirements.txt
├── sample_data/
│   ├── data_access_logs.csv
│   └── user_profiles.csv
└── outputs/                    ← Auto-created on first run
```

## Dashboard Pages

| Page | What it shows |
|------|--------------|
| 📊 Executive Summary | KPIs, severity breakdown, anomaly trend |
| 🚨 Live Alert Feed | All flagged events with narratives, filterable |
| 👤 User Risk Profiles | 100 users ranked by risk, drilldown per user |
| 📋 Compliance Report | GDPR / NIST IR-4 / SOX 302 coverage + downloads |

## Regulatory Coverage
- **GDPR Article 32** — personal data access monitoring + exfiltration detection
- **NIST IR-4** — automated detection capability + incident response procedures
- **SOX 302** — GL/AR/AP financial system access trail + unauthorised access alerts
