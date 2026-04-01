"""
============================================================
  CREDIT DECISIONING ENGINE — LAYER 5: ML INFERENCE (LOCAL)

  This is the LOCAL inference wrapper.
  It does NOT train anything.
  It loads pre-trained models and scores ONE borrower.

  Reads:  borrower_profile.json   (from Layer 2 + 4)
          model_lgb.pkl            (trained in Colab once)
          imputer.pkl
          scaler.pkl
          feature_names.json

  Writes: ml_output.json           (feeds into Layer 6)
          borrower_profile.json    (appends layer5_* keys)

  Run:    python layer5.py
  Time:   ~2 seconds
============================================================
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


import os
import json
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import joblib
import shap

print("✅ Layer 5 (Local Inference) — loaded")
print(f"📅 Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── Paths ──────────────────────────────────────────────────
BASE_DIR     = Path(os.path.dirname(os.path.abspath(__file__)))
PROFILE_PATH = BASE_DIR / "borrower_profile.json"
OUTPUT_PATH  = BASE_DIR / "ml_output.json"


# ============================================================
# SECTION 1 — LOAD MODELS
# ============================================================

print("\n[1/4] Loading pre-trained models...")

required_files = ["model_lgb.pkl", "imputer.pkl", "scaler.pkl", "feature_names.json"]
for fname in required_files:
    if not (BASE_DIR / fname).exists():
        raise FileNotFoundError(
            f"❌ {fname} not found.\n"
            f"   Download it from Colab after running save_models().\n"
            f"   Place it in: {BASE_DIR}"
        )

lgb_model      = joblib.load(BASE_DIR / "model_lgb.pkl")
imputer        = joblib.load(BASE_DIR / "imputer.pkl")
scaler         = joblib.load(BASE_DIR / "scaler.pkl")

with open(BASE_DIR / "feature_names.json", "r") as f:
    feature_names = json.load(f)

print(f"  ✅ LightGBM model loaded")
print(f"  ✅ Imputer + Scaler loaded")
print(f"  ✅ Feature names: {len(feature_names)} features expected")


# ============================================================
# SECTION 2 — LOAD BORROWER PROFILE
# ============================================================

print("\n[2/4] Loading borrower profile...")

if not PROFILE_PATH.exists():
    raise FileNotFoundError(
        f"borrower_profile.json not found. Run layer2.py first."
    )

with open(PROFILE_PATH, "r", encoding="utf-8") as f:
    profile = json.load(f)

company_name   = profile.get("company_name",  "Unknown")
loan_amount    = float(profile.get("loan_amount",   500000))
annual_income  = float(profile.get("annual_income", 800000))
collateral     = float(profile.get("collateral_value", 0))

print(f"  Company    : {company_name}")
print(f"  Loan Ask   : ₹{loan_amount:,.0f}")
print(f"  Income     : ₹{annual_income:,.0f}")


# ============================================================
# SECTION 3 — BUILD FEATURE VECTOR
# ============================================================

print("\n[3/4] Building feature vector...")

# Pull the financial_features dict built by Layer 2
raw_features = profile.get("financial_features", {})

# Align to the exact feature order the model was trained on
# Missing features get 0 (imputer will handle NaN-filling)
feature_vector = []
missing_features = []

for feat in feature_names:
    val = raw_features.get(feat, None)
    if val is None:
        feature_vector.append(np.nan)
        missing_features.append(feat)
    else:
        # Handle non-numeric values (categorical encoded as strings)
        try:
            feature_vector.append(float(val))
        except (TypeError, ValueError):
            feature_vector.append(np.nan)
            missing_features.append(feat)

X = np.array(feature_vector).reshape(1, -1)

if missing_features:
    print(f"  ⚠️  {len(missing_features)} features missing → filled with median by imputer")

# Apply imputer (fills NaN with training median)
X_imp = imputer.transform(X)

print(f"  ✅ Feature vector: {X_imp.shape[1]} features aligned")


# ============================================================
# SECTION 4 — PREDICT + SHAP
# ============================================================

print("\n[4/4] Running inference...")

# ── PD Score ───────────────────────────────────────────────
pd_score = float(lgb_model.predict_proba(X_imp)[0][1])
print(f"  PD Score (raw): {pd_score*100:.4f}%")

# ── SHAP Explanation ───────────────────────────────────────
print("  Computing SHAP values...")
explainer   = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_imp)

# LightGBM binary classification: shap_values is array of shape (1, n_features)
if isinstance(shap_values, list):
    sv = shap_values[1][0]   # positive class
else:
    sv = shap_values[0]

# Top 10 SHAP drivers
shap_df = sorted(
    [{"feature": feature_names[i], "shap_value": float(sv[i]),
      "abs_shap": abs(float(sv[i]))} for i in range(len(sv))],
    key=lambda x: x["abs_shap"], reverse=True
)[:10]

shap_explanations = []
for item in shap_df:
    shap_explanations.append({
        "feature"   : item["feature"],
        "shap_value": round(item["shap_value"], 4),
        "direction" : "risk_increase" if item["shap_value"] > 0 else "risk_decrease",
        "impact"    : "HIGH" if item["abs_shap"] > 0.1 else
                      "MEDIUM" if item["abs_shap"] > 0.05 else "LOW"
    })


# ============================================================
# HELPER FUNCTIONS (copied from Colab — pure Python, no deps)
# ============================================================

def compute_expected_loss(pd_score, loan_amount, collateral_value=0):
    pd_value = pd_score
    if collateral_value > 0:
        collateral_coverage = collateral_value / (loan_amount + 1)
        lgd = max(0.10, 1 - (collateral_coverage * 0.8))
        lgd = min(lgd, 0.90)
    else:
        lgd = 0.75
    ead = loan_amount
    el  = pd_value * lgd * ead
    return {
        "PD"                : round(pd_value, 4),
        "LGD"               : round(lgd, 4),
        "EAD"               : round(ead, 2),
        "Expected_Loss"     : round(el, 2),
        "Expected_Loss_Pct" : round((el / ead) * 100, 2) if ead > 0 else 0
    }


def assign_risk_rating(pd_score):
    rating_map = [
        (0.005, 1, "AAA", "Exceptional"),
        (0.01,  2, "AA",  "Excellent"),
        (0.02,  3, "A",   "Good"),
        (0.05,  4, "BBB", "Satisfactory"),
        (0.10,  5, "BB",  "Acceptable"),
        (0.20,  6, "B",   "Watch"),
        (0.35,  7, "CCC", "Substandard"),
        (1.00,  8, "D",   "Default/Loss")
    ]
    for threshold, rating, grade, label in rating_map:
        if pd_score <= threshold:
            return {"rating": rating, "grade": grade, "label": label}
    return {"rating": 8, "grade": "D", "label": "Default/Loss"}


def compute_risk_premium(pd_score, lgd, risk_free_rate=0.065):
    expected_loss_rate = pd_score * lgd
    capital_charge     = pd_score * lgd * 8
    operating_cost     = 0.015
    profit_margin      = 0.010
    total_spread       = expected_loss_rate + capital_charge * 0.12 + operating_cost + profit_margin
    final_rate         = risk_free_rate + total_spread
    return {
        "risk_free_rate"    : f"{risk_free_rate*100:.2f}%",
        "credit_spread"     : f"{total_spread*100:.2f}%",
        "recommended_rate"  : f"{final_rate*100:.2f}%",
        "rate_band"         : f"{(final_rate-0.005)*100:.2f}% - {(final_rate+0.01)*100:.2f}%"
    }


def make_lending_decision(pd_score, loan_amount, annual_income, collateral_value=0):
    el_metrics = compute_expected_loss(pd_score, loan_amount, collateral_value)
    rating     = assign_risk_rating(pd_score)
    pricing    = compute_risk_premium(pd_score, el_metrics["LGD"])
    dscr_proxy = annual_income / (loan_amount * 0.12 + 1)

    if pd_score < 0.05 and dscr_proxy > 1.5:
        decision  = "APPROVE"
        max_limit = min(loan_amount, annual_income * 6)
    elif pd_score < 0.15 and dscr_proxy > 1.2:
        decision  = "CONDITIONAL APPROVE"
        max_limit = min(loan_amount * 0.75, annual_income * 4)
    elif pd_score < 0.25:
        decision  = "REFER TO SENIOR CREDIT"
        max_limit = min(loan_amount * 0.5, annual_income * 3)
    else:
        decision  = "REJECT"
        max_limit = 0

    return {
        "decision"          : decision,
        "risk_rating"       : rating,
        "pd_score"          : pd_score,
        "metrics"           : el_metrics,
        "pricing"           : pricing,
        "credit_limit"      : round(max_limit, 0),
        "financial_features": raw_features,
        "shap_explanations" : shap_explanations,
        "timestamp"         : datetime.now().isoformat()
    }


# ============================================================
# SECTION 5 — COMPUTE ALL METRICS + SAVE
# ============================================================

result = make_lending_decision(pd_score, loan_amount, annual_income, collateral)

# ── Print decision ─────────────────────────────────────────
ICONS = {
    "APPROVE"              : "🟢",
    "CONDITIONAL APPROVE"  : "🟡",
    "REFER TO SENIOR CREDIT": "🟠",
    "REJECT"               : "🔴"
}
icon = ICONS.get(result["decision"], "⚪")

print(f"\n{'='*60}")
print(f"  ML CREDIT DECISION — {company_name}")
print(f"{'='*60}")
print(f"  Decision     : {icon} {result['decision']}")
print(f"  Risk Rating  : {result['risk_rating']['rating']} "
      f"({result['risk_rating']['grade']}) — {result['risk_rating']['label']}")
print(f"  PD Score     : {pd_score*100:.3f}%")
print(f"  LGD          : {result['metrics']['LGD']*100:.1f}%")
print(f"  Expected Loss: ₹{result['metrics']['Expected_Loss']:,.0f} "
      f"({result['metrics']['Expected_Loss_Pct']:.2f}%)")
print(f"  Credit Limit : ₹{result['credit_limit']:,.0f}")
print(f"  Rate         : {result['pricing']['recommended_rate']} "
      f"(Band: {result['pricing']['rate_band']})")
print(f"\n  Top SHAP Drivers:")
for s in shap_explanations[:5]:
    arrow = "⬆️" if s["direction"] == "risk_increase" else "⬇️"
    print(f"    {arrow} [{s['impact']:<6}] {s['feature']} ({s['shap_value']:+.4f})")
print(f"{'='*60}")

# ── Save ml_output.json ────────────────────────────────────
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"\n  ✅ Saved: {OUTPUT_PATH}")

# ── Update borrower_profile.json ───────────────────────────
profile["layer5_completed"]   = True
profile["layer5_decision"]    = result["decision"]
profile["layer5_final_pd"]    = pd_score
profile["layer5_shap_explanations"] = shap_explanations
profile["layer5_risk_rating"] = result["risk_rating"]
profile["layer5_credit_limit"]= result["credit_limit"]
profile["layer5_pricing"]     = result["pricing"]
profile["layer5_metrics"]     = result["metrics"]

with open(PROFILE_PATH, "w", encoding="utf-8") as f:
    json.dump(profile, f, indent=2, default=str)
print(f"  ✅ borrower_profile.json updated with layer5_* fields")

print(f"\n  NEXT → python layer6.py")
print(f"{'='*60}")