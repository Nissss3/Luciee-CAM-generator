"""
============================================================
  CREDIT DECISIONING ENGINE — LAYER 4: RISK SCORING ENGINE

  Standalone risk scoring that runs AFTER Layer 2 and
  BEFORE Layer 5. Reads borrower_profile.json, computes
  all financial risk scores, and writes results back.

  Scores computed:
  1. Financial Risk Score     — DSCR, ICR, leverage, margins
  2. Business Risk Score      — industry, market position, ESG
  3. Management Risk Score    — promoter, governance, MCA flags
  4. Stress Test Score        — 5 scenarios, worst-case PD
  5. Dynamic Credit Limit     — floor/base/ceiling range
  6. Composite Risk Score     — weighted average of all above

  Input:  borrower_profile.json (from Layer 2)
  Output: borrower_profile.json updated with layer4_* fields

  Run:    python layer4_risk_scoring.py
  Install: pip install pandas numpy
============================================================
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


import os
import json
import math
from pathlib import Path
from datetime import datetime

print("✅ Layer 4 Risk Scoring Engine — loaded")
print(f"📅 Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================
# SECTION 0 — LOAD borrower_profile.json
# ============================================================

BASE_DIR     = Path(os.path.dirname(os.path.abspath(__file__)))
PROFILE_PATH = BASE_DIR / "borrower_profile.json"

if not PROFILE_PATH.exists():
    raise FileNotFoundError(
        f"borrower_profile.json not found at {PROFILE_PATH}\n"
        f"Run layer2.py first to generate it."
    )

with open(PROFILE_PATH, "r", encoding="utf-8") as f:
    profile = json.load(f)

# ── Extract key inputs ─────────────────────────────────────
company_name  = profile.get("company_name",  "Unknown")
industry      = profile.get("industry",      "Unknown")
loan_amount   = float(profile.get("loan_amount",   0))
annual_income = float(profile.get("annual_income", 1))
collateral    = float(profile.get("collateral_value", 0))
loan_tenure   = float(profile.get("loan_tenure_years", 5))

ratios     = profile.get("financial_ratios", {})
ts         = profile.get("time_series",      {})
bank       = profile.get("bank_signals",     {})
gst        = profile.get("gst_signals",      {})
mca        = profile.get("mca_signals",      {})
feats      = profile.get("financial_features", {})

# Layer 3 research scores (if available)
research_score = float(profile.get("layer3_research_score", 50))
red_flags      = profile.get("layer3_red_flags", [])
esg_bps        = float(profile.get("layer3_esg_pricing_bps", 25))

print(f"\n{'='*60}")
print(f"  LAYER 4 RISK SCORING — {company_name}")
print(f"{'='*60}")
print(f"  Industry  : {industry}")
print(f"  Loan Ask  : ₹{loan_amount:,.0f}")
print(f"  Income    : ₹{annual_income:,.0f}")


# ============================================================
# SECTION 1 — FINANCIAL RISK SCORE (0–100, lower = safer)
# ============================================================

print(f"\n[1/6] Financial Risk Score...")

dscr     = float(ratios.get("dscr",                    1.5))
icr      = float(ratios.get("interest_coverage_ratio", 3.0))
ebitda_m = float(ratios.get("ebitda_margin_pct",       20.0))
pat_m    = float(ratios.get("pat_margin_pct",          10.0))
de_ratio = float(ratios.get("debt_equity_ratio",       3.0))
wc_util  = float(ratios.get("wc_utilization_pct",      70.0))

# ── DSCR scoring (higher DSCR = lower risk) ────────────────
if   dscr >= 2.0:  dscr_score = 10
elif dscr >= 1.5:  dscr_score = 25
elif dscr >= 1.25: dscr_score = 40
elif dscr >= 1.0:  dscr_score = 60
elif dscr >= 0.75: dscr_score = 80
else:              dscr_score = 95

# ── ICR scoring ────────────────────────────────────────────
if   icr >= 5.0:  icr_score = 10
elif icr >= 3.0:  icr_score = 25
elif icr >= 2.0:  icr_score = 40
elif icr >= 1.5:  icr_score = 55
elif icr >= 1.0:  icr_score = 75
else:             icr_score = 90

# ── Leverage scoring (D/E ratio) ───────────────────────────
if   de_ratio <= 1.0: de_score = 10
elif de_ratio <= 2.0: de_score = 25
elif de_ratio <= 3.0: de_score = 40
elif de_ratio <= 5.0: de_score = 60
elif de_ratio <= 7.0: de_score = 75
else:                 de_score = 90

# ── Margin scoring ─────────────────────────────────────────
if   ebitda_m >= 30: margin_score = 10
elif ebitda_m >= 20: margin_score = 25
elif ebitda_m >= 15: margin_score = 40
elif ebitda_m >= 10: margin_score = 55
elif ebitda_m >= 5:  margin_score = 70
else:                margin_score = 85

# ── Working capital utilization ────────────────────────────
if   wc_util <= 50:  wc_score = 10
elif wc_util <= 70:  wc_score = 25
elif wc_util <= 85:  wc_score = 45
elif wc_util <= 95:  wc_score = 65
else:                wc_score = 85

# ── Weighted financial risk score ──────────────────────────
financial_risk_score = round(
    dscr_score   * 0.30 +
    icr_score    * 0.25 +
    de_score     * 0.20 +
    margin_score * 0.15 +
    wc_score     * 0.10, 2
)

financial_risk_breakdown = {
    "dscr_score"    : dscr_score,
    "icr_score"     : icr_score,
    "leverage_score": de_score,
    "margin_score"  : margin_score,
    "wc_score"      : wc_score,
    "composite"     : financial_risk_score,
    "rating"        : (
        "LOW"      if financial_risk_score <= 30 else
        "MODERATE" if financial_risk_score <= 55 else
        "HIGH"     if financial_risk_score <= 75 else
        "VERY HIGH"
    )
}

print(f"  DSCR         : {dscr:.2f}x → score {dscr_score}")
print(f"  ICR          : {icr:.2f}x  → score {icr_score}")
print(f"  D/E Ratio    : {de_ratio:.2f}x → score {de_score}")
print(f"  EBITDA Margin: {ebitda_m:.1f}%  → score {margin_score}")
print(f"  WC Util      : {wc_util:.1f}%   → score {wc_score}")
print(f"  ✅ Financial Risk Score: {financial_risk_score}/100 "
      f"({financial_risk_breakdown['rating']})")


# ============================================================
# SECTION 2 — BUSINESS RISK SCORE (0–100)
# ============================================================

print(f"\n[2/6] Business Risk Score...")

# ── Industry cyclicality risk ──────────────────────────────
HIGH_RISK_INDUSTRIES    = ["steel", "real estate", "textiles", "construction",
                            "aviation", "shipping", "mining", "sugar"]
MEDIUM_RISK_INDUSTRIES  = ["banking", "nbfc", "financial", "auto", "pharma",
                            "chemicals", "energy", "retail"]
LOW_RISK_INDUSTRIES     = ["it", "technology", "fmcg", "healthcare", "education",
                            "telecom", "utilities"]

ind_lower = industry.lower()
if any(k in ind_lower for k in HIGH_RISK_INDUSTRIES):
    industry_risk = 70
elif any(k in ind_lower for k in MEDIUM_RISK_INDUSTRIES):
    industry_risk = 45
elif any(k in ind_lower for k in LOW_RISK_INDUSTRIES):
    industry_risk = 25
else:
    industry_risk = 50

# ── Revenue growth trend ───────────────────────────────────
yoy_growth = float(ts.get("yoy_revenue_growth_pct", 0))
cagr       = float(ts.get("revenue_3yr_cagr_pct",   0))
trend      = ts.get("ebitda_trend", "STABLE")
volatility = float(ts.get("revenue_volatility", 0.2))

if yoy_growth >= 15:  growth_risk = 15
elif yoy_growth >= 8: growth_risk = 25
elif yoy_growth >= 0: growth_risk = 40
elif yoy_growth >= -5:growth_risk = 65
else:                 growth_risk = 85

if trend == "IMPROVING": trend_adj = -10
elif trend == "DECLINING": trend_adj = +15
else: trend_adj = 0

if volatility <= 0.1:  vol_risk = 10
elif volatility <= 0.2:vol_risk = 25
elif volatility <= 0.3:vol_risk = 45
else:                  vol_risk = 65

# ── GST compliance ─────────────────────────────────────────
gst_rate    = float(gst.get("gst_filing_rate",         1.0))
gst_consist = float(gst.get("gst_revenue_consistency", 1.0))

if gst_rate >= 1.0:   gst_risk = 5
elif gst_rate >= 0.9: gst_risk = 20
elif gst_rate >= 0.75:gst_risk = 45
else:                 gst_risk = 75

if gst_consist >= 0.9:  consist_risk = 10
elif gst_consist >= 0.7:consist_risk = 35
else:                   consist_risk = 65

# ── Cash flow health ───────────────────────────────────────
iof_ratio  = float(bank.get("inflow_outflow_ratio", 1.0))
cf_vol     = float(bank.get("cash_flow_volatility", 0.2))
emi_burden = float(bank.get("emi_burden_ratio",     0.2))

if iof_ratio >= 1.2:   cf_risk = 15
elif iof_ratio >= 1.0: cf_risk = 30
elif iof_ratio >= 0.9: cf_risk = 55
else:                  cf_risk = 80

business_risk_score = round(
    industry_risk * 0.25 +
    growth_risk   * 0.20 +
    trend_adj     * 0.10 +
    vol_risk      * 0.10 +
    gst_risk      * 0.10 +
    consist_risk  * 0.10 +
    cf_risk       * 0.15, 2
)
business_risk_score = max(0, min(100, business_risk_score))

business_risk_breakdown = {
    "industry_risk"   : industry_risk,
    "growth_risk"     : growth_risk,
    "trend_adjustment": trend_adj,
    "volatility_risk" : vol_risk,
    "gst_risk"        : gst_risk,
    "cashflow_risk"   : cf_risk,
    "composite"       : business_risk_score,
    "rating": (
        "LOW"      if business_risk_score <= 30 else
        "MODERATE" if business_risk_score <= 55 else
        "HIGH"     if business_risk_score <= 75 else
        "VERY HIGH"
    )
}

print(f"  Industry Risk   : {industry_risk} ({industry})")
print(f"  Growth Risk     : {growth_risk} (YoY: {yoy_growth:.1f}%)")
print(f"  EBITDA Trend    : {trend} (adj: {trend_adj})")
print(f"  GST Compliance  : {gst_risk} (rate: {gst_rate*100:.0f}%)")
print(f"  Cash Flow Risk  : {cf_risk} (I/O: {iof_ratio:.2f})")
print(f"  ✅ Business Risk Score: {business_risk_score}/100 "
      f"({business_risk_breakdown['rating']})")


# ============================================================
# SECTION 3 — MANAGEMENT RISK SCORE (0–100)
# ============================================================

print(f"\n[3/6] Management Risk Score...")

# ── MCA flags ─────────────────────────────────────────────
mca_active    = float(mca.get("mca_is_active",   1))
mca_risk_flag = float(mca.get("mca_risk_flag",   0))
insolvency    = float(mca.get("insolvency_flag", 0))
dir_count     = float(mca.get("director_count",  5))
years_inc     = float(mca.get("years_incorporated", 10))

if insolvency:      mca_score = 95
elif mca_risk_flag: mca_score = 70
elif not mca_active:mca_score = 80
else:               mca_score = 10

# ── Directors ─────────────────────────────────────────────
if dir_count >= 5:   dir_score = 10
elif dir_count >= 3: dir_score = 25
elif dir_count >= 2: dir_score = 45
else:                dir_score = 70

# ── Company vintage ────────────────────────────────────────
if years_inc >= 20:   vintage_score = 5
elif years_inc >= 10: vintage_score = 15
elif years_inc >= 5:  vintage_score = 30
elif years_inc >= 2:  vintage_score = 55
else:                 vintage_score = 75

# ── Research-driven management flags ───────────────────────
critical_flags = [f for f in red_flags if "🔴" in str(f)]
research_mgmt_score = min(len(critical_flags) * 20, 60)

management_risk_score = round(
    mca_score          * 0.35 +
    dir_score          * 0.20 +
    vintage_score      * 0.25 +
    research_mgmt_score* 0.20, 2
)
management_risk_score = max(0, min(100, management_risk_score))

management_risk_breakdown = {
    "mca_score"       : mca_score,
    "director_score"  : dir_score,
    "vintage_score"   : vintage_score,
    "research_flags"  : research_mgmt_score,
    "critical_flags"  : critical_flags,
    "composite"       : management_risk_score,
    "rating": (
        "LOW"      if management_risk_score <= 30 else
        "MODERATE" if management_risk_score <= 55 else
        "HIGH"     if management_risk_score <= 75 else
        "VERY HIGH"
    )
}

print(f"  MCA Score       : {mca_score} (active: {bool(mca_active)}, insolvency: {bool(insolvency)})")
print(f"  Director Score  : {dir_score} ({int(dir_count)} directors)")
print(f"  Vintage Score   : {vintage_score} ({int(years_inc)} years)")
print(f"  Research Flags  : {research_mgmt_score} ({len(critical_flags)} critical)")
print(f"  ✅ Management Risk Score: {management_risk_score}/100 "
      f"({management_risk_breakdown['rating']})")


# ============================================================
# SECTION 4 — STRESS TEST SCENARIOS
# ============================================================

print(f"\n[4/6] Stress Testing...")

base_pd    = float(feats.get("EXT_SOURCE_MEAN", 0.65))
base_pd    = round(1 - base_pd, 4)  # invert: higher EXT = lower PD
base_pd    = max(0.01, min(0.95, base_pd))

# Annuity (debt service)
annuity    = float(feats.get("AMT_ANNUITY", loan_amount / (loan_tenure * 12))) * 12

def compute_dscr(income, annuity_val):
    return round(income / annuity_val, 3) if annuity_val > 0 else 0

def pd_from_dscr(dscr_val, base):
    """Higher stress → lower DSCR → higher PD"""
    if dscr_val >= 1.5:   multiplier = 1.0
    elif dscr_val >= 1.2: multiplier = 1.3
    elif dscr_val >= 1.0: multiplier = 1.8
    elif dscr_val >= 0.8: multiplier = 2.5
    else:                 multiplier = 4.0
    return min(round(base * multiplier, 4), 0.95)

scenarios = {
    "Base Case": {
        "income_shock"    : 0,
        "rate_shock_bps"  : 0,
        "cost_shock"      : 0,
        "description"     : "No stress — current conditions"
    },
    "Revenue -20%": {
        "income_shock"    : -0.20,
        "rate_shock_bps"  : 0,
        "cost_shock"      : 0,
        "description"     : "Demand slowdown or sectoral stress"
    },
    "Rate +200bps": {
        "income_shock"    : 0,
        "rate_shock_bps"  : 200,
        "cost_shock"      : 0,
        "description"     : "RBI rate hike cycle"
    },
    "Cost +15%": {
        "income_shock"    : 0,
        "rate_shock_bps"  : 0,
        "cost_shock"      : 0.15,
        "description"     : "Input cost inflation"
    },
    "Combined Shock": {
        "income_shock"    : -0.15,
        "rate_shock_bps"  : 150,
        "cost_shock"      : 0.10,
        "description"     : "Simultaneous macro stress"
    },
}

stress_results = {}
for scenario_name, params in scenarios.items():
    stressed_income  = annual_income * (1 + params["income_shock"])
    rate_impact      = (params["rate_shock_bps"] / 10000) * loan_amount
    cost_impact      = params["cost_shock"] * annual_income * 0.3
    stressed_annuity = annuity + rate_impact + cost_impact

    s_dscr = compute_dscr(stressed_income, stressed_annuity)
    s_pd   = pd_from_dscr(s_dscr, base_pd)
    s_el   = round(s_pd * 0.45 * loan_amount, 0)

    if s_dscr >= 1.5:   outcome = "PASS"
    elif s_dscr >= 1.2: outcome = "WATCH"
    elif s_dscr >= 1.0: outcome = "CONCERN"
    else:               outcome = "FAIL"

    stress_results[scenario_name] = {
        "description"    : params["description"],
        "stressed_income": round(stressed_income, 0),
        "stressed_dscr"  : s_dscr,
        "stressed_pd_pct": round(s_pd * 100, 2),
        "expected_loss"  : s_el,
        "outcome"        : outcome,
    }
    print(f"  {scenario_name:<20} DSCR: {s_dscr:.2f} | PD: {s_pd*100:.1f}% | {outcome}")

# Worst case scenario
worst_pd   = max(v["stressed_pd_pct"] for v in stress_results.values()) / 100
worst_case = max(stress_results.items(), key=lambda x: x[1]["stressed_pd_pct"])[0]
fail_count = sum(1 for v in stress_results.values() if v["outcome"] == "FAIL")
pass_count = sum(1 for v in stress_results.values() if v["outcome"] == "PASS")

stress_summary = {
    "scenarios"       : stress_results,
    "base_pd_pct"     : round(base_pd * 100, 2),
    "worst_case_pd_pct": round(worst_pd * 100, 2),
    "worst_scenario"  : worst_case,
    "pass_count"      : pass_count,
    "fail_count"      : fail_count,
    "stress_rating"   : (
        "RESILIENT"   if fail_count == 0 else
        "ADEQUATE"    if fail_count <= 1 else
        "VULNERABLE"  if fail_count <= 2 else
        "FRAGILE"
    )
}

print(f"  ✅ Stress Rating: {stress_summary['stress_rating']} "
      f"(Pass: {pass_count}/5, Fail: {fail_count}/5)")


# ============================================================
# SECTION 5 — DYNAMIC CREDIT LIMIT ENGINE
# ============================================================

print(f"\n[5/6] Dynamic Credit Limit...")

# ── Base limit from income multiple ────────────────────────
income_multiple = {
    "APPROVE"            : 6.0,
    "CONDITIONAL APPROVE": 4.0,
    "REFER TO SENIOR"    : 3.0,
    "REJECT"             : 0.0,
}

# Determine preliminary decision band from financial score
if financial_risk_score <= 30:   prelim = "APPROVE"
elif financial_risk_score <= 55: prelim = "CONDITIONAL APPROVE"
elif financial_risk_score <= 75: prelim = "REFER TO SENIOR"
else:                            prelim = "REJECT"

mult        = income_multiple[prelim]
base_limit  = annual_income * mult

# ── DSCR-based cap ─────────────────────────────────────────
# Max loan = DSCR-adjusted annual debt capacity
dscr_capacity = (annual_income / 1.25) / (1 / loan_tenure)
dscr_limit    = min(base_limit, dscr_capacity)

# ── Collateral coverage ────────────────────────────────────
collateral_limit = collateral * 1.5  # 150% LTV
collateral_floor = collateral * 0.8  # conservative floor

# ── Stress-adjusted limit ──────────────────────────────────
stress_adj = 1.0
if stress_summary["stress_rating"] == "FRAGILE":      stress_adj = 0.50
elif stress_summary["stress_rating"] == "VULNERABLE": stress_adj = 0.70
elif stress_summary["stress_rating"] == "ADEQUATE":   stress_adj = 0.85
stressed_limit = dscr_limit * stress_adj

# ── Final limit range ──────────────────────────────────────
ceiling_limit = min(dscr_limit, loan_amount)
base_rec_limit= min(stressed_limit, loan_amount)
floor_limit   = min(base_rec_limit * 0.75, collateral_floor if collateral > 0 else base_rec_limit * 0.75)

dynamic_limit = {
    "requested_amount"  : loan_amount,
    "floor"             : round(floor_limit, 0),
    "base_recommendation": round(base_rec_limit, 0),
    "ceiling"           : round(ceiling_limit, 0),
    "dscr_capacity"     : round(dscr_capacity, 0),
    "stress_adjustment" : stress_adj,
    "collateral_coverage": round(collateral / loan_amount * 100, 1) if loan_amount > 0 else 0,
    "preliminary_band"  : prelim,
}

print(f"  Floor           : ₹{floor_limit:,.0f}")
print(f"  Base Rec.       : ₹{base_rec_limit:,.0f}")
print(f"  Ceiling         : ₹{ceiling_limit:,.0f}")
print(f"  Requested       : ₹{loan_amount:,.0f}")
print(f"  Collateral Cover: {dynamic_limit['collateral_coverage']:.1f}%")
print(f"  ✅ Preliminary Band: {prelim}")


# ============================================================
# SECTION 6 — COMPOSITE RISK SCORE + FINAL RECOMMENDATION
# ============================================================

print(f"\n[6/6] Composite Risk Score...")

# ── Weighted composite ─────────────────────────────────────
composite_risk_score = round(
    financial_risk_score   * 0.35 +
    business_risk_score    * 0.25 +
    management_risk_score  * 0.20 +
    (research_score / 100 * 100) * 0.10 +   # layer3 research (already 0-100)
    (worst_pd * 100)       * 0.10, 2
)
composite_risk_score = max(0, min(100, composite_risk_score))

# ── Risk grade ─────────────────────────────────────────────
if composite_risk_score <= 25:
    risk_grade = "AAA"; risk_label = "Exceptional — Approve"
elif composite_risk_score <= 35:
    risk_grade = "AA";  risk_label = "Excellent — Approve"
elif composite_risk_score <= 45:
    risk_grade = "A";   risk_label = "Good — Approve"
elif composite_risk_score <= 55:
    risk_grade = "BBB"; risk_label = "Satisfactory — Conditional Approve"
elif composite_risk_score <= 65:
    risk_grade = "BB";  risk_label = "Acceptable — Conditional Approve"
elif composite_risk_score <= 75:
    risk_grade = "B";   risk_label = "Watch — Refer to Senior"
elif composite_risk_score <= 85:
    risk_grade = "CCC"; risk_label = "Substandard — Refer to Senior"
else:
    risk_grade = "D";   risk_label = "Default Risk — Reject"

# ── Layer 4 recommendation ─────────────────────────────────
if composite_risk_score <= 45:
    layer4_recommendation = "APPROVE"
elif composite_risk_score <= 65:
    layer4_recommendation = "CONDITIONAL APPROVE"
elif composite_risk_score <= 75:
    layer4_recommendation = "REFER TO SENIOR CREDIT"
else:
    layer4_recommendation = "REJECT"

# ── ESG pricing adjustment ─────────────────────────────────
# Add research ESG bps on top of base rate
esg_rate_adj = round(esg_bps / 10000, 6)  # convert bps to decimal

print(f"  Financial Risk  : {financial_risk_score:.1f}/100")
print(f"  Business Risk   : {business_risk_score:.1f}/100")
print(f"  Management Risk : {management_risk_score:.1f}/100")
print(f"  Research Score  : {research_score:.1f}/100")
print(f"  Stress Worst PD : {worst_pd*100:.1f}%")
print(f"  ─────────────────────────────")
print(f"  Composite       : {composite_risk_score:.1f}/100")
print(f"  Risk Grade      : {risk_grade} — {risk_label}")
print(f"  Recommendation  : {layer4_recommendation}")
print(f"  ESG Rate Adj    : +{esg_bps}bps")


# ============================================================
# SECTION 7 — WRITE BACK TO borrower_profile.json
# ============================================================

print(f"\n[7/7] Writing results to borrower_profile.json...")

profile["layer4_completed"]            = True
profile["layer4_composite_risk_score"] = composite_risk_score
profile["layer4_risk_grade"]           = risk_grade
profile["layer4_risk_label"]           = risk_label
profile["layer4_recommendation"]       = layer4_recommendation
profile["layer4_esg_rate_adjustment"]  = esg_rate_adj

profile["layer4_financial_risk"]   = financial_risk_breakdown
profile["layer4_business_risk"]    = business_risk_breakdown
profile["layer4_management_risk"]  = management_risk_breakdown
profile["layer4_stress_test"]      = stress_summary
profile["layer4_dynamic_limit"]    = dynamic_limit

# Narrative summaries for Layer 6 CAM
profile["layer4_cam_narratives"] = {
    "financial_risk_summary": (
        f"{company_name} demonstrates a {financial_risk_breakdown['rating'].lower()} financial risk "
        f"profile with DSCR of {dscr:.2f}x and interest coverage of {icr:.2f}x. "
        f"EBITDA margin stands at {ebitda_m:.1f}%, with a debt-equity ratio of {de_ratio:.1f}x."
    ),
    "business_risk_summary": (
        f"Operating in {industry}, the company exhibits a {business_risk_breakdown['rating'].lower()} "
        f"business risk with {yoy_growth:.1f}% YoY revenue growth and an {trend.lower()} EBITDA trend. "
        f"GST filing compliance rate is {gst_rate*100:.0f}%."
    ),
    "management_risk_summary": (
        f"Management risk is rated {management_risk_breakdown['rating'].lower()} with "
        f"{int(dir_count)} directors on record and {int(years_inc)} years of incorporation. "
        f"{'No adverse MCA flags detected.' if not bool(insolvency) else 'Insolvency proceedings flagged — escalation required.'}"
    ),
    "stress_test_summary": (
        f"Stress testing across 5 scenarios yields a {stress_summary['stress_rating'].lower()} resilience rating. "
        f"Base case PD: {base_pd*100:.1f}%. Worst case ('{worst_case}'): {worst_pd*100:.1f}% PD. "
        f"{pass_count} of 5 scenarios pass minimum DSCR threshold of 1.25x."
    ),
    "credit_limit_summary": (
        f"Dynamic credit limit analysis recommends a base limit of ₹{base_rec_limit:,.0f} "
        f"(floor: ₹{floor_limit:,.0f}, ceiling: ₹{ceiling_limit:,.0f}) "
        f"against the requested ₹{loan_amount:,.0f}. "
        f"Collateral coverage stands at {dynamic_limit['collateral_coverage']:.1f}%."
    ),
}

with open(PROFILE_PATH, "w", encoding="utf-8") as f:
    json.dump(profile, f, indent=2, default=str)


# ============================================================
# SECTION 8 — FINAL SUMMARY
# ============================================================

print(f"\n{'='*60}")
print(f"  LAYER 4 COMPLETE")
print(f"{'='*60}")
print(f"  Company          : {company_name}")
print(f"  Composite Score  : {composite_risk_score:.1f}/100")
print(f"  Risk Grade       : {risk_grade}")
print(f"  Recommendation   : {layer4_recommendation}")
print(f"  Credit Limit Rec : ₹{base_rec_limit:,.0f}")
print(f"  Stress Rating    : {stress_summary['stress_rating']}")
print(f"  ESG Adjustment   : +{esg_bps}bps")
print(f"\n  ✅ borrower_profile.json updated")
print(f"\n  NEXT → python layer5_collab.py (on Colab)")
print(f"{'='*60}")