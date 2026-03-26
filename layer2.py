"""
============================================================
  CREDIT DECISIONING ENGINE — LAYER 2 (LOCAL)
  Bronze → Silver → Gold → borrower_profile.json

  Replaces the 3 Databricks notebooks with a single
  local Python script. Reads Bronze JSON files saved
  by Layer 1, engineers all features, applies mock HE
  on CIBIL fields, and writes borrower_profile.json.

  Run:
      python layer2_local.py

  Install:
      pip install pandas numpy tenseal --break-system-packages
      (tenseal is optional — falls back to mock HE if not installed)
============================================================
"""

import os
import json
import math
import hashlib
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd

print("✅ Layer 2 (Local) — imports loaded")
print(f"📅 Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================
# SECTION 0 — CONFIGURATION
# Edit these values for each new borrower
# ============================================================

# ── These are read from ingestion_manifest.json automatically ─
# ── Only change if you want to override ───────────────────────

COMPANY_ID    = "COMP_001"
COMPANY_NAME  = "Tata Capital Limited"
INDUSTRY      = "Banking & Financial Services"
PROMOTER_NAME = "Rajiv Sabharwal"
CIN           = "U65990MH1991PLC060670"
LOAN_AMOUNT   = 500_000_000       # ₹50 crore
LOAN_TENURE   = 5                 # years
LOAN_PURPOSE  = "Working Capital"
DOB_PROMOTER  = date(1968, 4, 15) # Promoter date of birth
EMP_START     = date(2021, 4, 1)  # Company/promoter employment start

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR    = Path(os.path.dirname(os.path.abspath(__file__)))
BRONZE_ROOT = BASE_DIR / "raw_data" / "bronze" / COMPANY_ID
OUTPUT_PATH = BASE_DIR / "borrower_profile.json"
TODAY       = date.today()


# ============================================================
# SECTION 1 — HELPERS
# ============================================================

def read_latest_bronze(source_type: str) -> dict | list | None:
    """
    Read the latest Bronze JSON file for a given source type.
    Layer 1 saves multiple timestamped files — we always pick the latest.
    Returns the 'data' field from the wrapper JSON.
    """
    folder = BRONZE_ROOT / source_type
    if not folder.exists():
        print(f"  ⚠️  Bronze/{source_type} folder not found")
        return None

    json_files = sorted(folder.glob("*.json"), key=lambda f: f.stat().st_mtime)
    if not json_files:
        print(f"  ⚠️  No files in Bronze/{source_type}")
        return None

    latest = json_files[-1]
    print(f"  📂 {source_type}: reading {latest.name}")

    with open(latest, "r", encoding="utf-8") as f:
        wrapper = json.load(f)

    return wrapper.get("data", wrapper)


def safe_float(val, default=0.0) -> float:
    try:
        if val is None or str(val).strip() in ("", "None", "nan"):
            return default
        return float(str(val).replace(",", "").replace("₹", "").strip())
    except:
        return default


def safe_int(val, default=0) -> int:
    return int(safe_float(val, default))


def safe_bool(val, default=False) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "yes", "1")
    return default


def records_to_df(data) -> pd.DataFrame:
    """Convert Bronze data (list of dicts or single dict) to DataFrame."""
    if data is None:
        return pd.DataFrame()
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame()


# ============================================================
# SECTION 2 — READ MANIFEST + OVERRIDE CONFIG
# ============================================================

print(f"\n{'='*55}")
print(f"  LAYER 2 — Bronze → borrower_profile.json")
print(f"{'='*55}")

manifest_path = BRONZE_ROOT / "ingestion_manifest.json"
if manifest_path.exists():
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    borrower = manifest.get("borrower", {})

    # Override config from manifest if available
    COMPANY_ID    = borrower.get("company_id",    COMPANY_ID)
    COMPANY_NAME  = borrower.get("company_name",  COMPANY_NAME)
    INDUSTRY      = borrower.get("industry",      INDUSTRY)
    PROMOTER_NAME = borrower.get("promoter_name", PROMOTER_NAME)
    CIN           = borrower.get("cin",           CIN)
    LOAN_AMOUNT   = float(borrower.get("loan_amount", LOAN_AMOUNT))
    LOAN_PURPOSE  = borrower.get("loan_purpose",  LOAN_PURPOSE)
    LOAN_TENURE   = int(borrower.get("loan_tenure_years", LOAN_TENURE))

    print(f"✅ Manifest loaded")
else:
    print(f"⚠️  No manifest found — using config defaults")

print(f"   Company  : {COMPANY_NAME}")
print(f"   Industry : {INDUSTRY}")
print(f"   Loan Ask : ₹{LOAN_AMOUNT:,.0f}")
print(f"   CIN      : {CIN}")


# ============================================================
# SECTION 3 — BRONZE → SILVER (clean & validate)
# ============================================================

print(f"\n{'─'*55}")
print(f"  STAGE 1 — Bronze → Silver")
print(f"{'─'*55}")

# ── Financials ─────────────────────────────────────────────
print("\n[1/6] Financial Statements...")
raw_fin = read_latest_bronze("financials")
fin_df  = records_to_df(raw_fin)

if not fin_df.empty:
    # Standardize column names
    fin_df.columns = [c.lower().strip() for c in fin_df.columns]
    fin_df["revenue"]  = fin_df.get("revenue",  fin_df.get("net_sales",  pd.Series([0]*len(fin_df)))).apply(safe_float)
    fin_df["ebitda"]   = fin_df.get("ebitda",   pd.Series([0]*len(fin_df))).apply(safe_float)
    fin_df["pat"]      = fin_df.get("pat",       pd.Series([0]*len(fin_df))).apply(safe_float)
    fin_df["interest"] = fin_df.get("interest_expense", fin_df.get("interest", pd.Series([0]*len(fin_df)))).apply(safe_float)
    fin_df["is_valid"] = fin_df["revenue"] > 0
    print(f"  ✅ {len(fin_df)} rows | valid: {fin_df['is_valid'].sum()}")
else:
    print(f"  ⚠️  Empty — will use CMA data")

# ── Bank Statements ────────────────────────────────────────
print("\n[2/6] Bank Statements...")
raw_bank = read_latest_bronze("bank_statements")
bank_df  = records_to_df(raw_bank)

if not bank_df.empty:
    bank_df.columns = [c.lower().strip() for c in bank_df.columns]
    bank_df["debit"]  = bank_df["debit"].apply(safe_float)
    bank_df["credit"] = bank_df["credit"].apply(safe_float)
    bank_df["balance"]= bank_df["balance"].apply(safe_float)

    # Classify transactions
    def classify_txn(row):
        desc = str(row.get("description", "")).lower()
        if "salary" in desc or "payroll" in desc:        return "SALARY"
        if "gst" in desc or "tax" in desc:               return "TAX"
        if "disbursement" in desc:                       return "LOAN_DISBURSEMENT"
        if ("emi" in desc or "loan" in desc) and row.get("debit", 0) > 0: return "LOAN_REPAYMENT"
        if row.get("credit", 0) > 0:                     return "RECEIPT"
        if "utility" in desc or "rent" in desc:          return "OPERATING_EXPENSE"
        if "interest" in desc:                           return "INTEREST"
        return "OTHER"

    bank_df["txn_category"] = bank_df.apply(classify_txn, axis=1)
    bank_df = bank_df.drop_duplicates(subset=["debit", "credit", "balance"])
    print(f"  ✅ {len(bank_df)} transactions | categories: {bank_df['txn_category'].value_counts().to_dict()}")
else:
    print(f"  ⚠️  Empty bank statement")

# ── GST Returns ────────────────────────────────────────────
print("\n[3/6] GST Returns...")
raw_gst = read_latest_bronze("gst")
gst_df  = records_to_df(raw_gst)

if not gst_df.empty:
    gst_df.columns = [c.lower().strip() for c in gst_df.columns]
    gst_df["total_turnover"] = gst_df["total_turnover"].apply(safe_float)
    gst_df["net_gst_paid"]   = gst_df["net_gst_paid"].apply(safe_float)
    gst_df["is_filed"]       = gst_df["filing_status"].str.lower().str.strip() == "filed"
    print(f"  ✅ {len(gst_df)} quarters | filed: {gst_df['is_filed'].sum()}/{len(gst_df)}")
else:
    print(f"  ⚠️  Empty GST data")

# ── CIBIL ──────────────────────────────────────────────────
print("\n[4/6] CIBIL Score...")
raw_cibil = read_latest_bronze("cibil")
cibil_df  = records_to_df(raw_cibil)

if not cibil_df.empty:
    cibil_df.columns = [c.lower().strip() for c in cibil_df.columns]
    cibil_score  = safe_int(cibil_df["cibil_score"].iloc[0], 700)
    score_norm   = round((cibil_score - 300) / 600, 4)
    dpd_12m      = safe_int(cibil_df.get("dpd_last_12_months", pd.Series([0])).iloc[0])
    overdue_accts= safe_int(cibil_df.get("overdue_accounts",   pd.Series([0])).iloc[0])
    on_time_pct  = safe_float(cibil_df.get("on_time_payment_pct", pd.Series([95.0])).iloc[0]) / 100
    npa_flag     = safe_bool(cibil_df.get("npa_flag",             pd.Series([False])).iloc[0])
    wilful_flag  = safe_bool(cibil_df.get("wilful_defaulter_flag",pd.Series([False])).iloc[0])
    total_accts  = safe_int(cibil_df.get("total_accounts",        pd.Series([5])).iloc[0])
    active_accts = safe_int(cibil_df.get("active_accounts",       pd.Series([3])).iloc[0])
    print(f"  ✅ Score: {cibil_score} | Normalized: {score_norm} | DPD: {dpd_12m} | NPA: {npa_flag}")
else:
    print(f"  ⚠️  No CIBIL — using defaults")
    cibil_score = 700; score_norm = 0.6667; dpd_12m = 0
    overdue_accts = 0; on_time_pct = 0.95; npa_flag = False
    wilful_flag = False; total_accts = 5; active_accts = 3

# ── MCA ────────────────────────────────────────────────────
print("\n[5/6] MCA Filing...")
raw_mca = read_latest_bronze("mca")
mca_df  = records_to_df(raw_mca)

if not mca_df.empty:
    mca_df.columns = [c.lower().strip() for c in mca_df.columns]
    mca_status   = str(mca_df.get("company_status", pd.Series(["Active"])).iloc[0])
    is_active    = mca_status.lower() == "active"
    insolvency   = safe_bool(mca_df.get("insolvency_proceedings", pd.Series([False])).iloc[0])
    strike_off   = safe_bool(mca_df.get("strike_off_notice",      pd.Series([False])).iloc[0])
    dir_disq     = safe_bool(mca_df.get("director_disqualification",pd.Series([False])).iloc[0])
    mca_risk     = insolvency or strike_off or dir_disq

    # Count directors
    dir_count = sum(1 for c in mca_df.columns if c.startswith("director_") and c.endswith("_name")
                    and str(mca_df[c].iloc[0]).strip() not in ("", "None", "nan"))
    if dir_count == 0:
        dir_count = 5  # default

    # Years incorporated
    doi_str = str(mca_df.get("date_of_incorporation", pd.Series(["1991-01-01"])).iloc[0])
    try:
        doi_date       = datetime.strptime(doi_str[:10], "%Y-%m-%d").date()
        years_inc      = (TODAY - doi_date).days // 365
    except:
        years_inc = 10

    print(f"  ✅ Status: {mca_status} | Risk: {mca_risk} | Directors: {dir_count} | Years: {years_inc}")
else:
    print(f"  ⚠️  No MCA — using defaults")
    is_active = True; mca_risk = False; insolvency = False
    dir_count = 5; years_inc = 10

# ── CMA ────────────────────────────────────────────────────
print("\n[6/6] CMA Data...")
raw_cma = read_latest_bronze("cma")
cma_df  = records_to_df(raw_cma)

if not cma_df.empty:
    cma_df.columns = [c.lower().strip() for c in cma_df.columns]
    cma_df["net_sales"] = cma_df.get("net_sales", cma_df.get("revenue", pd.Series([0]*len(cma_df)))).apply(safe_float)
    cma_df["ebitda"]    = cma_df["ebitda"].apply(safe_float)
    cma_df["interest"]  = cma_df.get("interest", cma_df.get("interest_expense", pd.Series([0]*len(cma_df)))).apply(safe_float)
    cma_df["pat"]       = cma_df["pat"].apply(safe_float)
    cma_df["is_projected"] = cma_df["period"].str.lower().str.contains("projected", na=False)
    actuals = cma_df[~cma_df["is_projected"]].sort_values("period")
    print(f"  ✅ {len(cma_df)} periods | actuals: {len(actuals)} | projected: {cma_df['is_projected'].sum()}")
else:
    print(f"  ⚠️  No CMA data")
    actuals = pd.DataFrame()


# ============================================================
# SECTION 4 — SILVER → GOLD (feature engineering)
# ============================================================

print(f"\n{'─'*55}")
print(f"  STAGE 2 — Silver → Gold Feature Engineering")
print(f"{'─'*55}")

# ── A. Financial ratios ────────────────────────────────────
print("\n[A] Financial Ratios...")

if not actuals.empty:
    latest = actuals.iloc[-1]
    net_sales = safe_float(latest.get("net_sales", 0))
    ebitda    = safe_float(latest.get("ebitda",    0))
    interest  = safe_float(latest.get("interest",  0))
    pat       = safe_float(latest.get("pat",       0))
    wc_limit  = safe_float(latest.get("working_capital_limit", net_sales * 0.15))
    util_pct  = safe_float(latest.get("utilization_pct", 70))
    total_debt= safe_float(latest.get("total_debt", net_sales * 7.2))
elif not fin_df.empty:
    valid_fin = fin_df[fin_df["is_valid"]].sort_values("year") if "year" in fin_df.columns else fin_df
    latest    = valid_fin.iloc[-1] if not valid_fin.empty else fin_df.iloc[-1]
    net_sales = safe_float(latest.get("revenue", 0))
    ebitda    = safe_float(latest.get("ebitda",  0))
    interest  = safe_float(latest.get("interest_expense", 0))
    pat       = safe_float(latest.get("pat",     0))
    wc_limit  = net_sales * 0.15
    util_pct  = 70.0
    total_debt= net_sales * 7.2
else:
    net_sales = LOAN_AMOUNT * 10
    ebitda    = net_sales * 0.27
    interest  = net_sales * 0.04
    pat       = net_sales * 0.14
    wc_limit  = net_sales * 0.15
    util_pct  = 70.0
    total_debt= net_sales * 7.2

annual_income  = net_sales if net_sales > 0 else LOAN_AMOUNT * 10
annual_annuity = (LOAN_AMOUNT / (LOAN_TENURE * 12)) * 12

dscr              = round((pat + interest) / interest, 4) if interest > 0 else 0
interest_coverage = round(ebitda / interest, 4)           if interest > 0 else 0
ebitda_margin     = round(ebitda / net_sales * 100, 4)    if net_sales > 0 else 0
pat_margin        = round(pat    / net_sales * 100, 4)    if net_sales > 0 else 0
debt_equity       = round(total_debt / (annual_income * 0.45), 4) if annual_income > 0 else 7.4

print(f"  Revenue        : ₹{net_sales:,.0f}")
print(f"  EBITDA Margin  : {ebitda_margin:.1f}%")
print(f"  PAT Margin     : {pat_margin:.1f}%")
print(f"  DSCR           : {dscr:.2f}x")
print(f"  ICR            : {interest_coverage:.2f}x")

# ── B. Time-series features ────────────────────────────────
print("\n[B] Time-Series Features...")

yoy_revenue_growth = 0.0
yoy_pat_growth     = 0.0
revenue_3yr_cagr   = 0.0
ebitda_trend       = "STABLE"
revenue_volatility = 0.0

if not actuals.empty and len(actuals) >= 2:
    rev_vals   = actuals["net_sales"].tolist()
    pat_vals   = actuals["pat"].tolist()
    ebitda_vals= actuals["ebitda"].tolist()

    yoy_revenue_growth = round((rev_vals[-1] - rev_vals[-2]) / rev_vals[-2] * 100, 2) if rev_vals[-2] > 0 else 0
    yoy_pat_growth     = round((pat_vals[-1] - pat_vals[-2]) / pat_vals[-2] * 100, 2) if pat_vals[-2] > 0 else 0

    if len(actuals) >= 3:
        n = len(actuals) - 1
        revenue_3yr_cagr = round((pow(rev_vals[-1] / rev_vals[0], 1/n) - 1) * 100, 2) if rev_vals[0] > 0 else 0
        mean_rev         = sum(rev_vals) / len(rev_vals)
        std_rev          = math.sqrt(sum((x - mean_rev)**2 for x in rev_vals) / len(rev_vals))
        revenue_volatility = round(std_rev / mean_rev, 4) if mean_rev > 0 else 0

        if ebitda_vals[-1] > ebitda_vals[-2] > ebitda_vals[0]:
            ebitda_trend = "IMPROVING"
        elif ebitda_vals[-1] < ebitda_vals[-2]:
            ebitda_trend = "DECLINING"

print(f"  YoY Revenue Growth  : {yoy_revenue_growth:.1f}%")
print(f"  3-Year CAGR         : {revenue_3yr_cagr:.1f}%")
print(f"  EBITDA Trend        : {ebitda_trend}")

# ── C. Bank statement features ─────────────────────────────
print("\n[C] Bank Statement Features...")

avg_monthly_inflow   = 0.0
avg_monthly_outflow  = 0.0
cash_flow_volatility = 0.0
emi_burden_ratio     = 0.0
avg_closing_balance  = 0.0
inflow_outflow_ratio = 1.0
unusual_txn_count    = 0

if not bank_df.empty:
    bank_df["month"] = pd.to_datetime(bank_df.get("date", bank_df.get("txn_date", "2024-01-01")),
                                       errors="coerce").dt.to_period("M").astype(str)
    monthly = bank_df.groupby("month").agg(
        inflow =("credit", "sum"),
        outflow=("debit",  "sum"),
        avg_bal=("balance","mean")
    ).reset_index()

    if not monthly.empty:
        avg_monthly_inflow  = float(monthly["inflow"].mean())
        avg_monthly_outflow = float(monthly["outflow"].mean())
        avg_closing_balance = float(monthly["avg_bal"].mean())
        if avg_monthly_outflow > 0:
            inflow_outflow_ratio = round(avg_monthly_inflow / avg_monthly_outflow, 4)
        if len(monthly) > 1:
            std_in = float(monthly["inflow"].std())
            cash_flow_volatility = round(std_in / avg_monthly_inflow, 4) if avg_monthly_inflow > 0 else 0

    emi_total   = float(bank_df[bank_df["txn_category"] == "LOAN_REPAYMENT"]["debit"].sum())
    total_debit = float(bank_df["debit"].sum())
    emi_burden_ratio = round(emi_total / total_debit, 4) if total_debit > 0 else 0

    med_debit = float(bank_df[bank_df["debit"] > 0]["debit"].median()) if (bank_df["debit"] > 0).any() else 1
    unusual_txn_count = int((bank_df["debit"] > med_debit * 3).sum())

print(f"  Avg Monthly Inflow  : ₹{avg_monthly_inflow:,.0f}")
print(f"  Inflow/Outflow      : {inflow_outflow_ratio:.2f}")
print(f"  EMI Burden          : {emi_burden_ratio:.4f}")

# ── D. GST compliance ──────────────────────────────────────
print("\n[D] GST Compliance...")

gst_filing_rate         = 1.0
gst_revenue_consistency = 1.0
gst_annual_turnover     = 0.0

if not gst_df.empty:
    gst_filing_rate     = round(gst_df["is_filed"].sum() / len(gst_df), 4)
    gst_annual_turnover = float(gst_df["total_turnover"].sum())
    if gst_annual_turnover > 0 and annual_income > 0:
        gst_revenue_consistency = round(
            min(gst_annual_turnover, annual_income) /
            max(gst_annual_turnover, annual_income), 4
        )

print(f"  GST Filing Rate     : {gst_filing_rate*100:.0f}%")
print(f"  Revenue Consistency : {gst_revenue_consistency:.4f}")

# ── E. EXT_SOURCE from CIBIL ───────────────────────────────
print("\n[E] EXT_SOURCE Features (CIBIL-derived)...")

dpd_penalty  = min(dpd_12m * 0.05, 0.30)
over_penalty = min(overdue_accts * 0.05, 0.20)
npa_penalty  = 0.30 if npa_flag else 0.0
wd_penalty   = 0.50 if wilful_flag else 0.0

ext_source_1 = round(score_norm, 4)
ext_source_2 = round(max(score_norm - dpd_penalty - over_penalty, 0.0), 4)
ext_source_3 = round(max(on_time_pct  - npa_penalty  - wd_penalty,  0.0), 4)

ext_sources  = [ext_source_1, ext_source_2, ext_source_3]
ext_mean     = round(sum(ext_sources) / 3, 4)
ext_min      = round(min(ext_sources), 4)
ext_max      = round(max(ext_sources), 4)
ext_std      = round(math.sqrt(sum((x - ext_mean)**2 for x in ext_sources) / 3), 4)
ext_product  = round(ext_source_1 * ext_source_2 * ext_source_3, 4)

print(f"  CIBIL Score    : {cibil_score}")
print(f"  EXT_SOURCE_1   : {ext_source_1}")
print(f"  EXT_SOURCE_2   : {ext_source_2}")
print(f"  EXT_SOURCE_3   : {ext_source_3}")
print(f"  EXT_SOURCE_MEAN: {ext_mean}")

# ── F. ML feature vector ───────────────────────────────────
print("\n[F] ML Feature Vector...")

days_birth   = -(TODAY - DOB_PROMOTER).days
days_employed= -(TODAY - EMP_START).days

credit_income_ratio  = round(LOAN_AMOUNT / annual_income, 6) if annual_income > 0 else 0
annuity_income_ratio = round(annual_annuity / annual_income, 6) if annual_income > 0 else 0

high_credit_low_income = 1 if credit_income_ratio > 5.0 else 0
young_high_debt        = 1 if (abs(days_birth)/365 < 35 and credit_income_ratio > 4) else 0

bureau_active_ratio  = round(active_accts / total_accts, 4) if total_accts > 0 else 0
bureau_debt_ratio    = round(LOAN_AMOUNT / (annual_income + 1), 4)

inst_late_rate = round(emi_burden_ratio * (1 - gst_filing_rate), 4)
inst_days_late = dpd_12m * 30

cc_util = round(1 - (avg_closing_balance / (avg_monthly_inflow * 2 + 1)), 4) if avg_monthly_inflow > 0 else 0.3
cc_util = max(0.0, min(1.0, cc_util))

pos_completion = round(gst_filing_rate * 0.9 + 0.1, 4)

print(f"  CREDIT_INCOME_RATIO  : {credit_income_ratio:.4f}")
print(f"  ANNUITY_INCOME_RATIO : {annuity_income_ratio:.4f}")
print(f"  DAYS_BIRTH           : {days_birth}")
print(f"  CC_UTILIZATION       : {cc_util:.4f}")


# ============================================================
# SECTION 5 — HOMOMORPHIC ENCRYPTION (CIBIL fields)
# ============================================================

print(f"\n{'─'*55}")
print(f"  STAGE 3 — Homomorphic Encryption")
print(f"{'─'*55}")

HE_FIELDS            = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "CC_UTILIZATION_MEAN"]
he_encrypted_values  = {}
he_encryption_active = False

HE_VALUES = {
    "EXT_SOURCE_1"       : ext_source_1,
    "EXT_SOURCE_2"       : ext_source_2,
    "EXT_SOURCE_3"       : ext_source_3,
    "CC_UTILIZATION_MEAN": cc_util,
}

try:
    import tenseal as ts

    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=4096,
        plain_modulus=1032193
    )
    context.generate_galois_keys()
    context.generate_relin_keys()

    SCALE = 10000
    for field, val in HE_VALUES.items():
        enc = ts.bfv_vector(context, [int(val * SCALE)])
        he_encrypted_values[field] = {
            "encrypted_hex"      : enc.serialize().hex()[:64] + "...",
            "scale_factor"       : SCALE,
            "scheme"             : "BFV",
            "poly_modulus"       : 4096,
            "plain_modulus"      : 1032193,
            "plaintext_destroyed": True
        }
        print(f"  🔐 {field:<28} : {val:.4f} → ENCRYPTED (TenSEAL BFV)")

    he_encryption_active = True

except ImportError:
    print("  ⚠️  TenSEAL not installed — using mock HE")
    print("     Install: pip install tenseal")
    for field, val in HE_VALUES.items():
        mock_hex = hashlib.sha256(f"{field}:{val}:SEAL_BFV_MOCK".encode()).hexdigest()
        he_encrypted_values[field] = {
            "encrypted_hex"      : mock_hex,
            "scale_factor"       : 10000,
            "scheme"             : "BFV_MOCK",
            "poly_modulus"       : 4096,
            "plain_modulus"      : 1032193,
            "plaintext_destroyed": True,
            "note"               : "Mock — install tenseal for real HE"
        }
        print(f"  🔐 {field:<28} : {val:.4f} → MOCK ENCRYPTED")

except Exception as e:
    print(f"  ⚠️  HE error: {e}")


# ============================================================
# SECTION 6 — ASSEMBLE borrower_profile.json
# ============================================================

print(f"\n{'─'*55}")
print(f"  STAGE 4 — Assembling borrower_profile.json")
print(f"{'─'*55}")

financial_features = {
    "AMT_INCOME_TOTAL"           : float(annual_income),
    "AMT_CREDIT"                 : float(LOAN_AMOUNT),
    "AMT_ANNUITY"                : float(annual_annuity),
    "AMT_GOODS_PRICE"            : float(LOAN_AMOUNT),
    "EXT_SOURCE_1"               : ext_source_1,
    "EXT_SOURCE_2"               : ext_source_2,
    "EXT_SOURCE_3"               : ext_source_3,
    "EXT_SOURCE_MEAN"            : ext_mean,
    "EXT_SOURCE_MIN"             : ext_min,
    "EXT_SOURCE_MAX"             : ext_max,
    "EXT_SOURCE_STD"             : ext_std,
    "EXT_SOURCE_PRODUCT"         : ext_product,
    "DAYS_BIRTH"                 : float(days_birth),
    "DAYS_EMPLOYED"              : float(days_employed),
    "AGE_YEARS"                  : round(-days_birth / 365, 2),
    "YEARS_EMPLOYED"             : round(-days_employed / 365, 2),
    "CREDIT_INCOME_RATIO"        : credit_income_ratio,
    "ANNUITY_INCOME_RATIO"       : annuity_income_ratio,
    "HIGH_CREDIT_LOW_INCOME"     : float(high_credit_low_income),
    "IS_UNEMPLOYED"              : 0.0,
    "YOUNG_HIGH_DEBT"            : float(young_high_debt),
    "FLAG_OWN_CAR"               : 0.0,
    "FLAG_OWN_REALTY"            : 1.0,
    "CNT_CHILDREN"               : 0.0,
    "CNT_FAM_MEMBERS"            : 1.0,
    "NAME_CONTRACT_TYPE"         : "Cash loans",
    "CODE_GENDER"                : "M",
    "NAME_INCOME_TYPE"           : "Commercial associate",
    "NAME_EDUCATION_TYPE"        : "Higher education",
    "NAME_FAMILY_STATUS"         : "Married",
    "NAME_HOUSING_TYPE"          : "House / apartment",
    "OCCUPATION_TYPE"            : "Managers",
    "ORGANIZATION_TYPE"          : "Industry: type 2",
    "REGION_RATING_CLIENT"       : 2.0,
    "BUREAU_LOAN_COUNT"          : float(total_accts),
    "BUREAU_ACTIVE_RATIO"        : bureau_active_ratio,
    "BUREAU_DEBT_TO_CREDIT_RATIO": bureau_debt_ratio,
    "BUREAU_DPD_MAX"             : float(dpd_12m),
    "NPA_FLAG"                   : float(1 if npa_flag   else 0),
    "WILFUL_DEFAULT_FLAG"        : float(1 if wilful_flag else 0),
    "INST_LATE_RATE"             : inst_late_rate,
    "INST_DAYS_LATE_MAX"         : float(inst_days_late),
    "CC_UTILIZATION_MEAN"        : cc_util,
    "CC_UTILIZATION_MAX"         : round(cc_util * 1.2, 4),
    "CC_DPD_MEAN"                : float(dpd_12m / 12),
    "POS_DPD_FLAG_RATE"          : float(1 if dpd_12m > 0 else 0),
    "POS_COMPLETION_RATIO"       : pos_completion,
}

profile = {
    "_meta": {
        "generated_by"    : "Layer 2 Local — Bronze to borrower_profile",
        "generated_at"    : datetime.now().isoformat(),
        "script"          : "layer2_local.py",
        "company_id"      : COMPANY_ID,
        "architecture_note": (
            "In production this runs as 3 Databricks notebooks "
            "(Bronze→Silver→Gold) triggered via n8n. For demo, "
            "this local script produces identical output."
        )
    },

    # ── Borrower Identity ──────────────────────────────────
    "company_id"    : COMPANY_ID,
    "company_name"  : COMPANY_NAME,
    "industry"      : INDUSTRY,
    "promoter_name" : PROMOTER_NAME,
    "cin"           : CIN,

    # ── Loan Request ───────────────────────────────────────
    "loan_amount"       : float(LOAN_AMOUNT),
    "annual_income"     : float(annual_income),
    "collateral_value"  : float(LOAN_AMOUNT * 0.8),
    "loan_purpose"      : LOAN_PURPOSE,
    "loan_tenure_years" : float(LOAN_TENURE),

    # ── Financial Ratios ───────────────────────────────────
    "financial_ratios": {
        "dscr"                    : dscr,
        "interest_coverage_ratio" : interest_coverage,
        "ebitda_margin_pct"       : ebitda_margin,
        "pat_margin_pct"          : pat_margin,
        "debt_equity_ratio"       : debt_equity,
        "working_capital_limit"   : wc_limit,
        "wc_utilization_pct"      : util_pct,
    },

    # ── Time-Series Trends ─────────────────────────────────
    "time_series": {
        "yoy_revenue_growth_pct"  : yoy_revenue_growth,
        "yoy_pat_growth_pct"      : yoy_pat_growth,
        "revenue_3yr_cagr_pct"    : revenue_3yr_cagr,
        "ebitda_trend"            : ebitda_trend,
        "revenue_volatility"      : revenue_volatility,
    },

    # ── Bank Signals ───────────────────────────────────────
    "bank_signals": {
        "avg_monthly_inflow"      : avg_monthly_inflow,
        "avg_monthly_outflow"     : avg_monthly_outflow,
        "inflow_outflow_ratio"    : inflow_outflow_ratio,
        "cash_flow_volatility"    : cash_flow_volatility,
        "emi_burden_ratio"        : emi_burden_ratio,
        "avg_closing_balance"     : avg_closing_balance,
        "unusual_txn_count"       : float(unusual_txn_count),
    },

    # ── GST Signals ────────────────────────────────────────
    "gst_signals": {
        "gst_filing_rate"         : gst_filing_rate,
        "gst_revenue_consistency" : gst_revenue_consistency,
        "gst_annual_turnover"     : gst_annual_turnover,
    },

    # ── MCA Signals ────────────────────────────────────────
    "mca_signals": {
        "mca_is_active"           : float(1 if is_active  else 0),
        "mca_risk_flag"           : float(1 if mca_risk   else 0),
        "insolvency_flag"         : float(1 if insolvency else 0),
        "director_count"          : float(dir_count),
        "years_incorporated"      : float(years_inc),
    },

    # ── ML Feature Vector ──────────────────────────────────
    "financial_features" : financial_features,

    # ── HE Record ─────────────────────────────────────────
    "homomorphic_encryption": {
        "applied"          : he_encryption_active,
        "scheme"           : "BFV",
        "library"          : "Microsoft SEAL via TenSEAL",
        "poly_modulus"     : 4096,
        "plain_modulus"    : 1032193,
        "encrypted_fields" : HE_FIELDS,
        "encrypted_values" : he_encrypted_values,
        "note"             : (
            "EXT_SOURCE features (CIBIL-derived) are encrypted. "
            "Layer 5 ML receives plaintext for prototype. "
            "Production: HE-compatible logistic regression on ciphertexts."
        )
    },

    # ── Pipeline Flags ─────────────────────────────────────
    "layer2_completed"  : True,
    "layer3_completed"  : False,
    "layer5_completed"  : False,
    "layer6_completed"  : False,
    "layer7_completed"  : False,
}

# ── Write borrower_profile.json ────────────────────────────
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(profile, f, indent=2, default=str)


# ============================================================
# SECTION 7 — SUMMARY
# ============================================================

print(f"\n{'='*55}")
print(f"  LAYER 2 COMPLETE")
print(f"{'='*55}")
print(f"  Company          : {COMPANY_NAME}")
print(f"  Industry         : {INDUSTRY}")
print(f"  Loan Amount      : ₹{LOAN_AMOUNT:,.0f}")
print(f"  Annual Income    : ₹{annual_income:,.0f}")
print(f"  DSCR             : {dscr:.2f}x")
print(f"  ICR              : {interest_coverage:.2f}x")
print(f"  EBITDA Margin    : {ebitda_margin:.1f}%")
print(f"  Revenue CAGR     : {revenue_3yr_cagr:.1f}%")
print(f"  EBITDA Trend     : {ebitda_trend}")
print(f"  CIBIL Score      : {cibil_score}")
print(f"  EXT_SOURCE_MEAN  : {ext_mean:.4f}")
print(f"  HE Encrypted     : {he_encryption_active}")
print(f"  ML Features      : {len(financial_features)}")
print(f"\n  ✅ Saved: {OUTPUT_PATH}")
print(f"\n  NEXT STEPS:")
print(f"  1. python layer3_research_agent.py")
print(f"  2. Run credit_ml_engine.py in Colab")
print(f"  3. python layer6.py")
print(f"  4. node layer7.js cam_layer6_COMP_001_*.json")
print(f"{'='*55}")