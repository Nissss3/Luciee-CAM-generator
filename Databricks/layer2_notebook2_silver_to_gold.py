# Databricks notebook source
# MAGIC %md
# MAGIC # Layer 2 — Notebook 2: Silver → Gold
# MAGIC **Credit Decisioning Engine**
# MAGIC
# MAGIC Reads clean Silver Delta tables and engineers all features needed by:
# MAGIC - Layer 5 ML Model (Home Credit schema features)
# MAGIC - Layer 3 Research Agent (company identity fields)
# MAGIC - Layer 7 CAM Generator (financial ratios, narrative data)
# MAGIC
# MAGIC **Features computed:**
# MAGIC - Financial ratios: DSCR, ICR, D/E, Current Ratio, Working Capital Days
# MAGIC - Time-series: 3-year YoY growth rates, trend direction
# MAGIC - ML features: EXT_SOURCE_1/2/3, DAYS_BIRTH, CREDIT_INCOME_RATIO, etc.
# MAGIC - Bank behaviour: avg monthly inflow, EMI burden, cash flow volatility
# MAGIC - GST compliance: filing rate, revenue consistency vs financials
# MAGIC - HE flag: marks CIBIL fields for encryption in Notebook 3

# COMMAND ----------
# MAGIC %md ## 0. Setup

# COMMAND ----------

import json
import math
import os
from datetime import datetime, date
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, BooleanType, LongType
)

spark = SparkSession.builder.appName("CreditEngine_Layer2_Silver_Gold").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
spark.conf.set("spark.databricks.delta.preview.enabled", "true")

# ── Read parameters from n8n (passed via notebook_task base_parameters) ─
def get_param(name, default):
    try:
        return dbutils.widgets.get(name)
    except:
        return default

COMPANY_ID    = get_param("COMPANY_ID",    "COMP_001")
COMPANY_NAME  = get_param("COMPANY_NAME",  "Tata Capital Limited")
INDUSTRY      = get_param("INDUSTRY",      "Banking & Financial Services")
PROMOTER_NAME = get_param("PROMOTER_NAME", "Rajiv Sabharwal")
CIN           = get_param("CIN",           "U65990MH1991PLC060670")
LOAN_AMOUNT   = float(get_param("LOAN_AMOUNT",  "500000000"))
LOAN_TENURE   = int(get_param("LOAN_TENURE",    "5"))
LOAN_PURPOSE  = get_param("LOAN_PURPOSE",  "Working Capital")
DOB_PROMOTER  = date(1968, 4, 15)   # Promoter DOB — update per borrower
EMP_START     = date(2021, 4, 1)    # Employment/incorporation start date

# ── Paths ─────────────────────────────────────────────────────
SILVER_ROOT  = f"/Workspace/CreditEngine/silver/{COMPANY_ID}"
GOLD_ROOT    = f"/Workspace/CreditEngine/gold/{COMPANY_ID}"
os.makedirs(GOLD_ROOT, exist_ok=True)

print(f"✅ Configuration loaded")
print(f"   Company  : {COMPANY_NAME}")
print(f"   Silver   : {SILVER_ROOT}")
print(f"   Gold     : {GOLD_ROOT}")

TODAY = date.today()

# COMMAND ----------
# MAGIC %md ## 1. Load Silver Tables

# COMMAND ----------

def load_silver(table_name):
    path = f"file:{SILVER_ROOT}/{table_name}"
    try:
        df = spark.read.format("delta").load(path)
        print(f"  ✅ Loaded silver/{table_name} — {df.count()} rows")
        return df
    except Exception as e:
        print(f"  ⚠️  Could not load silver/{table_name}: {e}")
        return None

print("Loading Silver tables...")
fin_df    = load_silver("financials")
bank_df   = load_silver("bank_statements")
gst_df    = load_silver("gst_returns")
cibil_df  = load_silver("cibil")
mca_df    = load_silver("mca")
cma_df    = load_silver("cma")


# COMMAND ----------
# MAGIC %md ## 2. Financial Ratio Engineering

# COMMAND ----------

print("\n[A] Engineering Financial Ratios...")

# ── Use CMA data (5 periods) as primary financial source ───────
# Falls back to financials table if CMA not available

if cma_df:
    # Actual periods only for ratio calculation
    actual_cma = cma_df.filter(~F.col("is_projected")).orderBy("period")
    latest_cma = actual_cma.orderBy(F.col("period").desc()).limit(1).collect()

    if latest_cma:
        r = latest_cma[0]
        net_sales    = float(r["net_sales"])
        ebitda       = float(r["ebitda"])
        interest     = float(r["interest"])
        pat          = float(r["pat"])
        total_debt   = float(r["total_debt"]) if r["total_debt"] else net_sales * 7.2
        wc_limit     = float(r["working_capital_limit"])
        utilization  = float(r["utilization_pct"])

        # ── Core ratios ────────────────────────────────────────
        dscr              = round((pat + interest) / interest, 4) if interest > 0 else 0
        interest_coverage = round(ebitda / interest, 4) if interest > 0 else 0
        pat_margin        = round(pat / net_sales * 100, 4) if net_sales > 0 else 0
        ebitda_margin     = round(ebitda / net_sales * 100, 4) if net_sales > 0 else 0

        print(f"  Revenue        : ₹{net_sales:,.0f}")
        print(f"  EBITDA Margin  : {ebitda_margin:.1f}%")
        print(f"  PAT Margin     : {pat_margin:.1f}%")
        print(f"  DSCR           : {dscr:.2f}x")
        print(f"  ICR            : {interest_coverage:.2f}x")
    else:
        # Defaults
        net_sales = ebitda = interest = pat = total_debt = 1
        wc_limit = utilization = dscr = interest_coverage = 0
        pat_margin = ebitda_margin = 0
else:
    net_sales = ebitda = interest = pat = total_debt = 1
    wc_limit = utilization = dscr = interest_coverage = 0
    pat_margin = ebitda_margin = 0

# ── Balance sheet ratios from financials table ─────────────────
if fin_df:
    fin_latest = fin_df.filter(F.col("is_valid")).orderBy(
        F.col("fiscal_year").desc()).limit(1).collect()
    if fin_latest:
        fr = fin_latest[0]
        fin_revenue   = float(fr["revenue"]) if fr["revenue"] else net_sales
        fin_pat       = float(fr["pat"]) if fr["pat"] else pat
    else:
        fin_revenue = net_sales
        fin_pat     = pat
else:
    fin_revenue = net_sales
    fin_pat     = pat

# ── Annual income for ML model ─────────────────────────────────
annual_income = net_sales if net_sales > 0 else fin_revenue
annuity       = LOAN_AMOUNT / (LOAN_TENURE * 12)  # monthly EMI approximation
annual_annuity = annuity * 12


# COMMAND ----------
# MAGIC %md ## 3. Time-Series Feature Engineering

# COMMAND ----------

print("\n[B] Engineering Time-Series Features...")

yoy_revenue_growth    = 0.0
yoy_pat_growth        = 0.0
revenue_3yr_cagr      = 0.0
ebitda_trend          = "STABLE"
revenue_volatility    = 0.0

if cma_df:
    actuals = actual_cma.collect()
    if len(actuals) >= 2:
        # YoY growth (latest vs previous year)
        rev_latest   = float(actuals[-1]["net_sales"])
        rev_prev     = float(actuals[-2]["net_sales"])
        pat_latest   = float(actuals[-1]["pat"])
        pat_prev     = float(actuals[-2]["pat"])

        yoy_revenue_growth = round((rev_latest - rev_prev) / rev_prev * 100, 2) if rev_prev > 0 else 0
        yoy_pat_growth     = round((pat_latest - pat_prev) / pat_prev * 100, 2) if pat_prev > 0 else 0

    if len(actuals) >= 3:
        # 3-year CAGR
        rev_start = float(actuals[0]["net_sales"])
        rev_end   = float(actuals[-1]["net_sales"])
        n_years   = len(actuals) - 1
        revenue_3yr_cagr = round((pow(rev_end / rev_start, 1 / n_years) - 1) * 100, 2) if rev_start > 0 else 0

        # EBITDA trend
        ebitda_vals = [float(r["ebitda"]) for r in actuals]
        if ebitda_vals[-1] > ebitda_vals[-2] > ebitda_vals[0]:
            ebitda_trend = "IMPROVING"
        elif ebitda_vals[-1] < ebitda_vals[-2]:
            ebitda_trend = "DECLINING"
        else:
            ebitda_trend = "STABLE"

        # Revenue volatility (std / mean)
        rev_vals = [float(r["net_sales"]) for r in actuals]
        mean_rev = sum(rev_vals) / len(rev_vals)
        std_rev  = math.sqrt(sum((x - mean_rev)**2 for x in rev_vals) / len(rev_vals))
        revenue_volatility = round(std_rev / mean_rev, 4) if mean_rev > 0 else 0

print(f"  YoY Revenue Growth  : {yoy_revenue_growth:.1f}%")
print(f"  3-Year Revenue CAGR : {revenue_3yr_cagr:.1f}%")
print(f"  YoY PAT Growth      : {yoy_pat_growth:.1f}%")
print(f"  EBITDA Trend        : {ebitda_trend}")
print(f"  Revenue Volatility  : {revenue_volatility:.4f}")


# COMMAND ----------
# MAGIC %md ## 4. Bank Statement Features

# COMMAND ----------

print("\n[C] Engineering Bank Statement Features...")

avg_monthly_inflow     = 0.0
avg_monthly_outflow    = 0.0
cash_flow_volatility   = 0.0
emi_burden_ratio       = 0.0
avg_closing_balance    = 0.0
inflow_outflow_ratio   = 0.0
unusual_txn_count      = 0

if bank_df:
    # Monthly aggregates
    bank_monthly = bank_df \
        .withColumn("month", F.substring("txn_date", 1, 7)) \
        .groupBy("month") \
        .agg(
            F.sum("credit").alias("monthly_inflow"),
            F.sum("debit").alias("monthly_outflow"),
            F.avg("balance").alias("avg_balance"),
            F.count("*").alias("txn_count")
        ).collect()

    if bank_monthly:
        inflows  = [float(r["monthly_inflow"])  for r in bank_monthly]
        outflows = [float(r["monthly_outflow"]) for r in bank_monthly]
        balances = [float(r["avg_balance"])     for r in bank_monthly]

        avg_monthly_inflow   = round(sum(inflows) / len(inflows), 0)
        avg_monthly_outflow  = round(sum(outflows) / len(outflows), 0)
        avg_closing_balance  = round(sum(balances) / len(balances), 0)

        # Cash flow volatility
        if len(inflows) > 1:
            mean_in = avg_monthly_inflow
            std_in  = math.sqrt(sum((x - mean_in)**2 for x in inflows) / len(inflows))
            cash_flow_volatility = round(std_in / mean_in, 4) if mean_in > 0 else 0

        inflow_outflow_ratio = round(avg_monthly_inflow / avg_monthly_outflow, 4) if avg_monthly_outflow > 0 else 1

    # EMI burden: loan repayment transactions vs total outflow
    emi_total = bank_df.filter(
        F.col("txn_category").isin("LOAN_REPAYMENT")
    ).agg(F.sum("debit")).collect()[0][0] or 0
    total_outflow = bank_df.agg(F.sum("debit")).collect()[0][0] or 1
    emi_burden_ratio = round(float(emi_total) / float(total_outflow), 4)

    # Unusual transactions
    unusual_txn_count = bank_df.filter(F.col("is_unusual_txn")).count()

print(f"  Avg Monthly Inflow  : ₹{avg_monthly_inflow:,.0f}")
print(f"  Avg Monthly Outflow : ₹{avg_monthly_outflow:,.0f}")
print(f"  Inflow/Outflow Ratio: {inflow_outflow_ratio:.2f}")
print(f"  EMI Burden Ratio    : {emi_burden_ratio:.4f}")
print(f"  Cash Flow Volatility: {cash_flow_volatility:.4f}")
print(f"  Avg Closing Balance : ₹{avg_closing_balance:,.0f}")
print(f"  Unusual Transactions: {unusual_txn_count}")


# COMMAND ----------
# MAGIC %md ## 5. GST Compliance Features

# COMMAND ----------

print("\n[D] Engineering GST Compliance Features...")

gst_filing_rate         = 1.0
gst_revenue_consistency = 1.0
total_gst_annual        = 0.0
gst_annual_turnover     = 0.0

if gst_df:
    total_quarters = gst_df.count()
    filed_quarters = gst_df.filter(F.col("is_filed")).count()
    gst_filing_rate = round(filed_quarters / total_quarters, 4) if total_quarters > 0 else 1.0

    gst_agg = gst_df.agg(
        F.sum("total_turnover").alias("annual_turnover"),
        F.sum("net_gst_paid").alias("annual_gst")
    ).collect()[0]

    gst_annual_turnover = float(gst_agg["annual_turnover"] or 0)
    total_gst_annual    = float(gst_agg["annual_gst"] or 0)

    # Cross-check GST turnover vs financial revenue
    # GST turnover should be within 20% of reported revenue
    if gst_annual_turnover > 0 and annual_income > 0:
        gst_revenue_consistency = round(min(gst_annual_turnover, annual_income) /
                                        max(gst_annual_turnover, annual_income), 4)

print(f"  GST Filing Rate        : {gst_filing_rate*100:.0f}%")
print(f"  GST Annual Turnover    : ₹{gst_annual_turnover:,.0f}")
print(f"  Financial Revenue      : ₹{annual_income:,.0f}")
print(f"  Revenue Consistency    : {gst_revenue_consistency:.4f}")


# COMMAND ----------
# MAGIC %md ## 6. CIBIL Features + EXT_SOURCE Mapping

# COMMAND ----------

print("\n[E] Engineering CIBIL / Bureau Features...")

ext_source_1   = 0.65   # Defaults
ext_source_2   = 0.65
ext_source_3   = 0.65
cibil_score    = 700
dpd_12m        = 0
overdue_accts  = 0
npa_flag       = False
wilful_default = False

if cibil_df:
    cr = cibil_df.collect()[0]

    cibil_score   = int(cr["cibil_score"])
    score_norm    = float(cr["score_normalized"])
    dpd_12m       = int(cr["dpd_last_12_months"])
    overdue_accts = int(cr["overdue_accounts"])
    npa_flag      = bool(cr["npa_flag"])
    wilful_default= bool(cr["wilful_defaulter_flag"])
    on_time_pct   = float(cr["on_time_payment_pct"]) / 100

    # Map to 3 EXT_SOURCE features used by ML model
    # EXT_SOURCE_1: Raw CIBIL normalized (300-900 → 0-1)
    ext_source_1 = round(score_norm, 4)

    # EXT_SOURCE_2: Payment behaviour composite
    # Penalize DPD and overdue accounts
    dpd_penalty   = min(dpd_12m * 0.05, 0.30)
    over_penalty  = min(overdue_accts * 0.05, 0.20)
    ext_source_2  = round(max(score_norm - dpd_penalty - over_penalty, 0.0), 4)

    # EXT_SOURCE_3: On-time payment rate with NPA penalty
    npa_penalty   = 0.30 if npa_flag else 0.0
    wd_penalty    = 0.50 if wilful_default else 0.0
    ext_source_3  = round(max(on_time_pct - npa_penalty - wd_penalty, 0.0), 4)

print(f"  CIBIL Score    : {cibil_score}")
print(f"  EXT_SOURCE_1   : {ext_source_1} (normalized score)")
print(f"  EXT_SOURCE_2   : {ext_source_2} (payment behaviour)")
print(f"  EXT_SOURCE_3   : {ext_source_3} (on-time rate)")
print(f"  DPD 12M        : {dpd_12m}")
print(f"  NPA Flag       : {npa_flag}")
print(f"  ⚠️  HE Encryption: CIBIL fields will be encrypted in Notebook 3")


# COMMAND ----------
# MAGIC %md ## 7. MCA Features

# COMMAND ----------

print("\n[F] Engineering MCA Features...")

is_active            = True
mca_risk_flag        = False
insolvency_flag      = False
director_count       = 5
years_incorporated   = 10

if mca_df:
    mr = mca_df.collect()[0]
    is_active          = bool(mr["is_active"])
    mca_risk_flag      = bool(mr["mca_risk_flag"])
    insolvency_flag    = bool(mr["insolvency_proceedings"])
    director_count     = int(mr["director_count"])

    doi = str(mr["date_of_incorporation"])
    try:
        doi_date = datetime.strptime(doi[:10], "%Y-%m-%d").date()
        years_incorporated = (TODAY - doi_date).days // 365
    except:
        years_incorporated = 10

print(f"  Company Active    : {is_active}")
print(f"  MCA Risk Flag     : {mca_risk_flag}")
print(f"  Insolvency        : {insolvency_flag}")
print(f"  Directors         : {director_count}")
print(f"  Years Inc.        : {years_incorporated}")


# COMMAND ----------
# MAGIC %md ## 8. ML Feature Vector (Home Credit Schema)

# COMMAND ----------

print("\n[G] Building ML Feature Vector (Home Credit Schema)...")

# ── Date-based features ────────────────────────────────────────
days_birth     = -(TODAY - DOB_PROMOTER).days        # negative, per Home Credit schema
days_employed  = -(TODAY - EMP_START).days           # negative

# ── Core ML features ───────────────────────────────────────────
credit_income_ratio   = round(LOAN_AMOUNT / annual_income, 6) if annual_income > 0 else 0
annuity_income_ratio  = round(annual_annuity / annual_income, 6) if annual_income > 0 else 0

# EXT_SOURCE composite (mean, min, max, std for ensemble)
ext_sources    = [ext_source_1, ext_source_2, ext_source_3]
ext_mean       = round(sum(ext_sources) / 3, 4)
ext_min        = round(min(ext_sources), 4)
ext_max        = round(max(ext_sources), 4)
ext_std        = round(math.sqrt(sum((x - ext_mean)**2 for x in ext_sources) / 3), 4)
ext_product    = round(ext_source_1 * ext_source_2 * ext_source_3, 4)

# Risk flags
high_credit_low_income = 1 if credit_income_ratio > 5.0 else 0
is_unemployed          = 0  # company is operating
young_high_debt        = 1 if (abs(days_birth) / 365 < 35 and credit_income_ratio > 4) else 0

# Bureau features (from CIBIL)
bureau_loan_count    = int(cibil_df.collect()[0]["total_accounts"]) if cibil_df else 5
bureau_active        = int(cibil_df.collect()[0]["active_accounts"]) if cibil_df else 3
bureau_dpd_max       = dpd_12m
bureau_active_ratio  = round(bureau_active / bureau_loan_count, 4) if bureau_loan_count > 0 else 0
bureau_debt_ratio    = round(LOAN_AMOUNT / (annual_income + 1), 4)

# Installment features (from bank statement EMI behaviour)
inst_late_rate  = round(emi_burden_ratio * (1 - gst_filing_rate), 4)  # proxy
inst_days_late  = dpd_12m * 30  # approx days late

# Credit card features (from bank utilization proxy)
cc_utilization  = round(1 - (avg_closing_balance / (avg_monthly_inflow * 2 + 1)), 4) if avg_monthly_inflow > 0 else 0.3
cc_utilization  = max(0.0, min(1.0, cc_utilization))  # clamp 0-1

# POS features (payment consistency proxy)
pos_dpd_flag    = 1 if dpd_12m > 0 else 0
pos_completion  = round(gst_filing_rate * 0.9 + 0.1, 4)

print(f"  AMT_INCOME_TOTAL      : ₹{annual_income:,.0f}")
print(f"  AMT_CREDIT            : ₹{LOAN_AMOUNT:,.0f}")
print(f"  AMT_ANNUITY           : ₹{annual_annuity:,.0f}")
print(f"  CREDIT_INCOME_RATIO   : {credit_income_ratio:.4f}")
print(f"  ANNUITY_INCOME_RATIO  : {annuity_income_ratio:.4f}")
print(f"  EXT_SOURCE_MEAN       : {ext_mean:.4f}")
print(f"  DAYS_BIRTH            : {days_birth}")
print(f"  DAYS_EMPLOYED         : {days_employed}")
print(f"  CC_UTILIZATION        : {cc_utilization:.4f}")
print(f"  INST_LATE_RATE        : {inst_late_rate:.4f}")


# COMMAND ----------
# MAGIC %md ## 9. Write Gold Feature Table

# COMMAND ----------

print("\n[H] Writing Gold Feature Table...")

gold_record = {
    # ── Borrower Identity ──────────────────────────────────────
    "company_id"                  : COMPANY_ID,
    "company_name"                : COMPANY_NAME,
    "industry"                    : INDUSTRY,
    "promoter_name"               : PROMOTER_NAME,
    "cin"                         : CIN,
    "loan_purpose"                : LOAN_PURPOSE,
    "loan_tenure_years"           : float(LOAN_TENURE),

    # ── Loan Request ───────────────────────────────────────────
    "AMT_CREDIT"                  : float(LOAN_AMOUNT),
    "AMT_INCOME_TOTAL"            : float(annual_income),
    "AMT_ANNUITY"                 : float(annual_annuity),
    "AMT_GOODS_PRICE"             : float(LOAN_AMOUNT),
    "loan_amount"                 : float(LOAN_AMOUNT),
    "annual_income"               : float(annual_income),
    "collateral_value"            : float(LOAN_AMOUNT * 0.8),

    # ── ML EXT_SOURCE Features (CIBIL-derived) ─────────────────
    # NOTE: These will be HE-encrypted in Notebook 3
    "EXT_SOURCE_1"                : ext_source_1,
    "EXT_SOURCE_2"                : ext_source_2,
    "EXT_SOURCE_3"                : ext_source_3,
    "EXT_SOURCE_MEAN"             : ext_mean,
    "EXT_SOURCE_MIN"              : ext_min,
    "EXT_SOURCE_MAX"              : ext_max,
    "EXT_SOURCE_STD"              : ext_std,
    "EXT_SOURCE_PRODUCT"          : ext_product,

    # ── Date-based Features ────────────────────────────────────
    "DAYS_BIRTH"                  : float(days_birth),
    "DAYS_EMPLOYED"               : float(days_employed),
    "AGE_YEARS"                   : round(-days_birth / 365, 2),
    "YEARS_EMPLOYED"              : round(-days_employed / 365, 2),

    # ── Income / Credit Ratio Features ────────────────────────
    "CREDIT_INCOME_RATIO"         : credit_income_ratio,
    "ANNUITY_INCOME_RATIO"        : annuity_income_ratio,
    "HIGH_CREDIT_LOW_INCOME"      : float(high_credit_low_income),
    "IS_UNEMPLOYED"               : float(is_unemployed),
    "YOUNG_HIGH_DEBT"             : float(young_high_debt),

    # ── Categorical Features ────────────────────────────────────
    "NAME_CONTRACT_TYPE"          : "Cash loans",
    "CODE_GENDER"                 : "M",
    "NAME_INCOME_TYPE"            : "Commercial associate",
    "NAME_EDUCATION_TYPE"         : "Higher education",
    "NAME_FAMILY_STATUS"          : "Married",
    "NAME_HOUSING_TYPE"           : "House / apartment",
    "OCCUPATION_TYPE"             : "Managers",
    "ORGANIZATION_TYPE"           : "Industry: type 2",
    "FLAG_OWN_CAR"                : 0.0,
    "FLAG_OWN_REALTY"             : 1.0,
    "CNT_CHILDREN"                : 0.0,
    "CNT_FAM_MEMBERS"             : 1.0,
    "REGION_RATING_CLIENT"        : 2.0,

    # ── Bureau Features ────────────────────────────────────────
    "BUREAU_LOAN_COUNT"           : float(bureau_loan_count),
    "BUREAU_ACTIVE_RATIO"         : bureau_active_ratio,
    "BUREAU_DEBT_TO_CREDIT_RATIO" : bureau_debt_ratio,
    "BUREAU_DPD_MAX"              : float(bureau_dpd_max),
    "NPA_FLAG"                    : float(1 if npa_flag else 0),
    "WILFUL_DEFAULT_FLAG"         : float(1 if wilful_default else 0),

    # ── Installment Payment Features ───────────────────────────
    "INST_LATE_RATE"              : inst_late_rate,
    "INST_DAYS_LATE_MAX"          : float(inst_days_late),

    # ── Credit Card / Utilization Features ─────────────────────
    "CC_UTILIZATION_MEAN"         : cc_utilization,
    "CC_UTILIZATION_MAX"          : round(cc_utilization * 1.2, 4),
    "CC_DPD_MEAN"                 : float(dpd_12m / 12),

    # ── POS Features ────────────────────────────────────────────
    "POS_DPD_FLAG_RATE"           : float(pos_dpd_flag),
    "POS_COMPLETION_RATIO"        : pos_completion,

    # ── Financial Ratios (for CAM) ─────────────────────────────
    "dscr"                        : dscr,
    "interest_coverage_ratio"     : interest_coverage,
    "ebitda_margin_pct"           : ebitda_margin,
    "pat_margin_pct"              : pat_margin,
    "debt_equity_ratio"           : round(total_debt / (annual_income * 0.45), 4) if annual_income > 0 else 7.4,
    "working_capital_limit"       : wc_limit,
    "wc_utilization_pct"          : utilization,

    # ── Time-Series Features ────────────────────────────────────
    "yoy_revenue_growth_pct"      : yoy_revenue_growth,
    "yoy_pat_growth_pct"          : yoy_pat_growth,
    "revenue_3yr_cagr_pct"        : revenue_3yr_cagr,
    "ebitda_trend"                : ebitda_trend,
    "revenue_volatility"          : revenue_volatility,

    # ── Bank Statement Features ─────────────────────────────────
    "avg_monthly_inflow"          : avg_monthly_inflow,
    "avg_monthly_outflow"         : avg_monthly_outflow,
    "inflow_outflow_ratio"        : inflow_outflow_ratio,
    "cash_flow_volatility"        : cash_flow_volatility,
    "emi_burden_ratio"            : emi_burden_ratio,
    "avg_closing_balance"         : avg_closing_balance,
    "unusual_txn_count"           : float(unusual_txn_count),

    # ── GST Compliance Features ─────────────────────────────────
    "gst_filing_rate"             : gst_filing_rate,
    "gst_revenue_consistency"     : gst_revenue_consistency,
    "gst_annual_turnover"         : gst_annual_turnover,

    # ── MCA Features ────────────────────────────────────────────
    "mca_is_active"               : float(1 if is_active else 0),
    "mca_risk_flag"               : float(1 if mca_risk_flag else 0),
    "insolvency_flag"             : float(1 if insolvency_flag else 0),
    "director_count"              : float(director_count),
    "years_incorporated"          : float(years_incorporated),

    # ── HE Encryption Metadata ──────────────────────────────────
    "he_fields_to_encrypt"        : "EXT_SOURCE_1,EXT_SOURCE_2,EXT_SOURCE_3,CC_UTILIZATION_MEAN",
    "he_encryption_applied"       : False,    # Notebook 3 sets this to True

    # ── Pipeline Metadata ───────────────────────────────────────
    "feature_engineering_version" : "1.0",
    "processed_at"                : datetime.now().isoformat(),
    "ready_for_layer3"            : True,
    "ready_for_layer5"            : True,
}

# Write as single-row Delta table
gold_df = spark.createDataFrame([gold_record])
gold_path = f"file:{GOLD_ROOT}/feature_table"
gold_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(gold_path)

print(f"  ✅ Gold/feature_table written — {gold_df.count()} rows, {len(gold_record)} features")
print(f"  Path: {gold_path}")


# COMMAND ----------
# MAGIC %md ## 10. Summary

# COMMAND ----------

print("\n" + "="*60)
print("  NOTEBOOK 2 COMPLETE — Silver → Gold")
print("="*60)
print(f"  Features engineered : {len(gold_record)}")
print(f"  EXT_SOURCE_MEAN     : {ext_mean:.4f}")
print(f"  CREDIT_INCOME_RATIO : {credit_income_ratio:.4f}")
print(f"  DSCR                : {dscr:.2f}x")
print(f"  ICR                 : {interest_coverage:.2f}x")
print(f"  EBITDA Trend        : {ebitda_trend}")
print(f"  HE Encryption       : Pending (Notebook 3)")
print(f"\n  → Run Notebook 3 next: Gold → borrower_profile.json + HE")
print("="*60)
