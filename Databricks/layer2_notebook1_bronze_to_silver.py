# Databricks notebook source
# MAGIC %md
# MAGIC # Layer 2 — Notebook 1: Bronze → Silver
# MAGIC **Credit Decisioning Engine**
# MAGIC
# MAGIC This notebook reads raw JSON files from the Bronze layer (saved by Layer 1 Streamlit app),
# MAGIC validates and cleans them, and writes Delta tables to the Silver layer.
# MAGIC
# MAGIC **Run order:** Notebook 1 → Notebook 2 → Notebook 3
# MAGIC
# MAGIC **Input:**  `/Workspace/CreditEngine/bronze/{company_id}/`
# MAGIC **Output:** `/Workspace/CreditEngine/silver/{company_id}/` (Delta tables)

# COMMAND ----------
# MAGIC %md ## 0. Setup & Configuration

# COMMAND ----------

import json
import re
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, BooleanType, LongType, DateType
)

import os

spark = SparkSession.builder.appName("CreditEngine_Layer2_Bronze_Silver").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── Allow Spark to write Delta to /Workspace/ paths ────────────
spark.conf.set("spark.databricks.delta.preview.enabled", "true")

# ── CONFIGURE THIS FOR EACH BORROWER ──────────────────────────
# These are overridden by n8n via notebook parameters
COMPANY_ID   = dbutils.widgets.get("COMPANY_ID")   if "COMPANY_ID"   in [w.name for w in dbutils.widgets.getAll()] else "COMP_001"
COMPANY_NAME = dbutils.widgets.get("COMPANY_NAME") if "COMPANY_NAME" in [w.name for w in dbutils.widgets.getAll()] else "Tata Capital Limited"

# ── Paths — /Workspace/ is accessible as regular filesystem ────
# Use file: prefix for Delta table writes
BRONZE_ROOT  = f"/Workspace/CreditEngine/bronze/{COMPANY_ID}"
SILVER_ROOT  = f"/Workspace/CreditEngine/silver/{COMPANY_ID}"
DELTA_SILVER = f"file:{SILVER_ROOT}"   # Spark needs file: prefix for local paths

# ── Create directories if they don't exist ─────────────────────
os.makedirs(BRONZE_ROOT, exist_ok=True)
os.makedirs(SILVER_ROOT, exist_ok=True)

print(f"✅ Spark session ready")
print(f"   Company  : {COMPANY_NAME}")
print(f"   ID       : {COMPANY_ID}")
print(f"   Bronze   : {BRONZE_ROOT}")
print(f"   Silver   : {SILVER_ROOT}")

# COMMAND ----------
# MAGIC %md ## 1. Helper Functions

# COMMAND ----------

def read_bronze_json(source_type):
    """
    Read the latest Bronze JSON file for a given source type.
    Layer 1 saves files as: bronze/{company_id}/{source_type}/{source_type}_{timestamp}.json

    Uses standard Python os/glob — works with /Workspace/ paths which
    do NOT support dbutils.fs.ls or dbutils.fs.head.
    """
    import os, glob

    # /Workspace/ paths are accessible as regular filesystem paths
    path = f"{BRONZE_ROOT}/{source_type}"

    try:
        if not os.path.exists(path):
            print(f"  ⚠️  Path not found: {path}")
            return None

        # Find all JSON files in this source folder
        json_files = glob.glob(f"{path}/*.json")
        if not json_files:
            print(f"  ⚠️  No JSON files in: {path}")
            return None

        # Pick latest by file modification time
        latest = max(json_files, key=os.path.getmtime)
        print(f"  📂 Reading: {os.path.basename(latest)}")

        with open(latest, "r", encoding="utf-8") as f:
            wrapper = json.load(f)

        return wrapper.get("data", wrapper)

    except Exception as e:
        print(f"  ❌ Error reading {source_type}: {e}")
        return None


def safe_float(val, default=0.0):
    """Safely convert to float, return default on failure."""
    try:
        if val is None or val == "" or val == "None":
            return default
        return float(str(val).replace(",", "").replace("₹", "").strip())
    except:
        return default


def safe_int(val, default=0):
    try:
        return int(safe_float(val, default))
    except:
        return default


def safe_bool(val, default=False):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "yes", "1")
    return default


def write_silver_delta(df, table_name):
    """Write a Spark DataFrame to Silver Delta layer.
    Uses file: prefix required for /Workspace/ paths with Delta.
    """
    path = f"file:{SILVER_ROOT}/{table_name}"
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(path)
    count = df.count()
    print(f"  ✅ Silver/{table_name} — {count} rows → {path}")
    return path


# COMMAND ----------
# MAGIC %md ## 2. Financial Statements — Bronze → Silver

# COMMAND ----------

print("\n[1/6] Processing Financial Statements...")

raw_financials = read_bronze_json("financials")

if raw_financials:
    # raw_financials is a list of dicts (one per year)
    if isinstance(raw_financials, list):
        records = raw_financials
    else:
        records = [raw_financials]

    cleaned = []
    for r in records:
        cleaned.append({
            "company_id"          : COMPANY_ID,
            "company_name"        : str(r.get("company", COMPANY_NAME)),
            "fiscal_year"         : safe_int(r.get("year", 0)),
            "revenue"             : safe_float(r.get("revenue", r.get("net_sales", 0))),
            "cogs"                : safe_float(r.get("cogs", 0)),
            "gross_profit"        : safe_float(r.get("gross_profit", 0)),
            "operating_expenses"  : safe_float(r.get("operating_expenses", 0)),
            "ebitda"              : safe_float(r.get("ebitda", 0)),
            "depreciation"        : safe_float(r.get("depreciation", 0)),
            "ebit"                : safe_float(r.get("ebit", 0)),
            "interest_expense"    : safe_float(r.get("interest_expense", r.get("interest", 0))),
            "pbt"                 : safe_float(r.get("pbt", 0)),
            "tax"                 : safe_float(r.get("tax", 0)),
            "pat"                 : safe_float(r.get("pat", 0)),
            "source"              : str(r.get("source", "UNKNOWN")),
            "ingested_at"         : datetime.now().isoformat(),
        })

    fin_df = spark.createDataFrame(cleaned)

    # ── Validation: flag rows with zero revenue ────────────────
    fin_df = fin_df.withColumn(
        "is_valid",
        F.col("revenue") > 0
    )

    # ── Derived: gross margin, ebitda margin ───────────────────
    fin_df = fin_df \
        .withColumn("gross_margin_pct",
            F.when(F.col("revenue") > 0,
                   F.round(F.col("gross_profit") / F.col("revenue") * 100, 2)
            ).otherwise(0.0)
        ) \
        .withColumn("ebitda_margin_pct",
            F.when(F.col("revenue") > 0,
                   F.round(F.col("ebitda") / F.col("revenue") * 100, 2)
            ).otherwise(0.0)
        ) \
        .withColumn("pat_margin_pct",
            F.when(F.col("revenue") > 0,
                   F.round(F.col("pat") / F.col("revenue") * 100, 2)
            ).otherwise(0.0)
        )

    write_silver_delta(fin_df, "financials")
    fin_df.select("fiscal_year", "revenue", "ebitda", "pat",
                  "ebitda_margin_pct", "is_valid").show()
else:
    print("  ⚠️  No financial data — using empty placeholder")
    fin_df = spark.createDataFrame([], schema=StructType([
        StructField("company_id", StringType()), StructField("fiscal_year", IntegerType()),
        StructField("revenue", DoubleType()), StructField("ebitda", DoubleType()),
        StructField("pat", DoubleType()), StructField("is_valid", BooleanType()),
    ]))


# COMMAND ----------
# MAGIC %md ## 3. Bank Statements — Bronze → Silver

# COMMAND ----------

print("\n[2/6] Processing Bank Statements...")

raw_bank = read_bronze_json("bank_statements")

if raw_bank:
    records = raw_bank if isinstance(raw_bank, list) else [raw_bank]

    cleaned = []
    for r in records:
        date_str = str(r.get("date", "2024-01-01"))
        try:
            txn_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date().isoformat()
        except:
            txn_date = "2024-01-01"

        debit  = safe_float(r.get("debit", 0))
        credit = safe_float(r.get("credit", 0))
        desc   = str(r.get("description", ""))

        # Classify transaction type
        desc_lower = desc.lower()
        if "salary" in desc_lower or "payroll" in desc_lower:
            txn_category = "SALARY"
        elif "gst" in desc_lower or "tax" in desc_lower:
            txn_category = "TAX"
        elif "emi" in desc_lower or "loan" in desc_lower and debit > 0:
            txn_category = "LOAN_REPAYMENT"
        elif "disbursement" in desc_lower:
            txn_category = "LOAN_DISBURSEMENT"
        elif "receipt" in desc_lower or credit > 0:
            txn_category = "RECEIPT"
        elif "utility" in desc_lower or "rent" in desc_lower:
            txn_category = "OPERATING_EXPENSE"
        elif "interest" in desc_lower:
            txn_category = "INTEREST"
        else:
            txn_category = "OTHER"

        cleaned.append({
            "company_id"    : COMPANY_ID,
            "txn_date"      : txn_date,
            "description"   : desc,
            "debit"         : debit,
            "credit"        : credit,
            "balance"       : safe_float(r.get("balance", 0)),
            "txn_category"  : txn_category,
            "is_credit"     : credit > 0,
            "source"        : str(r.get("source", "UNKNOWN")),
        })

    bank_df = spark.createDataFrame(cleaned)

    # ── Remove duplicates ──────────────────────────────────────
    bank_df = bank_df.dropDuplicates(["txn_date", "description", "debit", "credit"])

    # ── Flag large unusual transactions (>3x median debit) ─────
    median_debit = bank_df.filter(F.col("debit") > 0).approxQuantile("debit", [0.5], 0.01)[0]
    bank_df = bank_df.withColumn(
        "is_unusual_txn",
        F.when(F.col("debit") > median_debit * 3, True).otherwise(False)
    )

    write_silver_delta(bank_df, "bank_statements")
    print(f"  Categories: {bank_df.groupBy('txn_category').count().collect()}")
else:
    print("  ⚠️  No bank statement data")
    bank_df = spark.createDataFrame([], schema=StructType([
        StructField("company_id", StringType()), StructField("txn_date", StringType()),
        StructField("debit", DoubleType()), StructField("credit", DoubleType()),
        StructField("balance", DoubleType()), StructField("txn_category", StringType()),
    ]))


# COMMAND ----------
# MAGIC %md ## 4. GST Returns — Bronze → Silver

# COMMAND ----------

print("\n[3/6] Processing GST Returns...")

raw_gst = read_bronze_json("gst")

if raw_gst:
    records = raw_gst if isinstance(raw_gst, list) else [raw_gst]

    cleaned = []
    for r in records:
        total_turnover = safe_float(r.get("total_turnover", 0))
        gst_collected  = safe_float(r.get("gst_collected", 0))
        net_gst_paid   = safe_float(r.get("net_gst_paid", 0))

        cleaned.append({
            "company_id"        : COMPANY_ID,
            "quarter"           : str(r.get("quarter", "")),
            "total_turnover"    : total_turnover,
            "taxable_supply"    : safe_float(r.get("taxable_supply", 0)),
            "gst_collected"     : gst_collected,
            "itc_claimed"       : safe_float(r.get("itc_claimed", 0)),
            "net_gst_paid"      : net_gst_paid,
            "filing_status"     : str(r.get("filing_status", "Unknown")),
            "is_filed"          : str(r.get("filing_status", "")).lower() == "filed",
            "effective_gst_rate": round(gst_collected / total_turnover * 100, 2) if total_turnover > 0 else 0.0,
            "source"            : str(r.get("source", "UNKNOWN")),
        })

    gst_df = spark.createDataFrame(cleaned)

    # ── Validate: flag unfiled quarters ───────────────────────
    gst_df = gst_df.withColumn(
        "compliance_flag",
        F.when(F.col("is_filed"), "COMPLIANT").otherwise("NON_COMPLIANT")
    )

    write_silver_delta(gst_df, "gst_returns")
    gst_df.select("quarter", "total_turnover", "net_gst_paid", "filing_status").show()
else:
    print("  ⚠️  No GST data")
    gst_df = spark.createDataFrame([], schema=StructType([
        StructField("company_id", StringType()), StructField("quarter", StringType()),
        StructField("total_turnover", DoubleType()), StructField("is_filed", BooleanType()),
    ]))


# COMMAND ----------
# MAGIC %md ## 5. CIBIL Score — Bronze → Silver (with HE flag)

# COMMAND ----------

print("\n[4/6] Processing CIBIL Score...")

raw_cibil = read_bronze_json("cibil")

if raw_cibil:
    # raw_cibil may be a list (CSV upload) or dict (JSON)
    record = raw_cibil[0] if isinstance(raw_cibil, list) else raw_cibil

    cibil_score = safe_int(record.get("cibil_score", 700))

    cleaned = [{
        "company_id"               : COMPANY_ID,
        "cibil_score"              : cibil_score,
        "score_band"               : str(record.get("score_band", "Good")),
        "total_accounts"           : safe_int(record.get("total_accounts", 0)),
        "active_accounts"          : safe_int(record.get("active_accounts", 0)),
        "overdue_accounts"         : safe_int(record.get("overdue_accounts", 0)),
        "dpd_last_12_months"       : safe_int(record.get("dpd_last_12_months", 0)),
        "enquiries_last_6_months"  : safe_int(record.get("enquiries_last_6_months", 0)),
        "total_outstanding"        : safe_float(record.get("total_outstanding", 0)),
        "on_time_payment_pct"      : safe_float(record.get("on_time_payment_pct", 95.0)),
        "npa_flag"                 : safe_bool(record.get("npa_flag", False)),
        "wilful_defaulter_flag"    : safe_bool(record.get("wilful_defaulter_flag", False)),
        "suit_filed_flag"          : safe_bool(record.get("suit_filed_flag", False)),
        # ── Normalize to 0-1 scale for ML (EXT_SOURCE equivalent) ──
        "score_normalized"         : round((cibil_score - 300) / 600, 4),
        # ── Mark for Homomorphic Encryption in Gold layer ──────────
        "he_encryption_required"   : True,
        "he_encrypted"             : False,   # will be True after Gold notebook
        "report_date"              : str(record.get("report_date", datetime.now().date())),
        "source"                   : str(record.get("source", "UNKNOWN")),
    }]

    cibil_df = spark.createDataFrame(cleaned)
    write_silver_delta(cibil_df, "cibil")
    cibil_df.select("cibil_score", "score_band", "score_normalized",
                     "dpd_last_12_months", "npa_flag", "he_encryption_required").show()
else:
    print("  ⚠️  No CIBIL data")
    cibil_df = spark.createDataFrame([], schema=StructType([
        StructField("company_id", StringType()), StructField("cibil_score", IntegerType()),
        StructField("score_normalized", DoubleType()), StructField("npa_flag", BooleanType()),
    ]))


# COMMAND ----------
# MAGIC %md ## 6. MCA Filing — Bronze → Silver

# COMMAND ----------

print("\n[5/6] Processing MCA Filing...")

raw_mca = read_bronze_json("mca")

if raw_mca:
    record = raw_mca[0] if isinstance(raw_mca, list) else raw_mca

    # Count directors from flattened CSV columns
    director_count = sum(
        1 for k in record.keys()
        if k.startswith("director_") and k.endswith("_name")
        and record[k] and str(record[k]).strip() not in ("", "None")
    )
    # Fallback: check directors list field
    if director_count == 0 and isinstance(record.get("directors"), list):
        director_count = len(record["directors"])

    status = str(record.get("company_status", "Unknown"))

    cleaned = [{
        "company_id"                 : COMPANY_ID,
        "cin"                        : str(record.get("cin", "")),
        "company_status"             : status,
        "company_type"               : str(record.get("company_type", "")),
        "roc"                        : str(record.get("roc", "")),
        "date_of_incorporation"      : str(record.get("date_of_incorporation", "")),
        "authorized_capital"         : safe_float(record.get("authorized_capital", 0)),
        "paid_up_capital"            : safe_float(record.get("paid_up_capital", 0)),
        "listing_status"             : str(record.get("listing_status", "")),
        "last_agm_date"              : str(record.get("last_agm_date", "")),
        "last_annual_return_filed"   : str(record.get("last_annual_return_filed", "")),
        "director_count"             : director_count,
        "insolvency_proceedings"     : safe_bool(record.get("insolvency_proceedings", False)),
        "strike_off_notice"          : safe_bool(record.get("strike_off_notice", False)),
        "director_disqualification"  : safe_bool(record.get("director_disqualification", False)),
        "is_active"                  : status.lower() == "active",
        "mca_risk_flag"              : safe_bool(record.get("insolvency_proceedings", False))
                                       or safe_bool(record.get("strike_off_notice", False))
                                       or safe_bool(record.get("director_disqualification", False)),
        "source"                     : str(record.get("source", "UNKNOWN")),
    }]

    mca_df = spark.createDataFrame(cleaned)
    write_silver_delta(mca_df, "mca")
    mca_df.select("cin", "company_status", "is_active", "director_count",
                  "insolvency_proceedings", "mca_risk_flag").show()
else:
    print("  ⚠️  No MCA data")
    mca_df = spark.createDataFrame([], schema=StructType([
        StructField("company_id", StringType()), StructField("company_status", StringType()),
        StructField("is_active", BooleanType()), StructField("mca_risk_flag", BooleanType()),
    ]))


# COMMAND ----------
# MAGIC %md ## 7. CMA Data — Bronze → Silver

# COMMAND ----------

print("\n[6/6] Processing CMA Data...")

raw_cma = read_bronze_json("cma")

if raw_cma:
    records = raw_cma if isinstance(raw_cma, list) else [raw_cma]

    cleaned = []
    for r in records:
        net_sales = safe_float(r.get("net_sales", r.get("revenue", 0)))
        ebitda    = safe_float(r.get("ebitda", 0))
        interest  = safe_float(r.get("interest", r.get("interest_expense", 0)))
        pat       = safe_float(r.get("pat", 0))

        cleaned.append({
            "company_id"            : COMPANY_ID,
            "period"                : str(r.get("period", "")),
            "net_sales"             : net_sales,
            "ebitda"                : ebitda,
            "interest"              : interest,
            "pat"                   : pat,
            "working_capital_limit" : safe_float(r.get("working_capital_limit", 0)),
            "utilization_pct"       : safe_float(r.get("utilization_pct", 0)),
            "ebitda_margin_pct"     : safe_float(r.get("ebitda_margin_pct",
                                        round(ebitda / net_sales * 100, 2) if net_sales > 0 else 0)),
            "pat_margin_pct"        : safe_float(r.get("pat_margin_pct",
                                        round(pat / net_sales * 100, 2) if net_sales > 0 else 0)),
            "interest_coverage"     : safe_float(r.get("interest_coverage",
                                        round(ebitda / interest, 2) if interest > 0 else 0)),
            "dscr"                  : safe_float(r.get("dscr",
                                        round((pat + interest) / interest, 2) if interest > 0 else 0)),
            "total_debt"            : safe_float(r.get("total_debt", 0)),
            "is_projected"          : "projected" in str(r.get("period", "")).lower(),
            "source"                : str(r.get("source", "UNKNOWN")),
        })

    cma_df = spark.createDataFrame(cleaned)
    write_silver_delta(cma_df, "cma")
    cma_df.select("period", "net_sales", "ebitda", "dscr",
                  "interest_coverage", "is_projected").show()
else:
    print("  ⚠️  No CMA data")
    cma_df = spark.createDataFrame([], schema=StructType([
        StructField("company_id", StringType()), StructField("period", StringType()),
        StructField("net_sales", DoubleType()), StructField("dscr", DoubleType()),
    ]))


# COMMAND ----------
# MAGIC %md ## 8. Silver Layer Summary

# COMMAND ----------

print("\n" + "="*60)
print("  NOTEBOOK 1 COMPLETE — Bronze → Silver")
print("="*60)

silver_tables = ["financials", "bank_statements", "gst_returns", "cibil", "mca", "cma"]
for table in silver_tables:
    try:
        path = f"file:{SILVER_ROOT}/{table}"
        df = spark.read.format("delta").load(path)
        print(f"  ✅ silver/{table:<18} — {df.count()} rows")
    except:
        print(f"  ⚠️  silver/{table:<18} — NOT FOUND")

print(f"\n  → Run Notebook 2 next: Silver → Gold")
print(f"  → COMPANY_ID: {COMPANY_ID}")
print("="*60)
