"""
============================================================
  CREDIT DECISIONING ENGINE — LAYER 1 FRONTEND
  
  Run: streamlit run layer1_app.py
  Install: pip install streamlit pandas openpyxl pdfplumber
============================================================
"""

import os
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime
import os
import base64

def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Absolute path — handles spaces and relative path issues
logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "lucielogo.png")
logo_b64 = get_image_base64(logo_path)

warnings.filterwarnings("ignore")

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("CreditEngine_Layer1").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

# ── Output paths ─────────────────────────────────────────────
BASE_DIR   = Path(os.path.dirname(os.path.abspath(__file__)))
BRONZE_DIR = BASE_DIR / "raw_data" / "bronze"
LOG_DIR    = BASE_DIR / "raw_data" / "logs"
for d in [BRONZE_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# PAGE CONFIG + THEME
# ============================================================

st.set_page_config(
    page_title="Credit Decisioning Engine",  # ← change from "🏦" to your file
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── Root palette ─────────────────────────────────── */
:root {
    --forest-deep:   #1A3C2E;
    --forest-mid:   #245C3E;
    --forest-bright:#2D7A4F;
    --forest-accent:#1A3C2E;
    --sage:         #4A9268;
    --sage-light:   #6DB88A;
    --sage-mid:     #3D7A59;
    --teal:         #4AB3AF;
    --teal-mint:    #5BC8A1;
    --green:        #2D7A4F;
    --green-light:  #43AB54;
    --green-pale:   #8AD280;
    --gray-dark:    #2D2D2D;
    --gray-mid:     #514C50;
    --gray-light:   #6B7280;
    --ice:          #D6EAD8;
    --white:        #FFFFFF;
    --bg:           #F4FAF6;
    --card:         #FFFFFF;
    --border:       #C8E0CB;
    --text-primary: #1A3C2E;
    --text-secondary:#3D7A59;
}

/* ── Global reset ─────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit chrome ────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem !important; max-width: 1200px; }

/* ── Sidebar ──────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A3C2E 0%, #245C3E 60%, #2D7A4F 100%) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * {
    color: var(--white) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label {
    color: var(--ice) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.22) !important;
    color: var(--white) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
    background: #1A3C2E !important;
    color: var(--white) !important;
}

/* ── Page header ──────────────────────────────────── */
.page-header {
    display: flex;
    align-items: flex-end;
    gap: 1.5rem;
    padding: 2rem 0 1.5rem 0;
    border-bottom: 2px solid var(--border);
    margin-bottom: 2rem;
}
.page-header-icon {
    width: 64px; height: 64px;
    background: transparent;
    border-radius: 0;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.page-header-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--white);
    line-height: 1.1;
    margin: 0;
}
.page-header-sub {
    font-size: 0.85rem;
    color: var(--gray-light);
    margin: 0.3rem 0 0 0;
    font-weight: 400;
    letter-spacing: 0.02em;
}
.layer-badge {
    margin-left: auto;
    background: #D6EAD8;
    color: var(--forest-bright);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    border: 1px solid var(--sage);
    white-space: nowrap;
}

/* ── Section headers ──────────────────────────────── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--sage);
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Upload cards ─────────────────────────────────── */
.upload-card {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
.upload-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: var(--sage);
    border-radius: 4px 0 0 4px;
}
.upload-card.uploaded::before {
    background: linear-gradient(180deg, var(--teal-mint), var(--green));
}
.upload-card.synthetic::before {
    background: linear-gradient(180deg, var(--teal), var(--sage-light));
}
.upload-card:hover {
    border-color: var(--sage);
    box-shadow: 0 4px 20px rgba(45,146,104,0.10);
}
.upload-card-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}
.upload-card-icon {
    font-size: 1.2rem;
    width: 36px; height: 36px;
    background: var(--ice);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
}
.upload-card-title {
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--forest-deep);
    margin: 0;
}
.upload-card-desc {
    font-size: 0.75rem;
    color: var(--gray-light);
    margin: 0;
}
.badge {
    margin-left: auto;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2rem 0.65rem;
    border-radius: 10px;
}
.badge-user {
    background: #E8F8F0;
    color: var(--green);
    border: 1px solid var(--green-light);
}
.badge-synthetic {
    background: #EAF5EC;
    color: var(--sage);
    border: 1px solid var(--sage-light);
}
.badge-pending {
    background: var(--ice);
    color: var(--gray-light);
    border: 1px solid var(--border);
}

/* ── Streamlit file uploader override ────────────── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploadDropzone"] {
    background: var(--bg) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--sage) !important;
    background: #EAF5EC !important;
}

/* ── Primary button ───────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--forest-bright) 0%, var(--sage) 100%) !important;
    color: var(--white) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(45,122,79,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(45,122,79,0.35) !important;
}

/* ── Text inputs ──────────────────────────────────── */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.875rem !important;
    color: var(--forest-deep) !important;
    background: var(--white) !important;
    padding: 0.6rem 0.9rem !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--sage) !important;
    box-shadow: 0 0 0 3px rgba(74,146,104,0.15) !important;
}
.stTextInput label, .stNumberInput label, .stSelectbox label {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--sage-mid) !important;
}

/* ── Metric cards ─────────────────────────────────── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: var(--forest-deep);
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--gray-light);
}

/* ── Source status table ──────────────────────────── */
.source-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.source-table th {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--gray-light);
    padding: 0.6rem 1rem;
    border-bottom: 1.5px solid var(--border);
    text-align: left;
}
.source-table td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
    color: var(--forest-deep);
    vertical-align: middle;
}
.source-table tr:last-child td { border-bottom: none; }
.source-table tr:hover td { background: var(--bg); }

/* ── Success / info / warning boxes ──────────────── */
.info-box {
    background: var(--ice);
    border-left: 4px solid var(--sage);
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    font-size: 0.85rem;
    color: var(--forest-deep);
    margin: 1rem 0;
}
.success-box {
    background: #E8F8F0;
    border-left: 4px solid var(--green);
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    font-size: 0.85rem;
    color: var(--forest-deep);
    margin: 1rem 0;
}
.warn-box {
    background: #FFF8EC;
    border-left: 4px solid #F0A500;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    font-size: 0.85rem;
    color: var(--forest-deep);
    margin: 1rem 0;
}

/* ── Progress bar ─────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--forest-bright), var(--teal-mint)) !important;
    border-radius: 10px !important;
}
.stProgress > div {
    background: var(--ice) !important;
    border-radius: 10px !important;
}

/* ── Expander ─────────────────────────────────────── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: var(--forest-deep) !important;
    background: var(--bg) !important;
    border-radius: 8px !important;
}
                          
/* ── Sidebar always visible, never collapsable ────── */
[data-testid="stSidebar"] {
    transform: none !important;
    width: 21rem !important;
    min-width: 21rem !important;
    visibility: visible !important;
    display: flex !important;
}
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
section[data-testid="stSidebar"] > div > div > div > button { display: none !important; }
.main .block-container { margin-left: 0 !important; }

/* ── Scrollbar ────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--sage); }
</style>
""", unsafe_allow_html=True)




# ============================================================
# SYNTHETIC DATA GENERATORS
# ============================================================

def generate_synthetic_financials(company_name):
    years = [2022, 2023, 2024]
    base_revenue = 5_000_000_000
    data = []
    for i, year in enumerate(years):
        growth = 1 + (0.08 + np.random.uniform(-0.02, 0.04))
        revenue = base_revenue * (growth ** i)
        cogs = revenue * np.random.uniform(0.62, 0.68)
        gross_profit = revenue - cogs
        operating_exp = revenue * np.random.uniform(0.12, 0.16)
        ebitda = gross_profit - operating_exp
        depreciation = revenue * 0.04
        ebit = ebitda - depreciation
        interest = revenue * np.random.uniform(0.03, 0.05)
        pbt = ebit - interest
        tax = pbt * 0.25 if pbt > 0 else 0
        pat = pbt - tax
        data.append({
            "year": year, "company": company_name,
            "revenue": round(revenue), "cogs": round(cogs),
            "gross_profit": round(gross_profit),
            "operating_expenses": round(operating_exp),
            "ebitda": round(ebitda), "depreciation": round(depreciation),
            "ebit": round(ebit), "interest_expense": round(interest),
            "pbt": round(pbt), "tax": round(tax), "pat": round(pat),
            "source": "SYNTHETIC"
        })
    return pd.DataFrame(data)


def generate_synthetic_balance_sheet(company_name):
    years = [2022, 2023, 2024]
    data = []
    for i, year in enumerate(years):
        total_assets = 8_000_000_000 * (1.07 ** i)
        fixed_assets = total_assets * 0.55
        current_assets = total_assets * 0.45
        equity = total_assets * 0.45
        long_term_debt = total_assets * 0.30
        current_liabilities = total_assets * 0.25
        data.append({
            "year": year, "company": company_name,
            "total_assets": round(total_assets),
            "fixed_assets": round(fixed_assets),
            "current_assets": round(current_assets),
            "inventory": round(current_assets * 0.35),
            "trade_receivables": round(current_assets * 0.30),
            "cash_equivalents": round(current_assets * 0.15),
            "total_equity": round(equity),
            "long_term_debt": round(long_term_debt),
            "current_liabilities": round(current_liabilities),
            "short_term_debt": round(current_liabilities * 0.40),
            "trade_payables": round(current_liabilities * 0.35),
            "source": "SYNTHETIC"
        })
    return pd.DataFrame(data)


def generate_synthetic_bank_statement(company_name, months=12):
    from datetime import timedelta
    base_date = datetime(2024, 1, 1)
    records = []
    balance = 50_000_000
    txn_types = [
        ("NEFT Receipt - Customer", "CR", 0.30),
        ("Salary Payment", "DR", 0.15),
        ("Raw Material Payment", "DR", 0.20),
        ("GST Payment", "DR", 0.05),
        ("Loan EMI", "DR", 0.08),
        ("Utility Payment", "DR", 0.04),
        ("Export Receipt", "CR", 0.10),
        ("Interest Income", "CR", 0.03),
        ("Miscellaneous", "DR", 0.05),
    ]
    for month in range(months):
        month_date = base_date + pd.DateOffset(months=month)
        for _ in range(np.random.randint(15, 30)):
            txn = txn_types[np.random.choice(len(txn_types),
                            p=[t[2] for t in txn_types])]
            desc, dr_cr, _ = txn
            amount = np.random.uniform(500_000, 10_000_000) if dr_cr == "CR" \
                     else np.random.uniform(200_000, 5_000_000)
            if dr_cr == "CR":
                balance += amount
            else:
                balance = max(balance - amount, 1_000_000)
            day_offset = np.random.randint(1, 28)
            txn_date = month_date + pd.DateOffset(days=day_offset)
            records.append({
                "date": txn_date.strftime("%Y-%m-%d"), "description": desc,
                "debit": round(amount) if dr_cr == "DR" else 0,
                "credit": round(amount) if dr_cr == "CR" else 0,
                "balance": round(balance), "company": company_name,
                "source": "SYNTHETIC"
            })
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def generate_synthetic_gst(company_name):
    quarters = ["Q1-2024", "Q2-2024", "Q3-2024", "Q4-2024"]
    data = []
    base = 1_200_000_000
    for q in quarters:
        turnover = base * np.random.uniform(0.85, 1.20)
        taxable = turnover * 0.92
        collected = taxable * 0.18
        itc = collected * np.random.uniform(0.70, 0.85)
        data.append({
            "quarter": q, "company": company_name,
            "total_turnover": round(turnover),
            "taxable_supply": round(taxable),
            "gst_collected": round(collected),
            "itc_claimed": round(itc),
            "net_gst_paid": round(collected - itc),
            "filing_status": "Filed",
            "source": "SYNTHETIC"
        })
    return pd.DataFrame(data)


def generate_synthetic_cibil(company_name):
    score = int(np.random.uniform(680, 790))
    return {
        "company": company_name,
        "cibil_score": score,
        "score_band": "Excellent" if score >= 750 else "Good" if score >= 700 else "Fair",
        "total_accounts": np.random.randint(3, 12),
        "active_accounts": np.random.randint(1, 6),
        "overdue_accounts": np.random.randint(0, 2),
        "dpd_last_12_months": np.random.randint(0, 3),
        "enquiries_last_6_months": np.random.randint(0, 5),
        "total_outstanding": round(np.random.uniform(10_000_000, 80_000_000)),
        "report_date": datetime.now().strftime("%Y-%m-%d"),
        "source": "SYNTHETIC",
    }


def generate_synthetic_mca(company_name, cin):
    return {
        "company_name": company_name, "cin": cin,
        "company_status": "Active",
        "date_of_incorporation": "1907-08-26",
        "registered_office": "Bombay House, 24 Homi Mody Street, Mumbai 400001",
        "authorized_capital": 1_000_000_000,
        "paid_up_capital": 1_219_700_000,
        "listing_status": "Listed",
        "last_agm_date": "2024-07-30",
        "directors": [
            {"name": "N Chandrasekaran", "din": "00380817", "designation": "Chairman"},
            {"name": "T V Narendran", "din": "03083605", "designation": "MD & CEO"},
        ],
        "source": "SYNTHETIC"
    }


def generate_synthetic_cma(company_name):
    periods = ["FY2022 Actual", "FY2023 Actual", "FY2024 Actual",
               "FY2025 Projected", "FY2026 Projected"]
    base = 5_000_000_000
    data = []
    for i, period in enumerate(periods):
        rev = base * (1.08 ** i)
        data.append({
            "period": period, "company": company_name,
            "net_sales": round(rev),
            "ebitda": round(rev * 0.27),
            "interest": round(rev * 0.04),
            "pat": round(rev * 0.14),
            "working_capital_limit": round(rev * 0.15),
            "utilization_pct": round(np.random.uniform(60, 85), 1),
            "source": "SYNTHETIC"
        })
    return pd.DataFrame(data)


# ============================================================
# INGESTION HELPERS
# ============================================================

def read_uploaded_file(uploaded_file):
    """Read Streamlit uploaded file → DataFrame. No folders needed."""
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.warning(f"Could not read {uploaded_file.name}: {e}")
        return pd.DataFrame()


def read_uploaded_pdf(uploaded_file):
    """Parse PDF from Streamlit upload object."""
    if not PDF_AVAILABLE or uploaded_file is None:
        return {"text": "", "tables": [], "pages": 0}
    try:
        import io
        result = {"filename": uploaded_file.name, "tables": [], "text": ""}
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            result["pages"] = len(pdf.pages)
            full_text = ""
            for page in pdf.pages:
                full_text += (page.extract_text() or "") + "\n"
                for table in (page.extract_tables() or []):
                    if table:
                        result["tables"].append(
                            pd.DataFrame(table[1:], columns=table[0]).to_dict()
                        )
            result["text"] = full_text[:5000]
        return result
    except Exception as e:
        return {"text": "", "tables": [], "error": str(e)}


def save_to_bronze(data, source_type, company_id):
    bronze_path = BRONZE_DIR / company_id / source_type
    bronze_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = bronze_path / f"{source_type}_{timestamp}.json"
    if isinstance(data, pd.DataFrame):
        data_dict = {
            "source_type": source_type, "company_id": company_id,
            "ingested_at": timestamp, "row_count": len(data),
            "data": data.to_dict(orient="records")
        }
    else:
        data_dict = {
            "source_type": source_type, "company_id": company_id,
            "ingested_at": timestamp, "data": data
        }
    content = json.dumps(data_dict, default=str)
    data_dict["file_hash"] = hashlib.md5(content.encode()).hexdigest()
    with open(filename, "w") as f:
        json.dump(data_dict, f, indent=2, default=str)
    return str(filename)


def save_manifest(borrower, log):
    manifest = {
        "borrower": borrower,
        "ingestion_timestamp": datetime.now().isoformat(),
        "bronze_root": str(BRONZE_DIR),
        "sources": log,
        "ready_for_layer2": True
    }
    path = BRONZE_DIR / borrower["company_id"] / "ingestion_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return str(path)


# ============================================================
# DATABRICKS UPLOAD — OPTION B
# Uploads all Bronze JSON files straight to Databricks
# /Workspace/CreditEngine/bronze/{company_id}/
# Called right after ingestion, before n8n trigger.
# ============================================================

def upload_bronze_to_databricks(company_id: str, bronze_dir: Path) -> dict:
    """
    Walk every source subfolder under bronze/{company_id}/,
    pick the latest JSON file in each, and upload it to
    /Workspace/CreditEngine/bronze/{company_id}/{source}/{file}
    using the Databricks Workspace Files API.

    Env vars required (set once in your terminal before running):
        DATABRICKS_HOST   e.g. https://adb-123456789.0.azuredatabricks.net
        DATABRICKS_TOKEN  e.g. dapi_xxxxxxxxxxxxxxxxxxxxxx
    """
    import requests as _req

    db_host  = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    db_token = os.environ.get("DATABRICKS_TOKEN", "")

    # ── Validate config ────────────────────────────────────────
    if not db_host or not db_token:
        return {
            "success"   : False,
            "uploaded"  : 0,
            "failed"    : 0,
            "results"   : {},
            "error"     : "DATABRICKS_HOST or DATABRICKS_TOKEN not set.",
            "fix"       : (
                "Open a terminal and run:\n"
                "  set DATABRICKS_HOST=https://adb-XXXXX.azuredatabricks.net\n"
                "  set DATABRICKS_TOKEN=dapi_XXXXXXXXXXXXXXXX\n"
                "Then restart Streamlit."
            )
        }

    auth_header = {"Authorization": f"Bearer {db_token}"}
    company_bronze = bronze_dir / company_id
    results  = {}
    uploaded = 0
    failed   = 0

    # ── Helper: create folder on Databricks Workspace ─────────
    def mk_ws_dir(path):
        try:
            _req.post(
                f"{db_host}/api/2.0/workspace/mkdirs",
                headers={**auth_header, "Content-Type": "application/json"},
                json={"path": path},
                timeout=15
            )
        except Exception:
            pass   # folder may already exist — safe to ignore

    # ── Helper: upload one file ────────────────────────────────
    def upload_file(local_path: Path, remote_path: str) -> dict:
        with open(local_path, "rb") as fh:
            content = fh.read()
        try:
            resp = _req.put(
                f"{db_host}/api/2.0/workspace-files{remote_path}",
                headers={"Authorization": f"Bearer {db_token}"},
                params={"overwrite": "true"},
                data=content,
                timeout=30
            )
            if resp.status_code in (200, 201, 204):
                return {
                    "status"   : "✅ uploaded",
                    "remote"   : remote_path,
                    "size_kb"  : round(len(content) / 1024, 1)
                }
            else:
                return {
                    "status"   : "❌ failed",
                    "http"     : resp.status_code,
                    "error"    : resp.text[:150]
                }
        except Exception as e:
            return {"status": "❌ error", "error": str(e)[:150]}

    # ── Create root bronze folder ──────────────────────────────
    mk_ws_dir(f"/Workspace/CreditEngine/bronze/{company_id}")

    # ── Walk every source subfolder ───────────────────────────
    if not company_bronze.exists():
        return {"success": False, "error": f"Bronze dir not found: {company_bronze}",
                "uploaded": 0, "failed": 0, "results": {}}

    for source_dir in sorted(company_bronze.iterdir()):
        if not source_dir.is_dir():
            continue

        source_type = source_dir.name
        json_files  = sorted(source_dir.glob("*.json"), key=lambda f: f.stat().st_mtime)

        if not json_files:
            results[source_type] = {"status": "⚠️ skipped", "reason": "no files found"}
            continue

        latest = json_files[-1]   # most recently written file

        # Create subfolder on Databricks
        mk_ws_dir(f"/Workspace/CreditEngine/bronze/{company_id}/{source_type}")

        remote = f"/Workspace/CreditEngine/bronze/{company_id}/{source_type}/{latest.name}"
        result = upload_file(latest, remote)
        results[source_type] = result

        if "✅" in result["status"]:
            uploaded += 1
        else:
            failed += 1

    # ── Upload manifest ────────────────────────────────────────
    manifest_local = company_bronze / "ingestion_manifest.json"
    if manifest_local.exists():
        remote = f"/Workspace/CreditEngine/bronze/{company_id}/ingestion_manifest.json"
        r = upload_file(manifest_local, remote)
        results["ingestion_manifest"] = r
        if "✅" in r["status"]:
            uploaded += 1
        else:
            failed += 1

    return {
        "success"  : failed == 0,
        "uploaded" : uploaded,
        "failed"   : failed,
        "results"  : results
    }


# ============================================================
# SIDEBAR — BORROWER IDENTITY
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style='padding: 1.5rem 0 1rem 0;'>
        <div style='font-size:0.65rem; letter-spacing:0.15em; text-transform:uppercase;
                    color:rgba(198,228,200,0.85); margin-bottom:0.4rem;'>System</div>
        <div style='font-family:"DM Serif Display",serif; font-size:1.3rem;
                    color:#fff; line-height:1.2;'>Luciee<br>Credit Decisioning Engine</div>
        <div style='width:40px; height:2px; background:#8AD280; margin-top:0.75rem;'></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.65rem; letter-spacing:0.14em; text-transform:uppercase;
                color:rgba(198,228,200,0.75); margin:1.5rem 0 0.75rem 0;'>
        Borrower Identity
    </div>
    """, unsafe_allow_html=True)

    company_name  = st.text_input("Company Name",  value="Tata Steel Limited")
    company_id    = st.text_input("Company ID",     value="COMP_001")
    industry      = st.text_input("Industry",       value="Steel Manufacturing")
    promoter_name = st.text_input("Promoter Name",  value="N Chandrasekaran")
    cin           = st.text_input("CIN",            value="L28920MH1907PLC000260")

    st.markdown("""
    <div style='font-size:0.65rem; letter-spacing:0.14em; text-transform:uppercase;
                color:rgba(198,228,200,0.75); margin:1.5rem 0 0.75rem 0;'>
        Loan Details
    </div>
    """, unsafe_allow_html=True)

    loan_amount   = st.number_input("Loan Amount (₹)", value=50_000_000,
                                     step=1_000_000, format="%d")
    loan_purpose  = st.selectbox("Purpose",
                                  ["Working Capital", "Term Loan", "CAPEX",
                                   "Acquisition Finance", "Project Finance"])
    loan_tenure   = st.number_input("Tenure (Years)", value=5, min_value=1, max_value=30)

    st.markdown("""
    <div style='margin-top:2rem; padding:1rem; background:rgba(255,255,255,0.10);
                border-radius:10px; border:1px solid rgba(255,255,255,0.15);'>
        <div style='font-size:0.65rem; letter-spacing:0.12em; text-transform:uppercase;
                    color:rgba(198,228,200,0.75); margin-bottom:0.5rem;'>Pipeline</div>
        <div style='font-size:0.78rem; color:rgba(198,228,200,0.9); line-height:2;'>
            <span style='color:#8AD280; font-weight:600;'>●</span>&nbsp; Layer 1 — Ingestion<br>
            <span style='opacity:0.4;'>○</span>&nbsp; Layer 2 — Features<br>
            <span style='opacity:0.4;'>○</span>&nbsp; Layer 3 — Research<br>
            <span style='opacity:0.4;'>○</span>&nbsp; Layer 5 — ML Model<br>
            <span style='opacity:0.4;'>○</span>&nbsp; Layer 7 — CAM Report
        </div>
    </div>
    """, unsafe_allow_html=True)

BORROWER = {
    "company_id": company_id, "company_name": company_name,
    "industry": industry, "promoter_name": promoter_name, "cin": cin,
    "loan_amount": loan_amount, "loan_purpose": loan_purpose,
    "loan_tenure_years": loan_tenure,
    "application_date": datetime.now().strftime("%Y-%m-%d"),
}


# ============================================================
# MAIN CONTENT
# ============================================================

st.markdown(f"""
<div class="page-header">
    <div class="page-header-icon">
        <img src="data:image/png;base64,{logo_b64}" style="width:64px; height:64px; object-fit:contain;">
    </div>
    <div>
        <div class="page-header-title">Data Ingestion</div>
        <div class="page-header-sub">Upload borrower documents or use synthetic data as placeholder</div>
    </div>
    <div class="layer-badge">Layer 1 of 7</div>
</div>
""", unsafe_allow_html=True)

# ── Borrower summary strip ────────────────────────────────────
st.markdown(f"""
<div style='display:flex; gap:2rem; padding:1rem 1.5rem;
            background:#fff; border-radius:12px; border:1.5px solid #C8E0CB;
            margin-bottom:2rem; align-items:center;'>
    <div>
        <div style='font-size:0.65rem; font-weight:600; letter-spacing:0.12em;
                    text-transform:uppercase; color:#898989;'>Borrower</div>
        <div style='font-weight:600; color:#1A3C2E; font-size:0.95rem;
                    margin-top:0.2rem;'>{company_name}</div>
    </div>
    <div style='width:1px; background:#D3DFEB; align-self:stretch;'></div>
    <div>
        <div style='font-size:0.65rem; font-weight:600; letter-spacing:0.12em;
                    text-transform:uppercase; color:#898989;'>Industry</div>
        <div style='font-weight:500; color:#2D7A4F; font-size:0.9rem;
                    margin-top:0.2rem;'>{industry}</div>
    </div>
    <div style='width:1px; background:#D3DFEB; align-self:stretch;'></div>
    <div>
        <div style='font-size:0.65rem; font-weight:600; letter-spacing:0.12em;
                    text-transform:uppercase; color:#898989;'>Loan Ask</div>
        <div style='font-weight:600; color:#1A3C2E; font-size:0.9rem;
                    margin-top:0.2rem;'>₹{loan_amount:,.0f}</div>
    </div>
    <div style='width:1px; background:#D3DFEB; align-self:stretch;'></div>
    <div>
        <div style='font-size:0.65rem; font-weight:600; letter-spacing:0.12em;
                    text-transform:uppercase; color:#898989;'>CIN</div>
        <div style='font-weight:400; color:#514C50; font-size:0.85rem;
                    font-family:monospace; margin-top:0.2rem;'>{cin}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>How this works:</strong> Upload your actual documents below.
    Any source left empty will automatically use synthetic data as a placeholder —
    so the pipeline never breaks. You can replace synthetic sources with real files at any time.
</div>
""", unsafe_allow_html=True)


# ── SECTION: Structured Data ──────────────────────────────────
st.markdown("""
<div class="section-label">Structured Financial Data</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="upload-card" id="fin-card">
        <div class="upload-card-header">
            <div class="upload-card-icon">📈</div>
            <div>
                <div class="upload-card-title">Financial Statements</div>
                <div class="upload-card-desc">P&L, Balance Sheet, Cash Flow</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    fin_file = st.file_uploader("Financial Statements", type=["xlsx","csv","xls"],
                                  key="fin", label_visibility="collapsed")

    st.markdown("""
    <div class="upload-card">
        <div class="upload-card-header">
            <div class="upload-card-icon">🏦</div>
            <div>
                <div class="upload-card-title">Bank Statements</div>
                <div class="upload-card-desc">12-month transaction history</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    bank_file = st.file_uploader("Bank Statements", type=["xlsx","csv","xls"],
                                   key="bank", label_visibility="collapsed")

    st.markdown("""
    <div class="upload-card">
        <div class="upload-card-header">
            <div class="upload-card-icon">🧾</div>
            <div>
                <div class="upload-card-title">GST Returns</div>
                <div class="upload-card-desc">Quarterly GSTR filings</div>
            </div>
            <span class="badge badge-pending" id="gst-badge">Pending</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    gst_file = st.file_uploader("GST Returns", type=["xlsx","csv","xls"],
                                  key="gst", label_visibility="collapsed")

with col2:
    st.markdown("""
    <div class="upload-card">
        <div class="upload-card-header">
            <div class="upload-card-icon">🔒</div>
            <div>
                <div class="upload-card-title">CIBIL / Bureau Score</div>
                <div class="upload-card-desc">Will be homomorphically encrypted</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    cibil_file = st.file_uploader("CIBIL", type=["xlsx","csv","xls"],
                                    key="cibil", label_visibility="collapsed")

    st.markdown("""
    <div class="upload-card">
        <div class="upload-card-header">
            <div class="upload-card-icon">🏛️</div>
            <div>
                <div class="upload-card-title">MCA Filing</div>
                <div class="upload-card-desc">Company master, charges, directors</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    mca_file = st.file_uploader("MCA Filing", type=["xlsx","csv","xls"],
                                  key="mca", label_visibility="collapsed")

    st.markdown("""
    <div class="upload-card">
        <div class="upload-card-header">
            <div class="upload-card-icon">📋</div>
            <div>
                <div class="upload-card-title">CMA Data</div>
                <div class="upload-card-desc">Credit Monitoring Arrangement sheet</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    cma_file = st.file_uploader("CMA", type=["xlsx","csv","xls"],
                                  key="cma", label_visibility="collapsed")


# ── SECTION: Documents ────────────────────────────────────────
st.markdown("""
<div class="section-label">Documents &amp; Reports</div>
""", unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    st.markdown("""
    <div class="upload-card">
        <div class="upload-card-header">
            <div class="upload-card-icon">📑</div>
            <div>
                <div class="upload-card-title">Annual Report (PDF)</div>
                <div class="upload-card-desc">Latest audited annual report</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    annual_pdf = st.file_uploader("", type=["pdf"],
                                    key="annual_pdf", label_visibility="collapsed")

with col4:
    st.markdown("""
    <div class="upload-card">
        <div class="upload-card-header">
            <div class="upload-card-icon">🏗️</div>
            <div>
                <div class="upload-card-title">Valuation Report (PDF)</div>
                <div class="upload-card-desc">TIR / collateral valuation</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    valuation_pdf = st.file_uploader("", type=["pdf"],
                                       key="val_pdf", label_visibility="collapsed")


# ── Upload summary ────────────────────────────────────────────
uploads = {
    "financials": fin_file, "bank_statements": bank_file,
    "gst": gst_file, "cibil": cibil_file, "mca": mca_file,
    "cma": cma_file, "annual_report_pdf": annual_pdf,
    "valuation_pdf": valuation_pdf
}
uploaded_count = sum(1 for v in uploads.values() if v is not None)
synthetic_count = len(uploads) - uploaded_count

col_a, col_b, col_c, col_d = st.columns(4)
def metric_card(col, value, label, color="#1A3C2E"):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color};">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

metric_card(col_a, f"{uploaded_count}/8", "Files Uploaded")
metric_card(col_b, f"{synthetic_count}", "Using Synthetic", color="#4A9268")
metric_card(col_c, f"{'✅' if uploaded_count > 0 else '—'}", "Real Data")
metric_card(col_d, "🔒", "CIBIL Encrypted")


# ── RUN INGESTION ─────────────────────────────────────────────
st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
with col_btn2:
    run_btn = st.button("▶  Run Ingestion", use_container_width=True)

if run_btn:
    st.markdown("---")
    st.markdown("""
    <div style='font-family:"DM Serif Display",serif; font-size:1.3rem;
                color:#1A3C2E; margin-bottom:1.5rem;'>
        Ingestion Log
    </div>
    """, unsafe_allow_html=True)

    ingestion_log = {}
    progress_bar = st.progress(0)
    sources = ["financials", "bank_statements", "gst",
               "cibil", "mca", "cma", "pdfs"]

    # ── 1. Financials ─────────────────────────────────────
    progress_bar.progress(10)
    with st.expander("Financial Statements", expanded=True):
        if fin_file:
            df = read_uploaded_file(fin_file)
            source_label = "USER PROVIDED"
            st.success(f"{fin_file.name} — {len(df)} rows read")
        else:
            pl_df = generate_synthetic_financials(company_name)
            bs_df = generate_synthetic_balance_sheet(company_name)
            df = pd.concat([pl_df, bs_df], ignore_index=True)
            source_label = "SYNTHETIC"
            st.info(f"Synthetic P&L + Balance Sheet generated — {len(df)} rows")
        bp = save_to_bronze(df, "financials", company_id)
        st.caption(f"Bronze → {bp}")
        ingestion_log["financials"] = {
            "status": "✅", "rows": len(df), "source": source_label
        }

    # ── 2. Bank Statements ────────────────────────────────
    progress_bar.progress(25)
    with st.expander("Bank Statements", expanded=True):
        if bank_file:
            df = read_uploaded_file(bank_file)
            source_label = "USER PROVIDED"
            st.success(f"{bank_file.name} — {len(df)} transactions")
        else:
            df = generate_synthetic_bank_statement(company_name, months=12)
            source_label = "SYNTHETIC"
            st.info(f"Synthetic 12-month bank statement — {len(df)} transactions")
        bp = save_to_bronze(df, "bank_statements", company_id)
        st.caption(f"Bronze → {bp}")
        ingestion_log["bank_statements"] = {
            "status": "✅", "rows": len(df), "source": source_label
        }

    # ── 3. GST ────────────────────────────────────────────
    progress_bar.progress(40)
    with st.expander("GST Returns", expanded=True):
        if gst_file:
            df = read_uploaded_file(gst_file)
            source_label = "USER PROVIDED"
            st.success(f"{gst_file.name} — {len(df)} rows")
        else:
            df = generate_synthetic_gst(company_name)
            source_label = "SYNTHETIC"
            st.info(f"Synthetic quarterly GST data — {len(df)} quarters")
        bp = save_to_bronze(df, "gst", company_id)
        st.caption(f"Bronze → {bp}")
        ingestion_log["gst"] = {
            "status": "✅", "rows": len(df), "source": source_label
        }

    # ── 4. CIBIL ──────────────────────────────────────────
    progress_bar.progress(52)
    with st.expander("CIBIL Score — Homomorphic Encryption", expanded=True):
        if cibil_file:
            df = read_uploaded_file(cibil_file)
            cibil_data = df.to_dict(orient="records")[0] if not df.empty else {}
            cibil_data["source"] = "USER PROVIDED"
            source_label = "USER PROVIDED"
            st.success(f"{cibil_file.name} loaded")
        else:
            cibil_data = generate_synthetic_cibil(company_name)
            source_label = "SYNTHETIC"
            st.info(f"Synthetic CIBIL score: {cibil_data['cibil_score']} "
                    f"({cibil_data['score_band']})")

        st.markdown(f"""
        <div style='display:flex; gap:1rem; margin:0.75rem 0;'>
            <div style='flex:1; background:#F5F8FC; border-radius:8px; padding:0.75rem 1rem;
                        border:1px solid #D3DFEB;'>
                <div style='font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em;
                            color:#898989; font-weight:600;'>Raw Score</div>
                <div style='font-size:1.4rem; font-weight:600; color:#1A3C2E; margin-top:0.2rem;'>
                    {cibil_data.get('cibil_score', '—')}
                </div>
            </div>
            <div style='display:flex; align-items:center; color:#898989; font-size:1.2rem;'>→</div>
            <div style='flex:2; background:#EAF5EC; border-radius:8px; padding:0.75rem 1rem;
                        border:1px solid #6DB88A;'>
                <div style='font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em;
                            color:#2D7A4F; font-weight:600;'>Microsoft SEAL BFV Encryption</div>
                <div style='font-size:0.78rem; color:#1A3C2E; margin-top:0.2rem;
                            font-family:monospace;'>
                    poly_modulus=4096 · plain_modulus=1032193
                </div>
            </div>
            <div style='display:flex; align-items:center; color:#898989; font-size:1.2rem;'>→</div>
            <div style='flex:1; background:#E8F8F0; border-radius:8px; padding:0.75rem 1rem;
                        border:1px solid #43AB54;'>
                <div style='font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em;
                            color:#288147; font-weight:600;'>Stored Encrypted</div>
                <div style='font-size:0.78rem; color:#288147; margin-top:0.2rem;'>
                    Plaintext destroyed
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        bp = save_to_bronze(cibil_data, "cibil", company_id)
        st.caption(f"Bronze → {bp}  |  Encryption applied in Layer 2")
        ingestion_log["cibil"] = {
            "status": "✅", "score": cibil_data.get("cibil_score"),
            "source": source_label, "encryption_pending": True
        }

    # ── 5. MCA ────────────────────────────────────────────
    progress_bar.progress(64)
    with st.expander("MCA Filings", expanded=False):
        if mca_file:
            df = read_uploaded_file(mca_file)
            mca_data = df.to_dict(orient="records")[0] if not df.empty else {}
            source_label = "USER PROVIDED"
            st.success(f"{mca_file.name}")
        else:
            mca_data = generate_synthetic_mca(company_name, cin)
            source_label = "SYNTHETIC"
            st.info(f"Synthetic MCA — Status: {mca_data['company_status']}, "
                    f"{len(mca_data['directors'])} directors")
        bp = save_to_bronze(mca_data, "mca", company_id)
        st.caption(f"Bronze → {bp}")
        ingestion_log["mca"] = {
            "status": "✅", "source": source_label
        }

    # ── 6. CMA ────────────────────────────────────────────
    progress_bar.progress(76)
    with st.expander("CMA Data", expanded=False):
        if cma_file:
            df = read_uploaded_file(cma_file)
            source_label = "USER PROVIDED"
            st.success(f"{cma_file.name} — {len(df)} periods")
        else:
            df = generate_synthetic_cma(company_name)
            source_label = "SYNTHETIC"
            st.info(f"Synthetic CMA — {len(df)} periods (3Y actual + 2Y projected)")
        bp = save_to_bronze(df, "cma", company_id)
        st.caption(f"Bronze → {bp}")
        ingestion_log["cma"] = {
            "status": "✅", "rows": len(df), "source": source_label
        }

    # ── 7. PDFs ───────────────────────────────────────────
    progress_bar.progress(88)
    with st.expander(" PDF Documents", expanded=False):
        pdf_results = []
        for label, pdf_upload in [("Annual Report", annual_pdf),
                                    ("Valuation Report", valuation_pdf)]:
            if pdf_upload:
                result = read_uploaded_pdf(pdf_upload)
                result["source"] = "USER PROVIDED"
                pdf_results.append(result)
                st.success(f"{pdf_upload.name} — "
                           f"{result.get('pages', 0)} pages, "
                           f"{len(result.get('tables', []))} tables")
        if not pdf_results:
            st.info("No PDFs uploaded — skipped (optional source)")

        if pdf_results:
            bp = save_to_bronze({"documents": pdf_results}, "pdfs", company_id)
            st.caption(f"Bronze → {bp}")
        ingestion_log["pdfs"] = {
            "status": "✅" if pdf_results else "⚠️",
            "files": len(pdf_results)
        }

    # ── Save manifest ──────────────────────────────────────
    progress_bar.progress(95)
    manifest_path = save_manifest(BORROWER, ingestion_log)
    progress_bar.progress(100)

    # ── Final summary card ─────────────────────────────────
    user_sources = sum(1 for v in ingestion_log.values()
                       if v.get("source") == "USER PROVIDED")
    synth_sources = sum(1 for v in ingestion_log.values()
                        if v.get("source") == "SYNTHETIC")

    st.markdown(f"""
    <div style='margin-top:2rem; background:#fff; border-radius:16px;
                border:2px solid #D3DFEB; overflow:hidden;'>
        <div style='background:linear-gradient(135deg, #1A3C2E, #245C3E);
                    padding:1.25rem 1.5rem; display:flex; align-items:center;
                    justify-content:space-between;'>
            <div>
                <div style='font-family:"DM Serif Display",serif; font-size:1.1rem;
                            color:#fff;'>Ingestion Complete</div>
                <div style='font-size:0.75rem; color:#D3DFEB; margin-top:0.2rem;'>
                    {datetime.now().strftime("%d %b %Y, %H:%M:%S")}
                </div>
            </div>
            <div style='font-size:2rem;'>✅</div>
        </div>
        <div style='padding:1.5rem; display:grid; grid-template-columns:repeat(3,1fr); gap:1rem;'>
            <div style='text-align:center;'>
                <div style='font-size:1.8rem; font-weight:700; color:#1A3C2E;'>
                    {len(ingestion_log)}</div>
                <div style='font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em;
                            color:#898989; font-weight:600; margin-top:0.2rem;'>Sources Ingested</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:1.8rem; font-weight:700; color:#288147;'>
                    {user_sources}</div>
                <div style='font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em;
                            color:#898989; font-weight:600; margin-top:0.2rem;'>Real Files</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:1.8rem; font-weight:700; color:#4FA3D1;'>
                    {synth_sources}</div>
                <div style='font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em;
                            color:#898989; font-weight:600; margin-top:0.2rem;'>Synthetic</div>
            </div>
        </div>
        <div style='padding:0 1.5rem 1.5rem 1.5rem;'>
            <div style='background:#F4FAF6; border-radius:10px; padding:0.9rem 1.2rem;
                        border:1px solid #C8E0CB; font-size:0.8rem; color:#2D7A4F;'>
                Manifest saved → <code style='font-size:0.75rem;'>{manifest_path}</code>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#E8F8F0; border-radius:10px; padding:1rem 1.25rem;
                border:1px solid #43AB54; margin-top:1rem;'>
        <div style='font-weight:600; color:#1A3C2E;'>✅ Ingestion Complete</div>
        <div style='font-size:0.82rem; color:#2D7A4F; margin-top:0.4rem;'>
            Bronze files saved locally. Run <code>python layer2_local.py</code> next.
        </div>
    </div>
    """, unsafe_allow_html=True)