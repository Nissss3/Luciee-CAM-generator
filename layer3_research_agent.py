"""
============================================================
  CREDIT DECISIONING ENGINE — LAYER 3: AI RESEARCH AGENT
  
  Covers:
  - News Sentiment Analysis
  - MCA Filing Checks
  - Litigation / Court Record Signals
  - ESG Risk Scoring
  - Industry Outlook
  - Competitor Benchmarking

  Stack:
  - Google Gemini API (LLM reasoning + synthesis)
  - Tavily API (web search) — FREE 1000/month
  - BeautifulSoup (scraping)
  - SQLite (research memory / cache)
  
  Run: python layer3_research_agent.py
============================================================

SETUP (run once in terminal):
    pip install google-genai tavily-python beautifulsoup4 requests

GET FREE API KEYS:
    Gemini : https://aistudio.google.com/app/apikey
    Tavily : https://app.tavily.com  (1000 free searches/month)
"""

# ============================================================
# SECTION 0 — IMPORTS & CONFIGURATION
# ============================================================

import os
import json
import time
import sqlite3
import hashlib
import requests
from datetime import datetime, timedelta
from typing import Optional
from bs4 import BeautifulSoup

from google import genai
from google.genai import types as genai_types

# ── API Keys — set these as environment variables ──────────
# In terminal: set GEMINI_API_KEY=your_key  (Windows)
#              export GEMINI_API_KEY=your_key (Mac/Linux)
# OR just paste directly below for testing:

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDx-dfsg8t2kKTmNLEK01rZIpPi9XKpgxo")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")

# ── Configure Gemini ───────────────────────────────────────
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
# Model priority list — correct names for google.genai SDK (2025)
# gemini-1.5-* names DON'T work with new SDK — must use 2.0+ names
GEMINI_MODELS = [
    "gemini-2.5-flash",       # ✅ Confirmed working on this API key
    "gemini-2.0-flash",       # Fallback (reset tomorrow)
    "gemini-2.0-flash-lite",  # Last resort
]
GEMINI_MODEL = GEMINI_MODELS[0]

# ── Research Cache DB path ─────────────────────────────────
DB_PATH = "research_cache.db"

# ── Validity windows per data type (days) ─────────────────
VALIDITY_DAYS = {
    "news":        1,   # Refresh daily
    "litigation":  7,   # Refresh weekly
    "mca":         7,   # Refresh weekly
    "esg":         7,   # Refresh weekly
    "industry":   30,   # Refresh monthly
    "competitor":  7,   # Refresh weekly
}

# ── Windows SSL Fix ───────────────────────────────────────
import ssl
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Requests session with retries + SSL workaround
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def make_session():
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

REQUEST_SESSION = make_session()

print("✅ Layer 3 Research Agent — imports loaded")
print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================
# SECTION 1 — RESEARCH MEMORY (SQLite Cache)
# Agents never redo research unless cache is expired
# ============================================================

def init_cache_db():
    """
    Create SQLite DB to store all research results.
    This is the persistent research memory for the agent.
    Schema: company_id | data_type | content | fetched_at | valid_until | content_hash
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_cache (
            company_id      TEXT NOT NULL,
            data_type       TEXT NOT NULL,
            content_json    TEXT NOT NULL,
            fetched_at      TEXT NOT NULL,
            valid_until     TEXT NOT NULL,
            content_hash    TEXT NOT NULL,
            PRIMARY KEY (company_id, data_type)
        )
    """)
    # Audit log — track every research refresh
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_audit (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id      TEXT,
            data_type       TEXT,
            action          TEXT,
            reason          TEXT,
            timestamp       TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Research cache DB initialized")


def compute_hash(content: dict) -> str:
    """Hash content to detect if research results actually changed."""
    content_str = json.dumps(content, sort_keys=True)
    return hashlib.md5(content_str.encode()).hexdigest()


def should_refresh(company_id: str, data_type: str) -> tuple[bool, str]:
    """
    Check if we need to re-run research for this company + data type.
    Returns (bool, reason)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT valid_until FROM research_cache WHERE company_id=? AND data_type=?",
        (company_id, data_type)
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return True, "Never fetched before"

    valid_until = datetime.fromisoformat(row[0])
    if datetime.now() > valid_until:
        return True, f"Cache expired at {valid_until.strftime('%Y-%m-%d')}"

    return False, f"Cache valid until {valid_until.strftime('%Y-%m-%d')}"


def save_to_cache(company_id: str, data_type: str, content: dict):
    """Save research result to cache with validity window."""
    validity_days = VALIDITY_DAYS.get(data_type, 7)
    fetched_at = datetime.now().isoformat()
    valid_until = (datetime.now() + timedelta(days=validity_days)).isoformat()
    content_hash = compute_hash(content)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if content actually changed before writing
    cursor.execute(
        "SELECT content_hash FROM research_cache WHERE company_id=? AND data_type=?",
        (company_id, data_type)
    )
    existing = cursor.fetchone()

    if existing and existing[0] == content_hash:
        # Content unchanged — just extend validity window
        cursor.execute(
            """UPDATE research_cache SET valid_until=? 
               WHERE company_id=? AND data_type=?""",
            (valid_until, company_id, data_type)
        )
        action = "EXTENDED"
        reason = "Content unchanged, validity refreshed"
    else:
        # New or changed content — full upsert
        cursor.execute(
            """INSERT OR REPLACE INTO research_cache 
               (company_id, data_type, content_json, fetched_at, valid_until, content_hash)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (company_id, data_type, json.dumps(content), fetched_at, valid_until, content_hash)
        )
        action = "UPDATED" if existing else "CREATED"
        reason = "New content fetched"

    # Audit log
    cursor.execute(
        """INSERT INTO research_audit (company_id, data_type, action, reason, timestamp)
           VALUES (?, ?, ?, ?, ?)""",
        (company_id, data_type, action, reason, fetched_at)
    )
    conn.commit()
    conn.close()


def load_from_cache(company_id: str, data_type: str) -> Optional[dict]:
    """Load cached research result."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT content_json FROM research_cache WHERE company_id=? AND data_type=?",
        (company_id, data_type)
    )
    row = cursor.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


# ============================================================
# SECTION 2 — WEB SEARCH (Tavily) + SCRAPING (BeautifulSoup)
# ============================================================

def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using Tavily API.
    Falls back to a direct Google scrape if Tavily key not set.
    Returns list of {title, url, content} dicts.
    """
    if TAVILY_API_KEY and TAVILY_API_KEY != "PASTE_YOUR_TAVILY_KEY_HERE":
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=TAVILY_API_KEY)
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results
            )
            results = []
            for r in response.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")
                })
            return results
        except Exception as e:
            print(f"Tavily error: {e}. Falling back to scrape.")

    # Fallback — scrape Google search results
    return scrape_google_fallback(query, max_results)


def scrape_google_fallback(query: str, max_results: int = 5) -> list[dict]:
    """
    Fallback web search using DuckDuckGo HTML (no API key, no SSL issues).
    Google blocks scraping with SSL errors on Windows — DDG works reliably.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    # DuckDuckGo HTML search — reliable, no bot detection
    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"

    try:
        response = REQUEST_SESSION.get(url, headers=headers, timeout=15, verify=False)
        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        for result in soup.find_all("div", class_="result")[:max_results]:
            title_tag = result.find("a", class_="result__a")
            snippet_tag = result.find("a", class_="result__snippet")

            title = title_tag.text.strip() if title_tag else ""
            url_link = title_tag["href"] if title_tag and "href" in title_tag.attrs else ""
            snippet = snippet_tag.text.strip() if snippet_tag else ""

            if title:
                results.append({
                    "title": title,
                    "url": url_link,
                    "content": snippet
                })

        if results:
            print(f"DDG fallback: {len(results)} results")
        return results

    except Exception as e:
        print(f" DDG fallback error: {e}")
        # Last resort — return empty, Gemini will reason from company name alone
        return []


def fetch_page_content(url: str, max_chars: int = 3000) -> str:
    """
    Fetch and parse the text content of a webpage.
    Used to get full article text beyond search snippets.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = REQUEST_SESSION.get(url, headers=headers, timeout=15, verify=False)
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        # Return first max_chars characters
        return text[:max_chars]
    except Exception:
        return ""


# ============================================================
# SECTION 3 — GEMINI LLM HELPER
# ============================================================

def ask_gemini(prompt: str, temperature: float = 0.1) -> str:
    """
    Send a prompt to Gemini with:
    - Auto-retry on 429 rate limit (waits then retries)
    - Model fallback (tries next model if quota exhausted)
    - Works even with empty web results (pure LLM reasoning)
    """
    global GEMINI_MODEL

    for model in GEMINI_MODELS:
        GEMINI_MODEL = model
        retries = 3
        for attempt in range(retries):
            try:
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=1500,
                    )
                )
                if attempt > 0 or model != GEMINI_MODELS[0]:
                    print(f"Success with model: {model}")
                time.sleep(3)  # 3s gap between calls — prevents per-minute quota hits
                return response.text.strip()

            except Exception as e:
                err_str = str(e)

                # Rate limit — extract wait time and sleep
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    # Try to extract retry delay from error message
                    import re
                    delay_match = re.search("retryDelay.*?([0-9]+)s", err_str)
                    wait = int(delay_match.group(1)) + 2 if delay_match else 15
                    
                    # Cap wait at 20s — if longer, move to next model
                    if wait > 60:  # Only skip if wait > 60s
                        print(f" {model} quota exhausted (wait={wait}s) — trying next model")
                        break  # Break retry loop, try next model
                    
                    print(f"Rate limited — waiting {wait}s then retrying (attempt {attempt+1}/{retries})...")
                    time.sleep(wait)
                    continue  # Retry same model after wait

                # Other error — log and try next model
                print(f"Gemini error ({model}): {err_str[:120]}")
                break

    print("All Gemini models failed — returning empty (check API key or quota)")
    return ""


def parse_gemini_json(raw_response: str) -> dict:
    """
    Parse JSON from Gemini response.
    Handles: markdown fences, extra text before/after JSON,
             single-quoted JSON, trailing commas.
    """
    if not raw_response:
        return {"parse_error": True, "reason": "empty response"}
    
    try:
        clean = raw_response.strip()
        
        # Method 1: Strip markdown code fences
        if "```" in clean:
            parts = clean.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    clean = part
                    break
        
        # Method 2: Find JSON object boundaries directly
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            clean = clean[start:end+1]
        
        # Method 3: Parse
        return json.loads(clean)
    
    except json.JSONDecodeError as e:
        # Method 4: Try to fix common Gemini quirks
        try:
            import re
            # Remove trailing commas before } or ]
            fixed = re.sub(r',\s*(?=[}\]])', '', clean)  # remove trailing commas
            # Replace True/False (Python) with true/false (JSON) 
            fixed = fixed.replace(": True", ": true").replace(": False", ": false")
            fixed = fixed.replace(":True", ":true").replace(":False", ":false")
            return json.loads(fixed)
        except:
            print(f"JSON parse failed: {str(e)[:60]}")
            print(f"  Raw response preview: {raw_response[:200]}")
            return {"raw_response": raw_response[:500], "parse_error": True}


# ============================================================
# SECTION 4 — MODULE A: NEWS SENTIMENT ANALYSIS
# ============================================================

def analyze_news_sentiment(company_name: str, company_id: str) -> dict:
    """
    Search recent news about the company and score sentiment.
    Looks for: financial distress signals, management issues,
               regulatory trouble, positive growth news.
    """
    refresh_needed, reason = should_refresh(company_id, "news")
    if not refresh_needed:
        print(f"News: using cache ({reason})")
        return load_from_cache(company_id, "news")

    print(f"News: fetching fresh data ({reason})")

    # Search multiple angles
    queries = [
        f"{company_name} financial news 2024 2025",
        f"{company_name} fraud default bankruptcy lawsuit",
        f"{company_name} expansion growth revenue profit",
    ]

    all_articles = []
    for q in queries:
        results = tavily_search(q, max_results=3)
        all_articles.extend(results)
        time.sleep(0.5)  # Rate limit courtesy

    if not all_articles:
        print(" No web results — Gemini will reason from training knowledge")

    # Compile article summaries for Gemini
    articles_text = ""
    for i, art in enumerate(all_articles[:8], 1):
        articles_text += f"\nArticle {i}: {art['title']}\n{art['content'][:400]}\n"

    prompt = f"""
You are a credit risk analyst reviewing news about a loan applicant company.

Company: {company_name}
Recent news articles:
{articles_text}

Analyze these articles and respond ONLY with a JSON object in this exact format:
{{
  "overall_sentiment": "Positive" | "Neutral" | "Negative" | "Highly Negative",
  "sentiment_score": <number from -100 to +100>,
  "key_positive_signals": ["signal1", "signal2"],
  "key_negative_signals": ["signal1", "signal2"],
  "financial_distress_signals": true | false,
  "management_red_flags": true | false,
  "regulatory_issues_detected": true | false,
  "fraud_signals": true | false,
  "news_risk_score": <number from 0 to 100, higher = more risky>,
  "analyst_summary": "<2-3 sentence summary for credit memo>",
  "articles_reviewed": {len(all_articles)}
}}
Only return the JSON. No other text.
"""

    raw = ask_gemini(prompt)
    result = parse_gemini_json(raw)
    result["company_name"] = company_name
    result["data_type"] = "news"
    result["fetched_at"] = datetime.now().isoformat()

    save_to_cache(company_id, "news", result)
    return result


# ============================================================
# SECTION 5 — MODULE B: MCA FILING CHECKS
# ============================================================

def check_mca_filings(company_name: str, company_id: str, cin: str = None) -> dict:
    """
    Check Ministry of Corporate Affairs (MCA) filing status.
    Looks for: strike-off risk, director changes, charge creation,
               delayed filings, paid-up capital changes.

    CIN = Corporate Identity Number (21-digit number on certificate of incorporation)
    If CIN provided, searches directly. Otherwise searches by name.
    """
    refresh_needed, reason = should_refresh(company_id, "mca")
    if not refresh_needed:
        print(f"MCA: using cache ({reason})")
        return load_from_cache(company_id, "mca")

    print(f"MCA: fetching fresh data ({reason})")

    # Search MCA-related information
    search_query = f"{company_name} MCA company registration India director CIN"
    if cin:
        search_query = f"CIN {cin} MCA company details India"

    results = tavily_search(search_query, max_results=5)

    # Also search for specific red flags
    red_flag_results = tavily_search(
        f"{company_name} MCA struck off winding up insolvency NCLT",
        max_results=3
    )

    all_content = ""
    for r in (results + red_flag_results)[:6]:
        all_content += f"\n{r['title']}: {r['content'][:300]}\n"

    prompt = f"""
You are a corporate law analyst checking MCA (Ministry of Corporate Affairs) 
compliance for a credit applicant.

Company: {company_name}
CIN: {cin if cin else "Not provided"}

Search findings:
{all_content}

Based on available information, respond ONLY with this JSON:
{{
  "company_status": "Active" | "Struck Off" | "Under Liquidation" | "Dormant" | "Unknown",
  "cin_found": true | false,
  "registration_verified": true | false,
  "director_change_recent": true | false,
  "director_change_detail": "<detail or null>",
  "charge_registered": true | false,
  "charge_detail": "<detail or null>",
  "filing_compliance": "Regular" | "Delayed" | "Non-compliant" | "Unknown",
  "strike_off_risk": "Low" | "Medium" | "High",
  "insolvency_proceedings": true | false,
  "nclt_proceedings": true | false,
  "paid_up_capital_change": true | false,
  "mca_risk_score": <0 to 100, higher = more risk>,
  "red_flags": ["flag1", "flag2"],
  "analyst_note": "<1-2 sentence summary>"
}}
Only return the JSON. No other text.
"""

    raw = ask_gemini(prompt)
    result = parse_gemini_json(raw)
    result["company_name"] = company_name
    result["data_type"] = "mca"
    result["fetched_at"] = datetime.now().isoformat()

    save_to_cache(company_id, "mca", result)
    return result


# ============================================================
# SECTION 6 — MODULE C: LITIGATION / COURT RECORDS
# ============================================================

def check_litigation(company_name: str, company_id: str,
                     promoter_name: str = None) -> dict:
    """
    Search for litigation signals — court cases, SEBI/RBI actions,
    consumer complaints, DRT (Debt Recovery Tribunal) cases.
    Also checks promoter name if provided.
    """
    refresh_needed, reason = should_refresh(company_id, "litigation")
    if not refresh_needed:
        print(f"Litigation: using cache ({reason})")
        return load_from_cache(company_id, "litigation")

    print(f"  🔍 Litigation: fetching fresh data ({reason})")

    queries = [
        f"{company_name} court case lawsuit India",
        f"{company_name} SEBI RBI penalty regulatory action",
        f"{company_name} DRT debt recovery tribunal NPA",
    ]
    if promoter_name:
        queries.append(f"{promoter_name} court case fraud default India")

    all_content = ""
    for q in queries:
        results = tavily_search(q, max_results=3)
        for r in results:
            all_content += f"\n{r['title']}: {r['content'][:300]}\n"
        time.sleep(0.5)

    prompt = f"""
You are a legal risk analyst checking litigation exposure for a credit applicant.

Company: {company_name}
Promoter/Director: {promoter_name if promoter_name else "Not provided"}

Search findings:
{all_content}

Respond ONLY with this JSON:
{{
  "active_litigation": true | false,
  "litigation_count_estimate": <number or null>,
  "litigation_types": ["Civil", "Criminal", "Regulatory", "Consumer", "Tax"],
  "sebi_action": true | false,
  "sebi_detail": "<detail or null>",
  "rbi_action": true | false,
  "rbi_detail": "<detail or null>",
  "drt_case": true | false,
  "drt_detail": "<detail or null>",
  "criminal_case_promoter": true | false,
  "wilful_defaulter_flag": true | false,
  "consumer_complaints_high": true | false,
  "tax_dispute": true | false,
  "litigation_risk_level": "Low" | "Medium" | "High" | "Very High",
  "litigation_risk_score": <0 to 100>,
  "key_findings": ["finding1", "finding2"],
  "analyst_note": "<2-3 sentence summary for credit memo>"
}}
Only return the JSON. No other text.
"""

    raw = ask_gemini(prompt)
    result = parse_gemini_json(raw)
    result["company_name"] = company_name
    result["data_type"] = "litigation"
    result["fetched_at"] = datetime.now().isoformat()

    save_to_cache(company_id, "litigation", result)
    return result


# ============================================================
# SECTION 7 — MODULE D: ESG RISK SCORING
# ============================================================

def score_esg_risk(company_name: str, company_id: str, industry: str) -> dict:
    """
    Score Environmental, Social, and Governance risks.
    E: Pollution, environmental violations, green compliance
    S: Labor disputes, employee welfare, social controversies
    G: Related party transactions, audit qualifications,
       board independence, promoter pledging
    """
    refresh_needed, reason = should_refresh(company_id, "esg")
    if not refresh_needed:
        print(f"ESG: using cache ({reason})")
        return load_from_cache(company_id, "esg")

    print(f"ESG: fetching fresh data ({reason})")

    queries = [
        f"{company_name} environmental pollution violation India",
        f"{company_name} labor dispute employee strike controversy",
        f"{company_name} corporate governance audit qualification promoter pledge",
        f"{industry} industry ESG risk environmental compliance India",
    ]

    all_content = ""
    for q in queries:
        results = tavily_search(q, max_results=2)
        for r in results:
            all_content += f"\n{r['title']}: {r['content'][:250]}\n"
        time.sleep(0.5)

    prompt = f"""
You are an ESG risk analyst evaluating credit applicant sustainability risk.

Company: {company_name}
Industry: {industry}

Search findings:
{all_content}

Respond ONLY with this JSON:
{{
  "environmental_risk_score": <0 to 100>,
  "environmental_flags": {{
    "pollution_violations": true | false,
    "environmental_clearance_issues": true | false,
    "high_carbon_industry": true | false,
    "water_usage_risk": true | false
  }},
  "social_risk_score": <0 to 100>,
  "social_flags": {{
    "labor_disputes": true | false,
    "safety_violations": true | false,
    "community_controversies": true | false,
    "supply_chain_issues": true | false
  }},
  "governance_risk_score": <0 to 100>,
  "governance_flags": {{
    "audit_qualification": true | false,
    "promoter_pledging_high": true | false,
    "related_party_transactions": true | false,
    "board_independence_concern": true | false,
    "disclosure_gaps": true | false
  }},
  "overall_esg_score": <0 to 100, higher = more ESG risk>,
  "esg_rating": "AA" | "A" | "BBB" | "BB" | "B" | "CCC",
  "esg_pricing_impact": "<e.g. +25bps for high ESG risk>",
  "key_esg_concerns": ["concern1", "concern2"],
  "analyst_note": "<2-3 sentence ESG summary for credit memo>"
}}
Only return the JSON. No other text.
"""

    raw = ask_gemini(prompt)
    result = parse_gemini_json(raw)
    result["company_name"] = company_name
    result["industry"] = industry
    result["data_type"] = "esg"
    result["fetched_at"] = datetime.now().isoformat()

    save_to_cache(company_id, "esg", result)
    return result


# ============================================================
# SECTION 8 — MODULE E: INDUSTRY OUTLOOK
# ============================================================

def analyze_industry_outlook(industry: str, company_id: str) -> dict:
    """
    Analyze the macro outlook for the borrower's industry.
    Feeds into the Business Risk section of the CAM.
    Covers: growth trends, regulatory environment, cyclicality,
            demand drivers, key risks.
    """
    # Use industry as the cache key (shared across companies in same industry)
    cache_key = f"INDUSTRY_{industry.upper().replace(' ', '_')}"
    refresh_needed, reason = should_refresh(cache_key, "industry")
    if not refresh_needed:
        print(f"Industry: using cache ({reason})")
        return load_from_cache(cache_key, "industry")

    print(f"Industry: fetching fresh data ({reason})")

    queries = [
        f"{industry} industry outlook India 2025 growth",
        f"{industry} sector risks challenges India",
        f"{industry} industry revenue market size CAGR India",
    ]

    all_content = ""
    for q in queries:
        results = tavily_search(q, max_results=3)
        for r in results:
            all_content += f"\n{r['title']}: {r['content'][:350]}\n"
        time.sleep(0.5)

    prompt = f"""
You are a sector analyst preparing an industry overview for a credit appraisal memo.

Industry: {industry}

Search findings:
{all_content}

Respond ONLY with this JSON:
{{
  "industry_growth_outlook": "Strong" | "Moderate" | "Stable" | "Declining" | "Stressed",
  "cagr_estimate": "<e.g. 8-10% or Unknown>",
  "market_size_india": "<e.g. ₹50,000 Cr or Unknown>",
  "cyclicality": "High" | "Medium" | "Low",
  "regulatory_risk": "High" | "Medium" | "Low",
  "regulatory_detail": "<key regulatory considerations>",
  "demand_drivers": ["driver1", "driver2", "driver3"],
  "key_risks": ["risk1", "risk2", "risk3"],
  "raw_material_dependency": "High" | "Medium" | "Low",
  "export_sensitivity": "High" | "Medium" | "Low",
  "industry_credit_risk": "Low" | "Moderate" | "High" | "Very High",
  "industry_risk_score": <0 to 100>,
  "tailwinds": ["tailwind1", "tailwind2"],
  "headwinds": ["headwind1", "headwind2"],
  "analyst_summary": "<3-4 sentence industry overview for credit memo>"
}}
Only return the JSON. No other text.
"""

    raw = ask_gemini(prompt)
    result = parse_gemini_json(raw)
    result["industry"] = industry
    result["data_type"] = "industry"
    result["fetched_at"] = datetime.now().isoformat()

    save_to_cache(cache_key, "industry", result)
    return result


# ============================================================
# SECTION 9 — MODULE F: COMPETITOR BENCHMARKING
# ============================================================

def benchmark_competitors(company_name: str, company_id: str,
                           industry: str) -> dict:
    """
    Compare the borrower against key competitors.
    Identifies relative market position, competitive strength,
    and whether company is gaining or losing market share.
    """
    refresh_needed, reason = should_refresh(company_id, "competitor")
    if not refresh_needed:
        print(f"Competitor: using cache ({reason})")
        return load_from_cache(company_id, "competitor")

    print(f"Competitor: fetching fresh data ({reason})")

    queries = [
        f"{company_name} competitors India {industry}",
        f"{company_name} market share revenue comparison peers",
        f"top companies {industry} India market leaders 2024 2025",
    ]

    all_content = ""
    for q in queries:
        results = tavily_search(q, max_results=3)
        for r in results:
            all_content += f"\n{r['title']}: {r['content'][:300]}\n"
        time.sleep(0.5)

    prompt = f"""
You are a competitive intelligence analyst preparing a peer benchmarking 
section for a credit appraisal memo.

Company: {company_name}
Industry: {industry}

Search findings:
{all_content}

Respond ONLY with this JSON:
{{
  "market_position": "Leader" | "Challenger" | "Niche Player" | "Marginal" | "Unknown",
  "identified_competitors": ["competitor1", "competitor2", "competitor3"],
  "market_share_estimate": "<e.g. ~5% or Unknown>",
  "competitive_advantages": ["advantage1", "advantage2"],
  "competitive_weaknesses": ["weakness1", "weakness2"],
  "customer_concentration_risk": "High" | "Medium" | "Low",
  "pricing_power": "Strong" | "Moderate" | "Weak",
  "brand_strength": "Strong" | "Moderate" | "Weak" | "Unknown",
  "market_share_trend": "Gaining" | "Stable" | "Losing" | "Unknown",
  "peer_revenue_range": "<e.g. ₹100-500 Cr or Unknown>",
  "relative_credit_quality": "Better than peers" | "In line" | "Weaker than peers" | "Unknown",
  "competitor_risk_score": <0 to 100>,
  "analyst_note": "<2-3 sentence competitive assessment for credit memo>"
}}
Only return the JSON. No other text.
"""

    raw = ask_gemini(prompt)
    result = parse_gemini_json(raw)
    result["company_name"] = company_name
    result["industry"] = industry
    result["data_type"] = "competitor"
    result["fetched_at"] = datetime.now().isoformat()

    save_to_cache(company_id, "competitor", result)
    return result


# ============================================================
# SECTION 10 — MASTER RISK SYNTHESIZER
# Combines all 6 module scores into one Research Report
# ============================================================

def synthesize_research_report(company_name: str,
                                news: dict,
                                mca: dict,
                                litigation: dict,
                                esg: dict,
                                industry: dict,
                                competitor: dict) -> dict:
    """
    Gemini synthesizes all 6 research modules into a unified
    Research Risk Report — this feeds directly into the CAM.
    """
    print("\n Synthesizing all research into unified report...")

    # Extract key scores
    def safe_score(d, key, default=50):
        v = d.get(key, default)
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    news_score       = safe_score(news, "news_risk_score")
    mca_score        = safe_score(mca, "mca_risk_score")
    litigation_score = safe_score(litigation, "litigation_risk_score")
    esg_score        = safe_score(esg, "overall_esg_score")
    industry_score   = safe_score(industry, "industry_risk_score")
    competitor_score = safe_score(competitor, "competitor_risk_score")

    # Weighted composite score
    # Litigation and MCA weighted higher (binary deal-breakers)
    weights = {
        "news": 0.15,
        "mca": 0.20,
        "litigation": 0.25,
        "esg": 0.15,
        "industry": 0.15,
        "competitor": 0.10,
    }
    composite_score = (
        news_score       * weights["news"] +
        mca_score        * weights["mca"] +
        litigation_score * weights["litigation"] +
        esg_score        * weights["esg"] +
        industry_score   * weights["industry"] +
        competitor_score * weights["competitor"]
    )

    # Collect all red flags across modules
    all_red_flags = []
    if news.get("financial_distress_signals"):
        all_red_flags.append("Financial distress signals in news")
    if news.get("fraud_signals"):
        all_red_flags.append("Fraud signals detected in news")
    if mca.get("company_status") not in ["Active", "Unknown"]:
        all_red_flags.append(f"Company MCA status: {mca.get('company_status')}")
    if mca.get("insolvency_proceedings"):
        all_red_flags.append("Insolvency proceedings active")
    if litigation.get("wilful_defaulter_flag"):
        all_red_flags.append("Wilful defaulter flag")
    if litigation.get("criminal_case_promoter"):
        all_red_flags.append("Criminal case against promoter")
    if litigation.get("drt_case"):
        all_red_flags.append("DRT case found")
    if esg.get("governance_flags", {}).get("audit_qualification"):
        all_red_flags.append("Audit qualification on financial statements")
    if esg.get("governance_flags", {}).get("promoter_pledging_high"):
        all_red_flags.append("High promoter share pledging")

    # Determine deal recommendation from research alone
    if any("🔴" in f for f in all_red_flags):
        research_recommendation = "REJECT — Critical red flags detected"
    elif composite_score > 65:
        research_recommendation = "HIGH RISK — Proceed with enhanced due diligence"
    elif composite_score > 40:
        research_recommendation = "MODERATE RISK — Standard due diligence"
    else:
        research_recommendation = "LOW RISK — Research clear, proceed to financial analysis"

    # ESG pricing adjustment
    esg_rating = esg.get("esg_rating", "BBB")
    esg_spread_map = {
        "AA": 0, "A": 0, "BBB": 0.0025,
        "BB": 0.005, "B": 0.01, "CCC": 0.02
    }
    esg_pricing_adjustment = esg_spread_map.get(esg_rating, 0.005)

    report = {
        "company_name": company_name,
        "report_generated_at": datetime.now().isoformat(),

        # Individual module scores
        "module_scores": {
            "news_risk_score": news_score,
            "mca_risk_score": mca_score,
            "litigation_risk_score": litigation_score,
            "esg_risk_score": esg_score,
            "industry_risk_score": industry_score,
            "competitor_risk_score": competitor_score,
        },

        # Composite
        "composite_research_risk_score": round(composite_score, 1),
        "research_risk_level": (
            "Critical" if composite_score > 75 else
            "High" if composite_score > 55 else
            "Moderate" if composite_score > 35 else "Low"
        ),

        # Red flags (deal-breakers go to credit committee)
        "red_flags": all_red_flags,
        "red_flag_count": len(all_red_flags),
        "has_critical_flags": any("🔴" in f for f in all_red_flags),

        # Pricing impact from ESG
        "esg_pricing_adjustment_bps": int(esg_pricing_adjustment * 10000),

        # Final research recommendation
        "research_recommendation": research_recommendation,

        # Key narratives for CAM sections
        "cam_narratives": {
            "news_summary": news.get("analyst_summary", ""),
            "mca_summary": mca.get("analyst_note", ""),
            "litigation_summary": litigation.get("analyst_note", ""),
            "esg_summary": esg.get("analyst_note", ""),
            "industry_summary": industry.get("analyst_summary", ""),
            "competitor_summary": competitor.get("analyst_note", ""),
        },

        # Raw module outputs (for detailed CAM sections)
        "modules": {
            "news": news,
            "mca": mca,
            "litigation": litigation,
            "esg": esg,
            "industry": industry,
            "competitor": competitor,
        }
    }

    return report


def print_research_report(report: dict):
    """Pretty print the research report to terminal."""
    print("\n" + "="*65)
    print(f"  RESEARCH REPORT — {report['company_name']}")
    print("="*65)

    print(f"\n  Composite Risk Score : {report['composite_research_risk_score']}/100")
    print(f"  Risk Level           : {report['research_risk_level']}")
    print(f"  Recommendation       : {report['research_recommendation']}")
    print(f"  ESG Pricing Impact   : +{report['esg_pricing_adjustment_bps']}bps")

    print(f"\n  Module Scores:")
    for module, score in report["module_scores"].items():
        bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        print(f"    {module:<28} {bar} {score}/100")

    if report["red_flags"]:
        print(f"\nRed Flags ({report['red_flag_count']}):")
        for flag in report["red_flags"]:
            print(f"    {flag}")
    else:
        print("\nNo red flags detected")

    print("\n  CAM Section Summaries:")
    for section, text in report["cam_narratives"].items():
        if text:
            print(f"\n  [{section.upper()}]")
            print(f"    {text}")

    print("\n" + "="*65)


# ============================================================
# SECTION 11 — FULL RESEARCH PIPELINE RUNNER
# ============================================================

def run_research_pipeline(
    company_name: str,
    company_id: str,
    industry: str,
    promoter_name: str = None,
    cin: str = None
) -> dict:
    """
    Master function — runs all 6 research modules with caching.
    Call this for every new loan application.

    Args:
        company_name  : Full registered company name
        company_id    : Unique internal ID (e.g. "COMP_001")
        industry      : Industry sector (e.g. "Textile Manufacturing")
        promoter_name : Name of main promoter/director (optional)
        cin           : CIN number from MCA (optional but improves MCA check)

    Returns:
        Full research report dict — ready to feed into Layer 5 ML + Layer 7 CAM
    """
    
    print(f"LAYER 3 RESEARCH AGENT — {company_name}")

    # Initialize cache
    init_cache_db()

    # ── Run all 6 modules ─────────────────────────────────
    print(f"\n  Primary model: {GEMINI_MODELS[0]} (auto-falls back if quota hit)")
    print("\n[1/6] News Sentiment Analysis...")
    news = analyze_news_sentiment(company_name, company_id)

    print("\n[2/6] MCA Filing Check...")
    time.sleep(5)  # Respect per-minute quota
    mca = check_mca_filings(company_name, company_id, cin)

    print("\n[3/6] Litigation Check...")
    time.sleep(5)
    litigation = check_litigation(company_name, company_id, promoter_name)

    print("\n[4/6] ESG Risk Scoring...")
    time.sleep(5)
    esg = score_esg_risk(company_name, company_id, industry)

    print("\n[5/6] Industry Outlook...")
    time.sleep(5)
    industry_data = analyze_industry_outlook(industry, company_id)

    print("\n[6/6] Competitor Benchmarking...")
    time.sleep(5)
    competitor = benchmark_competitors(company_name, company_id, industry)

    # ── Synthesize into unified report ────────────────────
    report = synthesize_research_report(
        company_name, news, mca, litigation, esg, industry_data, competitor
    )

    # ── Print to terminal ─────────────────────────────────
    print_research_report(report)

    # ── Save full report to JSON ──────────────────────────
    output_file = f"research_report_{company_id}_{datetime.now().strftime('%Y%m%d')}.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {output_file}")

    return report


# ============================================================
# ENTRY POINT — Demo Run
# ============================================================

if __name__ == "__main__":

    # ── Example: Run research on a borrower company ───────
    # Change these values for your actual borrower

    report = run_research_pipeline(
        company_name  = "Tata Steel Limited",       # Company applying for loan
        company_id    = "COMP_001",                  # Your internal ID
        industry      = "Steel Manufacturing",       # Their industry
        promoter_name = "N Chandrasekaran",          # Main promoter/director
        cin           = "L28920MH1907PLC000260"      # CIN from MCA (optional)
    )

    # The report dict feeds into:
    # → Layer 5: report["composite_research_risk_score"] adjusts PD model
    # → Layer 7: report["cam_narratives"] auto-fills CAM sections
    # → Credit committee: report["red_flags"] triggers escalation

    print("\nLayer 3 Research Agent complete.")
    print(f"   Research Risk Score : {report['composite_research_risk_score']}/100")
    print(f"   Red Flags Found     : {report['red_flag_count']}")
    print(f"   Recommendation      : {report['research_recommendation']}")