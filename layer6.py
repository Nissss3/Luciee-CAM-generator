"""
============================================================
  CREDIT DECISIONING ENGINE — LAYER 6: GENAI SYNTHESIS
  Multi-Agent Pipeline using LangGraph + Gemini

  Agents:
  1. Data Analyst Agent    — financial features → narrative
  2. Research Agent        — Layer 3 JSON → risk highlights
  3. Risk Rater Agent      — SHAP values → plain English explanation
  4. Recommendation Agent  — synthesizes all → Approve/Reject reasoning
  5. Document Writer Agent — formats everything into CAM-ready JSON

  Input:  Layer 5 ML output (PD, SHAP) + Layer 3 research JSON
  Output: Structured CAM JSON → feeds into Layer 7 document generator

SETUP:
    pip install langgraph google-genai
============================================================
"""

# ============================================================
# SECTION 0 — IMPORTS
# ============================================================

import os
import json
import time
import re
from datetime import datetime
from typing import TypedDict, Optional

# LangGraph
from langgraph.graph import StateGraph, END

# Direct Gemini SDK — same as Layer 3 (no LangChain wrapper)
from google import genai
from google.genai import types as genai_types

print("✅ Layer 6 imports loaded")
print(f"📅 Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================
# SECTION 1 — CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDx-dfsg8t2kKTmNLEK01rZIpPi9XKpgxo")

# ── Use whichever model test_gemini.py confirmed works ─────
GEMINI_MODEL = "gemini-2.0-flash"  # ← change if needed

# Direct Gemini client — same pattern as Layer 3
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

print(f"✅ LLM configured: {GEMINI_MODEL} (direct google.genai SDK)")


# ============================================================
# SECTION 2 — SHARED STATE (LangGraph TypedDict)
# This is the single object that flows through all 5 agents.
# Each agent reads from it and writes its output back to it.
# ============================================================

class CreditAnalysisState(TypedDict):
    """
    The shared state object that flows through the LangGraph pipeline.
    Each agent reads inputs from here and writes its section back here.
    """
    # ── Raw inputs from Layer 5 and Layer 3 ───────────────
    borrower_name:          str
    borrower_id:            str
    industry:               str
    loan_amount:            float
    loan_purpose:           str

    # From Layer 5 ML output
    pd_score:               float       # Probability of Default (0-1)
    lgd:                    float       # Loss Given Default (0-1)
    ead:                    float       # Exposure at Default (₹)
    expected_loss:          float       # PD × LGD × EAD
    credit_limit:           float       # Recommended limit
    risk_rating:            dict        # {rating, grade, label}
    ml_decision:            str         # Approve/Reject/Conditional
    pricing:                dict        # {recommended_rate, rate_band}
    shap_values:            list        # Top SHAP feature explanations
    financial_features:     dict        # Raw financial features used

    # From Layer 3 research JSON
    research_report:        dict        # Full Layer 3 output

    # ── Agent outputs (written by each agent) ─────────────
    financial_narrative:    Optional[str]    # Agent 1 output
    research_highlights:    Optional[str]    # Agent 2 output
    risk_explanation:       Optional[str]    # Agent 3 output
    recommendation_text:    Optional[str]    # Agent 4 output
    cam_json:               Optional[dict]   # Agent 5 final output

    # ── Pipeline metadata ─────────────────────────────────
    agent_log:              list        # Track which agents ran


# ============================================================
# SECTION 3 — HELPER: CALL LLM WITH RETRY
# ============================================================

def call_llm(system_prompt: str, user_prompt: str, agent_name: str) -> str:
    """
    Call Gemini directly via google.genai SDK (same as Layer 3).
    Combines system + user prompt into one message since genai
    handles system instructions via the combined prompt.
    Auto-retries on rate limits.
    """
    # Combine system + user into single prompt
    # (google.genai handles this better than separate message objects)
    full_prompt = f"""{system_prompt}

---

{user_prompt}"""

    for attempt in range(3):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1500,
                )
            )
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait_match = re.search(r"([0-9]+)s", err)
                wait = min(int(wait_match.group(1)) + 2 if wait_match else 15, 30)
                print(f"  ⏳ {agent_name}: rate limited, waiting {wait}s (attempt {attempt+1}/3)...")
                time.sleep(wait)
            else:
                print(f"  ⚠️ {agent_name} LLM error: {err[:120]}")
                return f"[{agent_name} could not generate output: {err[:80]}]"
    return f"[{agent_name} failed after 3 retries]"


def parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON from LLM response."""
    clean = raw.strip()
    if "```" in clean:
        parts = clean.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:]
            try:
                return json.loads(part.strip())
            except Exception:
                continue
    try:
        return json.loads(clean)
    except Exception:
        return {"raw": raw, "parse_error": True}


# ============================================================
# SECTION 4 — AGENT 1: DATA ANALYST AGENT
# Reads financial features → writes Financial Analysis narrative
# ============================================================

def data_analyst_agent(state: CreditAnalysisState) -> CreditAnalysisState:
    """
    Agent 1: Data Analyst
    Input:  financial_features, pd_score, loan_amount, credit_limit
    Output: financial_narrative (written prose for CAM Financial Analysis section)
    """
    print("\n  🤖 [Agent 1/5] Data Analyst Agent running...")

    features = state["financial_features"]
    borrower = state["borrower_name"]
    pd_pct = state["pd_score"] * 100
    loan_amt = state["loan_amount"]

    # Build a readable features summary for the LLM
    feature_lines = []
    for feat, val in features.items():
        if isinstance(val, float):
            feature_lines.append(f"  - {feat}: {val:.4f}")
        else:
            feature_lines.append(f"  - {feat}: {val}")
    features_text = "\n".join(feature_lines[:20])  # Top 20 features

    system = """You are a senior credit analyst at a large Indian bank.
Your role is to write the Financial Analysis section of a Credit Appraisal Memo (CAM).
Write in formal, professional banking language.
Be specific with numbers. Highlight strengths and weaknesses clearly.
Write in flowing prose paragraphs — no bullet points."""

    user = f"""Write the Financial Analysis section for the CAM for this borrower.

Borrower: {borrower}
Loan Amount Requested: ₹{loan_amt:,.0f}
ML Model PD Score: {pd_pct:.2f}% probability of default

Key financial features from the credit model:
{features_text}

Write 3-4 paragraphs covering:
1. Income and repayment capacity (ANNUITY_INCOME_RATIO, AMT_INCOME_TOTAL if available)
2. Credit exposure analysis (CREDIT_INCOME_RATIO, AMT_CREDIT)
3. External credit score interpretation (EXT_SOURCE scores)
4. Overall financial health summary with key risk flags

Keep each paragraph 3-5 sentences. Use specific numbers from the data above."""

    narrative = call_llm(system, user, "DataAnalystAgent")

    state["financial_narrative"] = narrative
    state["agent_log"].append({
        "agent": "DataAnalystAgent",
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "output_length": len(narrative)
    })

    print(f"  ✅ Agent 1 done — {len(narrative)} chars written")
    time.sleep(4)  # Rate limit spacing between agents
    return state


# ============================================================
# SECTION 5 — AGENT 2: RESEARCH AGENT
# Reads Layer 3 JSON → writes Risk Highlights narrative
# ============================================================

def research_agent(state: CreditAnalysisState) -> CreditAnalysisState:
    """
    Agent 2: Research Agent
    Input:  research_report (full Layer 3 JSON)
    Output: research_highlights (narrative summarizing all 6 research modules)
    """
    print("\n  🤖 [Agent 2/5] Research Agent running...")

    report = state["research_report"]
    borrower = state["borrower_name"]

    # Extract key data points from research report
    composite_score = report.get("composite_research_risk_score", 50)
    risk_level = report.get("research_risk_level", "Unknown")
    red_flags = report.get("red_flags", [])
    module_scores = report.get("module_scores", {})
    narratives = report.get("cam_narratives", {})
    esg_bps = report.get("esg_pricing_adjustment_bps", 0)

    # Build summary for LLM
    scores_text = "\n".join([f"  - {k}: {v}/100" for k, v in module_scores.items()])
    flags_text = "\n".join(red_flags) if red_flags else "No critical red flags detected"
    narratives_text = "\n".join([
        f"[{k.upper()}]: {v}" for k, v in narratives.items() if v
    ])

    system = """You are a senior credit research analyst writing the External Due Diligence 
section of a Credit Appraisal Memo (CAM).
Synthesize web research findings into a coherent risk narrative.
Be specific about what was found. Flag critical risks clearly.
Write in formal banking prose. No bullet points."""

    user = f"""Write the External Due Diligence & Research Findings section for the CAM.

Borrower: {borrower}
Overall Research Risk Score: {composite_score}/100 ({risk_level} Risk)
ESG Pricing Adjustment: +{esg_bps} basis points

Module Risk Scores (0=safe, 100=high risk):
{scores_text}

Red Flags Detected:
{flags_text}

Research Module Summaries:
{narratives_text}

Write 4-5 paragraphs covering:
1. News & reputation analysis findings
2. MCA/regulatory compliance status
3. Litigation exposure summary
4. ESG risk profile and governance quality
5. Industry outlook and competitive position

Be specific. If red flags exist, state them clearly with credit implications."""

    highlights = call_llm(system, user, "ResearchAgent")

    state["research_highlights"] = highlights
    state["agent_log"].append({
        "agent": "ResearchAgent",
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "output_length": len(highlights)
    })

    print(f"  ✅ Agent 2 done — {len(highlights)} chars written")
    time.sleep(4)
    return state


# ============================================================
# SECTION 6 — AGENT 3: RISK RATER AGENT
# SHAP values → plain English risk explanation
# ============================================================

def risk_rater_agent(state: CreditAnalysisState) -> CreditAnalysisState:
    """
    Agent 3: Risk Rater
    Input:  shap_values (list of {feature, direction, impact, shap_value})
            pd_score, risk_rating
    Output: risk_explanation (SHAP → human-readable risk narrative)
    """
    print("\n  🤖 [Agent 3/5] Risk Rater Agent running...")

    shap_values = state["shap_values"]
    pd_score = state["pd_score"]
    pd_pct = pd_score * 100
    rating = state["risk_rating"]
    borrower = state["borrower_name"]

    # Format SHAP values for LLM
    shap_lines = []
    for item in shap_values[:10]:
        feat = item.get("feature", "unknown")
        direction = item.get("direction", "unknown")
        impact = item.get("impact", "LOW")
        shap_val = item.get("shap_value", 0)
        arrow = "↑ increases" if direction == "risk_increase" else "↓ decreases"
        shap_lines.append(
            f"  - {feat}: {arrow} default risk [{impact} impact, SHAP={shap_val:.4f}]"
        )
    shap_text = "\n".join(shap_lines)

    system = """You are a credit risk model explainability specialist.
Your job is to translate machine learning model outputs into clear, 
human-readable explanations for credit committees and regulators.
Convert technical SHAP values into meaningful credit risk language.
Write in formal but accessible prose. Be specific with percentages."""

    user = f"""Write the Model-Based Risk Assessment section for the CAM.

Borrower: {borrower}
ML Model Probability of Default (PD): {pd_pct:.2f}%
Internal Risk Rating: {rating.get('rating', 'N/A')} 
  Grade: {rating.get('grade', 'N/A')} — {rating.get('label', 'N/A')}

Top SHAP Feature Contributions (what drove this PD score):
{shap_text}

Write 3 paragraphs:
1. Overall PD score interpretation — what {pd_pct:.1f}% default probability means 
   in practical terms, and where this borrower sits on the risk spectrum
2. Top 3 risk-increasing factors — explain each in credit analyst language
   (e.g. "The EXT_SOURCE_MEAN score of 0.31 indicates below-average creditworthiness 
   from external bureaus, contributing significantly to elevated default probability")
3. Top 2-3 risk-mitigating factors — what's working in the borrower's favor
   and how much they offset the risk factors

Be specific. Convert SHAP values to percentages where helpful.
This section justifies the risk rating to the credit committee."""

    explanation = call_llm(system, user, "RiskRaterAgent")

    state["risk_explanation"] = explanation
    state["agent_log"].append({
        "agent": "RiskRaterAgent",
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "output_length": len(explanation)
    })

    print(f"  ✅ Agent 3 done — {len(explanation)} chars written")
    time.sleep(4)
    return state


# ============================================================
# SECTION 7 — AGENT 4: RECOMMENDATION AGENT
# Synthesizes all 3 agent outputs → final Approve/Reject reasoning
# ============================================================

def recommendation_agent(state: CreditAnalysisState) -> CreditAnalysisState:
    """
    Agent 4: Recommendation Agent
    Input:  financial_narrative + research_highlights + risk_explanation
            + all raw metrics
    Output: recommendation_text (final credit committee recommendation)
    """
    print("\n  🤖 [Agent 4/5] Recommendation Agent running...")

    borrower        = state["borrower_name"]
    pd_pct          = state["pd_score"] * 100
    lgd_pct         = state["lgd"] * 100
    el              = state["expected_loss"]
    credit_limit    = state["credit_limit"]
    ml_decision     = state["ml_decision"]
    rating          = state["risk_rating"]
    pricing         = state["pricing"]
    loan_amount     = state["loan_amount"]
    red_flags       = state["research_report"].get("red_flags", [])
    research_score  = state["research_report"].get("composite_research_risk_score", 50)
    fin_narrative   = state["financial_narrative"]
    research_hi     = state["research_highlights"]
    risk_exp        = state["risk_explanation"]
    esg_bps         = state["research_report"].get("esg_pricing_adjustment_bps", 0)

    system = """You are the Chief Credit Officer of a large Indian bank.
You are writing the final Recommendation section of a Credit Appraisal Memo.
Your recommendation must be well-reasoned, specific, and defensible to regulators.
State the decision clearly. Justify it with specific data points.
Include conditions if approving conditionally. Write in formal banking language."""

    user = f"""Write the Final Credit Recommendation section for the CAM.

═══ BORROWER SUMMARY ═══
Borrower: {borrower}
Loan Requested: ₹{loan_amount:,.0f}
ML Decision: {ml_decision}
Recommended Limit: ₹{credit_limit:,.0f}

═══ RISK METRICS ═══
Probability of Default (PD): {pd_pct:.2f}%
Loss Given Default (LGD): {lgd_pct:.1f}%
Expected Loss: ₹{el:,.0f}
Risk Rating: {rating.get('rating')} ({rating.get('grade')}) — {rating.get('label')}
Recommended Rate: {pricing.get('recommended_rate')} (Range: {pricing.get('rate_band')})
ESG Pricing Adjustment: +{esg_bps}bps

═══ RESEARCH RISK ═══
Composite Research Score: {research_score}/100
Red Flags: {len(red_flags)} detected
{chr(10).join(red_flags) if red_flags else 'None'}

═══ ANALYST SUMMARIES ═══
Financial Analysis Key Points:
{fin_narrative[:500] if fin_narrative else 'N/A'}...

Research Highlights Key Points:
{research_hi[:400] if research_hi else 'N/A'}...

Risk Model Explanation:
{risk_exp[:400] if risk_exp else 'N/A'}...

Write the Final Recommendation section with these 4 parts:
1. DECISION STATEMENT — one clear sentence stating Approve/Reject/Conditional
2. JUSTIFICATION — 3-4 paragraphs with specific data points from all sections above
3. CONDITIONS (if Conditional/Approve) — list specific pre-disbursement conditions
4. COVENANTS — 3-4 financial covenants the borrower must maintain
   (e.g. "Maintain DSCR above 1.25x throughout loan tenor")

Be direct. Credit committee needs to be able to defend this decision to RBI auditors."""

    recommendation = call_llm(system, user, "RecommendationAgent")

    state["recommendation_text"] = recommendation
    state["agent_log"].append({
        "agent": "RecommendationAgent",
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "output_length": len(recommendation)
    })

    print(f"  ✅ Agent 4 done — {len(recommendation)} chars written")
    time.sleep(4)
    return state


# ============================================================
# SECTION 8 — AGENT 5: DOCUMENT WRITER AGENT
# Assembles all agent outputs into structured CAM JSON
# ============================================================

def document_writer_agent(state: CreditAnalysisState) -> CreditAnalysisState:
    """
    Agent 5: Document Writer
    Input:  all 4 agent narratives + all raw metrics
    Output: cam_json — fully structured CAM ready for Layer 7 doc generator
    """
    print("\n  🤖 [Agent 5/5] Document Writer Agent running...")

    borrower        = state["borrower_name"]
    borrower_id     = state["borrower_id"]
    industry        = state["industry"]
    rating          = state["risk_rating"]
    pricing         = state["pricing"]
    research        = state["research_report"]

    # Write the Executive Summary — the only NEW LLM call in this agent
    system = """You are a senior credit analyst writing an executive summary.
Write a concise, information-dense executive summary for a Credit Appraisal Memo.
Maximum 150 words. Every sentence must contain a specific data point."""

    user = f"""Write the Executive Summary for this CAM:

Borrower: {borrower} | Industry: {industry}
Loan: ₹{state['loan_amount']:,.0f} | Decision: {state['ml_decision']}
PD: {state['pd_score']*100:.2f}% | Rating: {rating.get('grade')} ({rating.get('label')})
Rate: {pricing.get('recommended_rate')} | Limit: ₹{state['credit_limit']:,.0f}
Research Risk: {research.get('composite_research_risk_score')}/100
Red Flags: {research.get('red_flag_count', 0)}
ESG Impact: +{research.get('esg_pricing_adjustment_bps', 0)}bps

Write exactly 3 sentences:
1. Who is the borrower, what they want, and the decision
2. Key financial metrics and risk rating justification
3. Final recommendation with rate and conditions"""

    exec_summary = call_llm(system, user, "DocumentWriterAgent")

    # ── Assemble the full structured CAM JSON ─────────────
    cam_json = {
        "meta": {
            "cam_id":           f"CAM-{borrower_id}-{datetime.now().strftime('%Y%m%d')}",
            "generated_at":     datetime.now().isoformat(),
            "generated_by":     "AI Credit Decisioning Engine v1.0",
            "layer":            "Layer 6 — GenAI Synthesis",
            "borrower_id":      borrower_id,
        },

        # ── Section 1: Executive Summary ──────────────────
        "executive_summary": exec_summary,

        # ── Section 2: Borrower Profile ───────────────────
        "borrower_profile": {
            "name":             borrower,
            "industry":         industry,
            "loan_purpose":     state["loan_purpose"],
            "loan_requested":   state["loan_amount"],
        },

        # ── Section 3: Industry Overview ──────────────────
        "industry_overview": {
            "narrative":        research.get("cam_narratives", {}).get("industry_summary", ""),
            "risk_score":       research.get("module_scores", {}).get("industry_risk_score", 50),
            "outlook":          research.get("modules", {}).get("industry", {}).get(
                                    "industry_growth_outlook", "Unknown"),
            "cyclicality":      research.get("modules", {}).get("industry", {}).get(
                                    "cyclicality", "Unknown"),
            "key_risks":        research.get("modules", {}).get("industry", {}).get(
                                    "key_risks", []),
        },

        # ── Section 4: Financial Analysis ─────────────────
        "financial_analysis": {
            "narrative":        state["financial_narrative"],
            "key_features":     state["financial_features"],
            "pd_score":         state["pd_score"],
            "pd_percentage":    f"{state['pd_score']*100:.2f}%",
        },

        # ── Section 5: External Due Diligence ─────────────
        "external_due_diligence": {
            "narrative":        state["research_highlights"],
            "composite_score":  research.get("composite_research_risk_score"),
            "risk_level":       research.get("research_risk_level"),
            "module_scores":    research.get("module_scores", {}),
            "red_flags":        research.get("red_flags", []),
            "mca_status":       research.get("modules", {}).get("mca", {}).get(
                                    "company_status", "Unknown"),
            "litigation_level": research.get("modules", {}).get("litigation", {}).get(
                                    "litigation_risk_level", "Unknown"),
        },

        # ── Section 6: Risk Assessment ────────────────────
        "risk_assessment": {
            "narrative":        state["risk_explanation"],
            "pd_score":         state["pd_score"],
            "lgd":              state["lgd"],
            "ead":              state["ead"],
            "expected_loss":    state["expected_loss"],
            "risk_rating":      state["risk_rating"],
            "shap_top_factors": state["shap_values"][:5],
        },

        # ── Section 7: ESG Risk ───────────────────────────
        "esg_risk": {
            "narrative":        research.get("cam_narratives", {}).get("esg_summary", ""),
            "overall_score":    research.get("modules", {}).get("esg", {}).get(
                                    "overall_esg_score", 50),
            "esg_rating":       research.get("modules", {}).get("esg", {}).get(
                                    "esg_rating", "BBB"),
            "pricing_impact_bps": research.get("esg_pricing_adjustment_bps", 0),
            "key_concerns":     research.get("modules", {}).get("esg", {}).get(
                                    "key_esg_concerns", []),
        },

        # ── Section 8: Competitive Position ───────────────
        "competitive_position": {
            "narrative":        research.get("cam_narratives", {}).get("competitor_summary", ""),
            "market_position":  research.get("modules", {}).get("competitor", {}).get(
                                    "market_position", "Unknown"),
            "competitors":      research.get("modules", {}).get("competitor", {}).get(
                                    "identified_competitors", []),
        },

        # ── Section 9: Final Recommendation ──────────────
        "recommendation": {
            "narrative":        state["recommendation_text"],
            "decision":         state["ml_decision"],
            "credit_limit":     state["credit_limit"],
            "risk_rating":      state["risk_rating"],
            "pricing": {
                "base_rate":            pricing.get("risk_free_rate"),
                "credit_spread":        pricing.get("credit_spread"),
                "recommended_rate":     pricing.get("recommended_rate"),
                "rate_band":            pricing.get("rate_band"),
                "esg_adjustment_bps":   research.get("esg_pricing_adjustment_bps", 0),
            },
        },

        # ── Section 10: Agent Pipeline Log ────────────────
        "pipeline_log": {
            "agents_run":       [a["agent"] for a in state["agent_log"]],
            "total_agents":     len(state["agent_log"]),
            "agent_details":    state["agent_log"],
        }
    }

    state["cam_json"] = cam_json
    state["agent_log"].append({
        "agent": "DocumentWriterAgent",
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "cam_sections": len(cam_json)
    })

    print(f"  ✅ Agent 5 done — CAM JSON assembled ({len(cam_json)} sections)")
    return state


# ============================================================
# SECTION 9 — BUILD LANGGRAPH PIPELINE
# ============================================================

def build_pipeline() -> StateGraph:
    """
    Wire all 5 agents into a LangGraph StateGraph.
    Flow: DataAnalyst → Research → RiskRater → Recommendation → DocumentWriter → END
    """
    graph = StateGraph(CreditAnalysisState)

    # Add all 5 agent nodes
    graph.add_node("data_analyst",    data_analyst_agent)
    graph.add_node("research",        research_agent)
    graph.add_node("risk_rater",      risk_rater_agent)
    graph.add_node("recommendation",  recommendation_agent)
    graph.add_node("document_writer", document_writer_agent)

    # Define the linear flow
    graph.set_entry_point("data_analyst")
    graph.add_edge("data_analyst",   "research")
    graph.add_edge("research",       "risk_rater")
    graph.add_edge("risk_rater",     "recommendation")
    graph.add_edge("recommendation", "document_writer")
    graph.add_edge("document_writer", END)

    return graph.compile()


# ============================================================
# SECTION 10 — RUNNER
# ============================================================

def run_layer6(
    ml_output: dict,
    research_report: dict,
    borrower_name: str,
    borrower_id: str,
    industry: str,
    loan_purpose: str = "Working Capital",
) -> dict:
    """
    Master function — runs the full 5-agent LangGraph pipeline.

    Args:
        ml_output:          Dict from Layer 5 make_lending_decision() output
        research_report:    Dict from Layer 3 run_research_pipeline() output
        borrower_name:      Full company name
        borrower_id:        Internal ID (e.g. "COMP_001")
        industry:           Industry sector
        loan_purpose:       Purpose of loan

    Returns:
        cam_json: Fully structured CAM dict → feed directly into Layer 7
    """
    print(f"\n{'🧠 '*20}")
    print("LAYER 6 — GENAI SYNTHESIS PIPELINE")
    print(f"{'🧠 '*20}")
    print(f"\nBorrower    : {borrower_name}")
    print(f"Agents      : 5 (DataAnalyst → Research → RiskRater → Recommendation → Writer)")
    print(f"LLM         : {GEMINI_MODEL}")
    print(f"Flow        : Linear StateGraph (LangGraph)\n")

    # ── Build initial state from Layer 5 + Layer 3 outputs ─
    initial_state: CreditAnalysisState = {
        # Borrower info
        "borrower_name":        borrower_name,
        "borrower_id":          borrower_id,
        "industry":             industry,
        "loan_amount":          ml_output.get("metrics", {}).get("EAD", 500000),
        "loan_purpose":         loan_purpose,

        # From Layer 5 ML output
        "pd_score":             ml_output.get("pd_score", 0.10),
        "lgd":                  ml_output.get("metrics", {}).get("LGD", 0.45),
        "ead":                  ml_output.get("metrics", {}).get("EAD", 500000),
        "expected_loss":        ml_output.get("metrics", {}).get("Expected_Loss", 22500),
        "credit_limit":         ml_output.get("credit_limit", 400000),
        "risk_rating":          ml_output.get("risk_rating", {"rating": 5, "grade": "BB", "label": "Acceptable"}),
        "ml_decision":          ml_output.get("decision", "CONDITIONAL APPROVE"),
        "pricing":              ml_output.get("pricing", {}),
        "shap_values":          ml_output.get("shap_explanations", []),
        "financial_features":   ml_output.get("financial_features", {}),

        # From Layer 3 research
        "research_report":      research_report,

        # Agent outputs — empty until each agent runs
        "financial_narrative":  None,
        "research_highlights":  None,
        "risk_explanation":     None,
        "recommendation_text":  None,
        "cam_json":             None,

        # Pipeline metadata
        "agent_log":            [],
    }

    # ── Run the LangGraph pipeline ─────────────────────────
    pipeline = build_pipeline()
    print("Pipeline compiled ✅ — running agents...\n")

    final_state = pipeline.invoke(initial_state)
    cam_json = final_state["cam_json"]

    # ── Save output JSON ───────────────────────────────────
    output_file = f"cam_layer6_{borrower_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cam_json, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  LAYER 6 COMPLETE")
    print(f"{'='*60}")
    print(f"  CAM ID       : {cam_json['meta']['cam_id']}")
    print(f"  Decision     : {cam_json['recommendation']['decision']}")
    print(f"  Risk Rating  : {cam_json['risk_assessment']['risk_rating'].get('grade')} — "
          f"{cam_json['risk_assessment']['risk_rating'].get('label')}")
    print(f"  Rate         : {cam_json['recommendation']['pricing']['recommended_rate']}")
    print(f"  Limit        : ₹{cam_json['recommendation']['credit_limit']:,.0f}")
    print(f"  Sections     : {len(cam_json)}")
    print(f"  Saved to     : {output_file}")
    print(f"{'='*60}")
    print(f"\n  → Feed {output_file} into Layer 7 for document generation")

    return cam_json


# ============================================================
# SECTION 11 — MOCK DATA + ENTRY POINT
# Replace mock data with real Layer 5 + Layer 3 outputs
# ============================================================

# ── MOCK LAYER 5 OUTPUT ────────────────────────────────────
# Replace this with the actual dict returned by
# make_lending_decision() from credit_ml_engine.py

MOCK_ML_OUTPUT = {
    "borrower":     "Tata Steel Limited",
    "decision":     "CONDITIONAL APPROVE",
    "pd_score":     0.087,
    "credit_limit": 4500000,
    "risk_rating":  {"rating": 4, "grade": "BBB", "label": "Satisfactory"},
    "metrics": {
        "PD": 0.087,
        "LGD": 0.42,
        "EAD": 5000000,
        "Expected_Loss": 182700,
        "Expected_Loss_Pct": 3.65
    },
    "pricing": {
        "risk_free_rate":    "6.50%",
        "credit_spread":     "3.20%",
        "recommended_rate":  "9.70%",
        "rate_band":         "9.20% - 10.20%"
    },
    "shap_explanations": [
        {"feature": "EXT_SOURCE_MEAN",        "direction": "risk_decrease", "impact": "HIGH",   "shap_value": -0.2341},
        {"feature": "CREDIT_INCOME_RATIO",    "direction": "risk_increase", "impact": "HIGH",   "shap_value":  0.1876},
        {"feature": "ANNUITY_INCOME_RATIO",   "direction": "risk_increase", "impact": "MEDIUM", "shap_value":  0.1203},
        {"feature": "BUREAU_DPD_MAX",         "direction": "risk_increase", "impact": "MEDIUM", "shap_value":  0.0934},
        {"feature": "INST_LATE_RATE",         "direction": "risk_decrease", "impact": "MEDIUM", "shap_value": -0.0721},
        {"feature": "YEARS_EMPLOYED",         "direction": "risk_decrease", "impact": "LOW",    "shap_value": -0.0512},
        {"feature": "POS_DPD_FLAG_RATE",      "direction": "risk_increase", "impact": "LOW",    "shap_value":  0.0389},
        {"feature": "CC_UTILIZATION_MEAN",    "direction": "risk_increase", "impact": "LOW",    "shap_value":  0.0276},
    ],
    "financial_features": {
        "AMT_INCOME_TOTAL":         800000,
        "AMT_CREDIT":               5000000,
        "AMT_ANNUITY":              72000,
        "CREDIT_INCOME_RATIO":      6.25,
        "ANNUITY_INCOME_RATIO":     0.09,
        "EXT_SOURCE_MEAN":          0.612,
        "EXT_SOURCE_MIN":           0.41,
        "AGE_YEARS":                42.3,
        "YEARS_EMPLOYED":           8.5,
        "BUREAU_LOAN_COUNT":        4,
        "BUREAU_DPD_MAX":           15,
        "INST_LATE_RATE":           0.04,
        "CC_UTILIZATION_MEAN":      0.38,
        "POS_DPD_FLAG_RATE":        0.06,
        "DOCUMENT_COUNT":           12,
    }
}

# ── MOCK LAYER 3 OUTPUT ────────────────────────────────────
# Replace with actual JSON loaded from Layer 3 output file
# e.g: with open("research_report_COMP_001_20260308.json") as f:
#          research_report = json.load(f)

MOCK_RESEARCH_REPORT = {
    "company_name":                     "Tata Steel Limited",
    "composite_research_risk_score":    32.5,
    "research_risk_level":              "Low",
    "research_recommendation":          "LOW RISK — Research clear, proceed",
    "red_flag_count":                   0,
    "has_critical_flags":               False,
    "red_flags":                        [],
    "esg_pricing_adjustment_bps":       25,
    "module_scores": {
        "news_risk_score":              25,
        "mca_risk_score":               10,
        "litigation_risk_score":        35,
        "esg_risk_score":               40,
        "industry_risk_score":          45,
        "competitor_risk_score":        30,
    },
    "cam_narratives": {
        "news_summary":     "Recent news for Tata Steel reflects stable operational performance with ongoing capacity expansion initiatives. No financial distress signals or fraud-related coverage detected in the review period.",
        "mca_summary":      "Company registration status confirmed Active with MCA. No insolvency proceedings or director disqualification notices on record.",
        "litigation_summary": "Routine commercial litigation in line with company scale. No DRT proceedings, wilful defaulter classification, or SEBI enforcement actions detected.",
        "esg_summary":      "Moderate ESG risk profile consistent with steel manufacturing sector. Environmental compliance maintained; governance structure sound with independent board representation.",
        "industry_summary": "Indian steel sector demonstrates moderate growth outlook driven by infrastructure spend and PLI scheme tailwinds. Key risks include raw material cost volatility and export sensitivity.",
        "competitor_summary": "Tata Steel holds market leadership position in Indian steel with strong brand equity. Competes with JSW Steel and SAIL across product segments.",
    },
    "modules": {
        "mca":          {"company_status": "Active", "mca_risk_score": 10},
        "litigation":   {"litigation_risk_level": "Low", "litigation_risk_score": 35, "wilful_defaulter_flag": False},
        "esg":          {"overall_esg_score": 40, "esg_rating": "BBB", "key_esg_concerns": ["Carbon intensity of steel production", "Water usage in manufacturing"]},
        "industry":     {"industry_growth_outlook": "Moderate", "cyclicality": "High", "key_risks": ["Raw material price volatility", "Global steel oversupply", "Export duty changes"]},
        "competitor":   {"market_position": "Leader", "identified_competitors": ["JSW Steel", "SAIL", "JSPL"]},
    }
}


if __name__ == "__main__":

    # ── OPTION A: Use mock data (default — works immediately) ──
    ml_output       = MOCK_ML_OUTPUT
    research_report = MOCK_RESEARCH_REPORT

    # ── OPTION B: Load real Layer 3 + Layer 5 outputs ─────────
    # Uncomment and update paths to use your actual outputs:
    #
    # with open("research_report_COMP_001_20260308.json") as f:
    #     research_report = json.load(f)
    #
    # ml_output = make_lending_decision(...)  # from credit_ml_engine.py

    cam = run_layer6(
        ml_output       = ml_output,
        research_report = research_report,
        borrower_name   = "Tata Steel Limited",
        borrower_id     = "COMP_001",
        industry        = "Steel Manufacturing",
        loan_purpose    = "Capex — Blast Furnace Expansion",
    )

    # cam dict is ready — pass directly to Layer 7
    print("\n✅ Layer 6 complete. CAM JSON ready for Layer 7.")