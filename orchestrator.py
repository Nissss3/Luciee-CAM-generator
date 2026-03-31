"""
============================================================
  CREDIT DECISIONING ENGINE — run_pipeline.py
  Single orchestrator: L2 → L3 → L4 → L6 → L7

  NOTE: Layer 5 (ML model) runs on Google Colab separately.
        This script handles all local layers.
        After Layer 4 completes, it pauses and waits for
        you to run Layer 5 on Colab and update
        borrower_profile.json before continuing to L6/L7.

  Run:   python run_pipeline.py
  Usage: python run_pipeline.py --skip-pause   (skip Colab wait)
         python run_pipeline.py --from-layer 6 (start from L6)
         python run_pipeline.py --from-layer 3 (re-run L3 onwards)
============================================================
"""

import os
import sys
import json
import time
import glob
import subprocess
from pathlib import Path
from datetime import datetime

# ── Configuration ──────────────────────────────────────────
BASE_DIR     = Path(os.path.dirname(os.path.abspath(__file__)))
PROFILE_PATH = BASE_DIR / "borrower_profile.json"
PYTHON       = sys.executable   # use same Python that runs this script


def banner(text, char="=", width=60):
    print(f"\n{char*width}")
    print(f"  {text}")
    print(f"{char*width}")


def step(num, total, text):
    print(f"\n{'─'*60}")
    print(f"  [{num}/{total}] {text}")
    print(f"{'─'*60}")


def load_profile():
    if not PROFILE_PATH.exists():
        return {}
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def check_layer_done(layer_key):
    profile = load_profile()
    return profile.get(f"{layer_key}_completed", False)


def run_script(script_name, description):
    """Run a Python script and return True if it succeeded."""
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        print(f"  ❌ {script_name} not found at {script_path}")
        return False

    print(f"  ▶  Running {script_name}...")
    start = time.time()

    result = subprocess.run(
        [PYTHON, str(script_path)],
        cwd=str(BASE_DIR),
        capture_output=False,   # let output stream to terminal
    )

    elapsed = round(time.time() - start, 1)
    if result.returncode == 0:
        print(f"  ✅ {description} complete ({elapsed}s)")
        return True
    else:
        print(f"  ❌ {description} FAILED (exit code {result.returncode})")
        return False


def run_node_script(script_name, cam_json_path, description):
    """Run a Node.js script."""
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        print(f"  ❌ {script_name} not found")
        return False

    print(f"  ▶  Running node {script_name} {cam_json_path}...")
    start = time.time()

    result = subprocess.run(
        ["node", str(script_path), str(cam_json_path)],
        cwd=str(BASE_DIR),
        capture_output=False,
    )

    elapsed = round(time.time() - start, 1)
    if result.returncode == 0:
        print(f"  ✅ {description} complete ({elapsed}s)")
        return True
    else:
        print(f"  ❌ {description} FAILED (exit code {result.returncode})")
        return False


def find_latest_cam_json():
    """Find the most recently created cam_layer6_*.json file."""
    pattern = str(BASE_DIR / "cam_layer6_*.json")
    files   = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)



# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    # ── Parse args ────────────────────────────────────────
    args        = sys.argv[1:]
    from_layer  = 2

    for i, arg in enumerate(args):
        if arg == "--from-layer" and i + 1 < len(args):
            try:
                from_layer = int(args[i + 1])
            except:
                pass

    banner("CREDIT DECISIONING ENGINE — Full Pipeline")
    print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Base Dir   : {BASE_DIR}")
    print(f"  From Layer : {from_layer}")

    profile = load_profile()
    if profile:
        print(f"  Borrower   : {profile.get('company_name', 'Unknown')}")
        print(f"  Loan Ask   : ₹{profile.get('loan_amount', 0):,.0f}")

    pipeline_start = time.time()
    results = {}

    # ── LAYER 2 ────────────────────────────────────────────
    if from_layer <= 2:
        step(1, 6, "Layer 2 — Bronze → Silver → Gold → borrower_profile.json")
        ok = run_script("layer2.py", "Layer 2 Feature Engineering")
        results["layer2"] = ok
        if not ok:
            print("  ❌ Pipeline aborted — Layer 2 failed")
            sys.exit(1)
    else:
        print(f"\n  ⏭  Skipping Layer 2 (from_layer={from_layer})")
        results["layer2"] = True

    # ── LAYER 3 ────────────────────────────────────────────
    if from_layer <= 3:
        step(2, 6, "Layer 3 — AI Research Agent (Gemini)")
        print("  ⏱  This takes ~10 minutes on first run (Gemini rate limits)")
        print("  ⚡ Instant on repeat runs (smart cache)")
        ok = run_script("layer3_research_agent.py", "Layer 3 Research")
        results["layer3"] = ok
        if not ok:
            print("  ⚠️  Layer 3 failed — continuing with default research scores")
            results["layer3"] = False
            # Don't abort — Layer 4 has fallback defaults
    else:
        print(f"\n  ⏭  Skipping Layer 3 (from_layer={from_layer})")
        results["layer3"] = True

    # ── LAYER 4 ────────────────────────────────────────────
    if from_layer <= 4:
        step(3, 6, "Layer 4 — Risk Scoring Engine")
        ok = run_script("layer4_risk_scoring.py", "Layer 4 Risk Scoring")
        results["layer4"] = ok
        if not ok:
            print("  ❌ Pipeline aborted — Layer 4 failed")
            sys.exit(1)
    else:
        print(f"\n  ⏭  Skipping Layer 4 (from_layer={from_layer})")
        results["layer4"] = True

# ── LAYER 5 ────────────────────────────────────────────
    if from_layer <= 5:
        step(4, 6, "Layer 5 — ML Decisioning Engine (Local)")
        ok = run_script("layer5.py", "Layer 5 ML Inference")
        results["layer5"] = ok
        if not ok:
            print("  ❌ Pipeline aborted — Layer 5 failed")
            sys.exit(1)
    else:
        print(f"\n  ⏭  Skipping Layer 5 (from_layer={from_layer})")
        results["layer5"] = True

    # ── LAYER 6 ────────────────────────────────────────────
    if from_layer <= 6:
        step(5, 6, "Layer 6 — GenAI Synthesis (LangGraph + Gemini)")
        ok = run_script("layer6.py", "Layer 6 CAM Synthesis")
        results["layer6"] = ok
        if not ok:
            print("  ❌ Pipeline aborted — Layer 6 failed")
            sys.exit(1)
    else:
        print(f"\n  ⏭  Skipping Layer 6 (from_layer={from_layer})")
        results["layer6"] = True

    # ── LAYER 7 ────────────────────────────────────────────
    if from_layer <= 7:
        step(6, 6, "Layer 7 — Document Generation (Node.js)")

        cam_json_path = find_latest_cam_json()
        if not cam_json_path:
            print("  ❌ No cam_layer6_*.json found — Layer 6 may have failed")
            sys.exit(1)

        print(f"  📄 CAM JSON: {Path(cam_json_path).name}")
        ok = run_node_script("layer7.js", cam_json_path, "Layer 7 Documents")
        results["layer7"] = ok
        if not ok:
            print("  ❌ Layer 7 failed — check Node.js and docx package")
            sys.exit(1)
    else:
        print(f"\n  ⏭  Skipping Layer 7 (from_layer={from_layer})")
        results["layer7"] = True

    # ── FINAL SUMMARY ─────────────────────────────────────
    elapsed_total = round(time.time() - pipeline_start, 1)
    profile       = load_profile()

    banner("PIPELINE COMPLETE")
    print(f"  Company    : {profile.get('company_name', 'Unknown')}")
    print(f"  Decision   : {profile.get('layer5_decision', profile.get('layer4_recommendation', 'N/A'))}")
    print(f"  Risk Grade : {profile.get('layer4_risk_grade', 'N/A')}")
    print(f"  Final PD   : {profile.get('layer5_final_pd', 0)*100:.2f}%")
    print(f"  Credit Limit: ₹{profile.get('layer5_credit_limit', profile.get('layer4_dynamic_limit', {}).get('base_recommendation', 0)):,.0f}")
    print(f"\n  Layer Results:")
    for layer, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"    {status}  {layer}")
    print(f"\n  Total Time : {elapsed_total}s ({elapsed_total/60:.1f} min)")
    print(f"\n  Output Documents: {BASE_DIR / 'output_documents'}/")
    print(f"  CAM JSON        : {find_latest_cam_json() or 'Not found'}")
    print(f"  Profile         : {PROFILE_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()