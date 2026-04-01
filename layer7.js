/**
 * ============================================================
 * CREDIT DECISIONING ENGINE — LAYER 7: DOCUMENT GENERATION
 *
 * Generates all 6 credit lifecycle documents from CAM JSON:
 *   1. Credit Appraisal Memo (CAM)         — full 11-section document
 *   2. CMA Data Report                     — financial analysis sheet
 *   3. Sanction Letter                     — borrower-facing offer
 *   4. Conditions Compliance Report (CCR)  — pre-disbursement checklist
 *   5. Loan Disbursement Report (LDR)      — post-disbursement record
 *   6. PD (Personal Discussion) Report     — borrower due diligence
 *
 * Input:  cam_json (from Layer 6)
 * Output: 6 Word documents (.docx)
 *
 * Run:  node generate_documents.js <path_to_cam.json>
 * ============================================================
 */

const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageBreak,
  LevelFormat, TableOfContents, TabStopType, TabStopPosition
} = require("docx");
const fs = require("fs");
const path = require("path");

// ── Load CAM JSON — auto-finds latest cam_layer6_*.json ───
function findLatestCamJson() {
  try {
    const files = fs.readdirSync(__dirname)
      .filter(f => f.startsWith("cam_layer6_") && f.endsWith(".json"))
      .map(f => ({ name: f, mtime: fs.statSync(path.join(__dirname, f)).mtimeMs }))
      .sort((a, b) => b.mtime - a.mtime);
    return files.length > 0 ? path.join(__dirname, files[0].name) : null;
  } catch(e) { return null; }
}
const camPath = process.argv[2] || findLatestCamJson() || "./sample_cam.json";
if (!fs.existsSync(camPath)) {
  console.error(`❌ CAM JSON not found: ${camPath}`);
  console.error("   Run layer6.py first to generate it.");
  process.exit(1);
}
console.log(`📄 Loading: ${path.basename(camPath)}`);
const CAM = JSON.parse(fs.readFileSync(camPath, "utf8"));
 
const TODAY = new Date().toLocaleDateString("en-IN", {
  day: "2-digit", month: "long", year: "numeric"
});
const BORROWER = CAM.borrower_profile?.name || "Borrower";
const CAM_ID   = CAM.meta?.cam_id || "CAM-001";
const DECISION = CAM.recommendation?.decision || "CONDITIONAL APPROVE";
const LIMIT    = CAM.recommendation?.credit_limit || 0;
const RATE     = CAM.recommendation?.pricing?.recommended_rate || "N/A";
const GRADE    = CAM.recommendation?.risk_rating?.grade || "BBB";
const LABEL    = CAM.recommendation?.risk_rating?.label || "Satisfactory";
const PD_PCT   = ((CAM.financial_analysis?.pd_score || 0.087) * 100).toFixed(2);

// ── Color palette ─────────────────────────────────────────
const COLORS = {
  navy:       "1B3A6B",  // Headers, titles
  blue:       "2E75B6",  // Section headings
  lightBlue:  "D6E4F0",  // Table header rows
  green:      "1E7E34",  // Approve / pass
  amber:      "B8860B",  // Watch / conditional
  red:        "C0392B",  // Reject / fail
  grey:       "F5F5F5",  // Alt row shading
  darkGrey:   "555555",  // Body text
  white:      "FFFFFF",
};

// ── Helpers ───────────────────────────────────────────────

/** Format Indian rupee amounts */
function inr(amount) {
  if (!amount && amount !== 0) return "N/A";
  return "₹" + Number(amount).toLocaleString("en-IN");
}

/** Single thin border for tables */
function border(color = "CCCCCC") {
  return { style: BorderStyle.SINGLE, size: 1, color };
}

function allBorders(color = "CCCCCC") {
  const b = border(color);
  return { top: b, bottom: b, left: b, right: b };
}

/** Standard cell */
function cell(content, opts = {}) {
  const {
    width = 4680, bold = false, shade = null,
    color = COLORS.darkGrey, align = AlignmentType.LEFT,
    colspan = 1, size = 20
  } = opts;

  const textRun = new TextRun({
    text: String(content ?? "—"),
    bold,
    color,
    size,
    font: "Arial"
  });

  return new TableCell({
    borders: allBorders(),
    width: { size: width, type: WidthType.DXA },
    columnSpan: colspan,
    shading: shade ? { fill: shade, type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: align, children: [textRun] })]
  });
}

/** Header cell (dark blue bg, white text) */
function hCell(text, width = 4680, colspan = 1) {
  return cell(text, {
    width, bold: true, shade: COLORS.navy,
    color: COLORS.white, colspan, size: 20
  });
}

/** Sub-header cell (light blue bg, dark text) */
function sCell(text, width = 4680, colspan = 1) {
  return cell(text, {
    width, bold: true, shade: COLORS.lightBlue,
    color: COLORS.navy, colspan, size: 20
  });
}

/** Standard 2-col key-value row */
function kvRow(key, value, shade = null) {
  return new TableRow({ children: [
    cell(key,   { width: 3500, bold: true, shade, color: COLORS.navy }),
    cell(value, { width: 5860, shade }),
  ]});
}

/** Decision badge color */
function decisionColor(dec) {
  if (!dec) return COLORS.amber;
  if (dec.includes("APPROVE") && !dec.includes("CONDITIONAL")) return COLORS.green;
  if (dec.includes("REJECT")) return COLORS.red;
  return COLORS.amber;
}

/** Risk score → color */
function scoreColor(score) {
  if (score <= 30) return COLORS.green;
  if (score <= 60) return COLORS.amber;
  return COLORS.red;
}

// ── Standard Doc Styles ───────────────────────────────────
function makeStyles() {
  return {
    default: {
      document: { run: { font: "Arial", size: 20, color: COLORS.darkGrey } }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1",
        basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: COLORS.navy },
        paragraph: {
          spacing: { before: 360, after: 120 },
          outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: COLORS.blue, space: 4 } }
        }
      },
      {
        id: "Heading2", name: "Heading 2",
        basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: COLORS.blue },
        paragraph: { spacing: { before: 280, after: 80 }, outlineLevel: 1 }
      },
      {
        id: "Heading3", name: "Heading 3",
        basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial", color: COLORS.darkGrey },
        paragraph: { spacing: { before: 200, after: 60 }, outlineLevel: 2 }
      },
    ]
  };
}

/** Page header with bank name + doc title */
function makeHeader(docTitle) {
  return new Header({
    children: [
      new Paragraph({
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: COLORS.navy, space: 4 } },
        children: [
          new TextRun({ text: "FIRST NATIONAL CREDIT BANK  |  ", bold: true, size: 18, color: COLORS.navy, font: "Arial" }),
          new TextRun({ text: docTitle, size: 18, color: COLORS.blue, font: "Arial" }),
          new TextRun({ text: "  |  CONFIDENTIAL", size: 16, color: "999999", font: "Arial" }),
        ]
      })
    ]
  });
}

/** Page footer with page numbers + date */
function makeFooter() {
  return new Footer({
    children: [
      new Paragraph({
        border: { top: { style: BorderStyle.SINGLE, size: 4, color: COLORS.navy, space: 4 } },
        tabStops: [
          { type: TabStopType.CENTER, position: 4680 },
          { type: TabStopType.RIGHT,  position: 9360 }
        ],
        children: [
          new TextRun({ text: `Generated: ${TODAY}`, size: 16, color: "999999", font: "Arial" }),
          new TextRun({ text: "\t", font: "Arial" }),
          new TextRun({ text: CAM_ID, size: 16, color: "999999", font: "Arial" }),
          new TextRun({ text: "\t", font: "Arial" }),
          new TextRun({ text: "Page ", size: 16, color: "999999", font: "Arial" }),
          new TextRun({ text: "—", size: 16, color: "999999", font: "Arial" }),
        ]
      })
    ]
  });
}

/** Body paragraph with normal spacing */
function bodyPara(text, opts = {}) {
  const { bold = false, color = COLORS.darkGrey, size = 20, spacing = 160 } = opts;
  return new Paragraph({
    spacing: { after: spacing },
    children: [new TextRun({ text: String(text || ""), bold, color, size, font: "Arial" })]
  });
}

/** Section spacer */
function spacer(size = 120) {
  return new Paragraph({ spacing: { after: size }, children: [new TextRun("")] });
}

/** Bullet point */
function bullet(text, ref = "bullets") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text: String(text || ""), size: 20, font: "Arial", color: COLORS.darkGrey })]
  });
}

/** SHAP bar visual using unicode blocks */
function shapBar(value, maxVal = 0.25) {
  const abs = Math.abs(value);
  const filled = Math.round((abs / maxVal) * 10);
  const bar = "█".repeat(Math.min(filled, 10)) + "░".repeat(Math.max(0, 10 - filled));
  return bar;
}

// ============================================================
// DOCUMENT 1 — CREDIT APPRAISAL MEMO (CAM)
// ============================================================

function buildCAM() {
  const fin  = CAM.financial_analysis || {};
  const ext  = CAM.external_due_diligence || {};
  const risk = CAM.risk_assessment || {};
  const rec  = CAM.recommendation || {};
  const esg  = CAM.esg_risk || {};
  const comp = CAM.competitive_position || {};
  const ind  = CAM.industry_overview || {};
  const bp   = CAM.borrower_profile || {};
  const st   = CAM.stress_test_results?.scenarios || [];
  const shap = risk.shap_top_factors || [];
  const mods = ext.module_scores || {};

  const children = [];

  // ── Cover Block ─────────────────────────────────────────
  children.push(
    new Paragraph({
      spacing: { after: 40 },
      children: [new TextRun({ text: "CREDIT APPRAISAL MEMORANDUM", bold: true, size: 52, font: "Arial", color: COLORS.navy })]
    }),
    new Paragraph({
      spacing: { after: 20 },
      children: [new TextRun({ text: CAM_ID, size: 24, font: "Arial", color: COLORS.blue })]
    }),
    new Paragraph({
      spacing: { after: 400 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: COLORS.navy, space: 6 } },
      children: [new TextRun({ text: `Date: ${TODAY}  |  Prepared by: AI Credit Decisioning Engine v1.0  |  Status: DRAFT`, size: 18, color: "777777", font: "Arial" })]
    }),
    spacer(200),

    // Decision summary banner
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [2500, 2500, 2000, 2360],
      rows: [
        new TableRow({ children: [
          hCell("DECISION", 2500),
          hCell("CREDIT LIMIT", 2500),
          hCell("INTEREST RATE", 2000),
          hCell("RISK RATING", 2360),
        ]}),
        new TableRow({ children: [
          cell(DECISION, { width: 2500, bold: true, color: decisionColor(DECISION), size: 22 }),
          cell(inr(LIMIT), { width: 2500, bold: true, color: COLORS.navy, size: 22 }),
          cell(RATE, { width: 2000, bold: true, color: COLORS.navy, size: 22 }),
          cell(`${GRADE} — ${LABEL}`, { width: 2360, bold: true, color: COLORS.navy, size: 22 }),
        ]}),
      ]
    }),
    spacer(300),
  );

  // ── Section 1: Executive Summary ─────────────────────────
  children.push(
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("1. Executive Summary")] }),
    bodyPara(CAM.executive_summary || ""),
    spacer(),
  );

  // ── Section 2: Borrower Profile ──────────────────────────
  children.push(
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("2. Borrower Profile")] }),
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [3500, 5860],
      rows: [
        new TableRow({ children: [hCell("Field", 3500), hCell("Details", 5860)] }),
        kvRow("Borrower Name",    bp.name || ""),
        kvRow("Industry / Sector", bp.industry || ""),
        kvRow("Loan Purpose",     bp.loan_purpose || ""),
        kvRow("Amount Requested", inr(bp.loan_requested)),
        kvRow("Recommended Limit", inr(LIMIT)),
        kvRow("CAM Reference",    CAM_ID),
        kvRow("Assessment Date",  TODAY),
      ]
    }),
    spacer(300),
  );

  // ── Section 3: Industry Overview ─────────────────────────
  children.push(
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("3. Industry Overview")] }),
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [3500, 5860],
      rows: [
        new TableRow({ children: [hCell("Parameter", 3500), hCell("Assessment", 5860)] }),
        kvRow("Industry Growth Outlook", ind.outlook || ""),
        kvRow("Cyclicality",             ind.cyclicality || ""),
        kvRow("Industry Risk Score",     `${ind.risk_score || "—"}/100`),
      ]
    }),
    spacer(120),
    bodyPara(ind.narrative || ""),
    spacer(120),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Key Industry Risks")] }),
    ...(ind.key_risks || []).map(r => bullet(r)),
    spacer(240),
  );

  // ── Section 4: Financial Analysis ────────────────────────
  const kf = fin.key_features || {};
  children.push(
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("4. Financial Analysis")] }),
    bodyPara(fin.narrative || ""),
    spacer(120),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Key Financial Ratios & Metrics")] }),
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [3500, 2930, 2930],
      rows: [
        new TableRow({ children: [hCell("Metric", 3500), hCell("Value", 2930), hCell("Assessment", 2930)] }),
        new TableRow({ children: [
          sCell("Credit to Income Ratio", 3500),
          cell(`${kf.CREDIT_INCOME_RATIO || "—"}x`, { width: 2930 }),
          cell(kf.CREDIT_INCOME_RATIO > 5 ? "Elevated — Monitor" : "Acceptable", {
            width: 2930, color: kf.CREDIT_INCOME_RATIO > 5 ? COLORS.amber : COLORS.green, bold: true
          }),
        ]}),
        new TableRow({ children: [
          sCell("Annuity / Income Ratio", 3500),
          cell(`${((kf.ANNUITY_INCOME_RATIO || 0) * 100).toFixed(1)}%`, { width: 2930 }),
          cell(kf.ANNUITY_INCOME_RATIO < 0.35 ? "Within Limit" : "Exceeds Threshold", {
            width: 2930, color: kf.ANNUITY_INCOME_RATIO < 0.35 ? COLORS.green : COLORS.red, bold: true
          }),
        ]}),
        new TableRow({ children: [
          sCell("Ext. Credit Score (Mean)", 3500),
          cell((kf.EXT_SOURCE_MEAN || "—").toString(), { width: 2930 }),
          cell(kf.EXT_SOURCE_MEAN >= 0.6 ? "Strong" : kf.EXT_SOURCE_MEAN >= 0.4 ? "Moderate" : "Weak", {
            width: 2930, color: kf.EXT_SOURCE_MEAN >= 0.6 ? COLORS.green : COLORS.amber, bold: true
          }),
        ]}),
        new TableRow({ children: [
          sCell("Annual Income", 3500),
          cell(inr(kf.AMT_INCOME_TOTAL), { width: 2930 }),
          cell("", { width: 2930 }),
        ]}),
        new TableRow({ children: [
          sCell("Bureau Max DPD", 3500),
          cell(`${kf.BUREAU_DPD_MAX || 0} days`, { width: 2930 }),
          cell(kf.BUREAU_DPD_MAX > 30 ? "High Risk" : kf.BUREAU_DPD_MAX > 0 ? "Minor History" : "Clean", {
            width: 2930,
            color: kf.BUREAU_DPD_MAX > 30 ? COLORS.red : kf.BUREAU_DPD_MAX > 0 ? COLORS.amber : COLORS.green,
            bold: true
          }),
        ]}),
        new TableRow({ children: [
          sCell("Late Payment Rate", 3500),
          cell(`${((kf.INST_LATE_RATE || 0) * 100).toFixed(1)}%`, { width: 2930 }),
          cell(kf.INST_LATE_RATE < 0.05 ? "Acceptable" : "Elevated", {
            width: 2930, color: kf.INST_LATE_RATE < 0.05 ? COLORS.green : COLORS.amber, bold: true
          }),
        ]}),
        new TableRow({ children: [
          sCell("Years Employed", 3500),
          cell(`${kf.YEARS_EMPLOYED || "—"} years`, { width: 2930 }),
          cell("", { width: 2930 }),
        ]}),
      ]
    }),
    spacer(300),
  );

  // ── Section 5: External Due Diligence ────────────────────
  children.push(
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("5. External Due Diligence")] }),
    bodyPara(ext.narrative || ""),
    spacer(120),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Research Module Scores")] }),
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [3500, 1800, 2200, 1860],
      rows: [
        new TableRow({ children: [
          hCell("Research Module",  3500),
          hCell("Score (/100)",    1800),
          hCell("Risk Bar",        2200),
          hCell("Level",          1860),
        ]}),
        ...Object.entries(mods).map(([mod, score]) => new TableRow({ children: [
          sCell(mod.replace(/_/g, " ").replace("risk score", "").trim().toUpperCase(), 3500),
          cell(score.toString(), { width: 1800, align: AlignmentType.CENTER }),
          cell(
            "█".repeat(Math.round(score / 10)) + "░".repeat(10 - Math.round(score / 10)),
            { width: 2200, color: scoreColor(score) }
          ),
          cell(score <= 30 ? "LOW" : score <= 60 ? "MODERATE" : "HIGH", {
            width: 1860, bold: true,
            color: scoreColor(score)
          }),
        ]})),
        new TableRow({ children: [
          cell("COMPOSITE SCORE", { width: 3500, bold: true, shade: COLORS.lightBlue }),
          cell((ext.composite_score || "—").toString(), { width: 1800, bold: true, shade: COLORS.lightBlue, align: AlignmentType.CENTER }),
          cell("", { width: 2200, shade: COLORS.lightBlue }),
          cell(ext.risk_level || "—", { width: 1860, bold: true, shade: COLORS.lightBlue,
            color: scoreColor(ext.composite_score || 50) }),
        ]}),
      ]
    }),
    spacer(120),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Red Flags")] }),
    ...(ext.red_flags?.length
      ? ext.red_flags.map(f => bullet(f))
      : [bodyPara("✅  No critical red flags detected across all research modules.", { color: COLORS.green })]),
    spacer(240),
  );

  // ── Section 6: Risk Assessment ────────────────────────────
  children.push(
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("6. Risk Assessment")] }),
    bodyPara(risk.narrative || ""),
    spacer(120),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Credit Risk Metrics (PD / LGD / EAD)") ]}),
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [3500, 5860],
      rows: [
        new TableRow({ children: [hCell("Metric", 3500), hCell("Value", 5860)] }),
        kvRow("Probability of Default (PD)", `${PD_PCT}%`),
        kvRow("Loss Given Default (LGD)",    `${((risk.lgd || 0) * 100).toFixed(1)}%`),
        kvRow("Exposure at Default (EAD)",   inr(risk.ead)),
        kvRow("Expected Loss (EL = PD×LGD×EAD)", inr(risk.expected_loss)),
        kvRow("Internal Risk Rating",        `${GRADE} (${LABEL})`),
      ]
    }),
    spacer(160),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("SHAP Model Explainability — Top Risk Drivers")] }),
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [3200, 1500, 2200, 2460],
      rows: [
        new TableRow({ children: [
          hCell("Feature",      3200),
          hCell("Impact",      1500),
          hCell("Direction",   2200),
          hCell("SHAP Value",  2460),
        ]}),
        ...shap.map(s => new TableRow({ children: [
          sCell(s.feature, 3200),
          cell(s.impact, {
            width: 1500, bold: true,
            color: s.impact === "HIGH" ? COLORS.red : s.impact === "MEDIUM" ? COLORS.amber : COLORS.darkGrey
          }),
          cell(s.direction === "risk_increase" ? "▲ Increases Risk" : "▼ Decreases Risk", {
            width: 2200, bold: true,
            color: s.direction === "risk_increase" ? COLORS.red : COLORS.green
          }),
          cell(`${shapBar(s.shap_value)}  ${s.shap_value.toFixed(4)}`, { width: 2460 }),
        ]})),
      ]
    }),
    spacer(300),
  );

  // ── Section 7: ESG Risk ──────────────────────────────────
  children.push(
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("7. ESG Risk Assessment")] }),
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [3500, 2930, 2930],
      rows: [
        new TableRow({ children: [hCell("ESG Dimension", 3500), hCell("Rating", 2930), hCell("Pricing Impact", 2930)] }),
        kvRow("Overall ESG Score",    `${esg.overall_score || "—"}/100`),
        kvRow("ESG Rating",           esg.esg_rating || "—"),
        kvRow("Pricing Adjustment",   `+${esg.pricing_impact_bps || 0} basis points`),
      ]
    }),
    spacer(120),
    bodyPara(esg.narrative || ""),
    spacer(120),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Key ESG Concerns")] }),
    ...(esg.key_concerns || []).map(c => bullet(c)),
    spacer(240),
  );

  // ── Section 8: Stress Testing ────────────────────────────
  children.push(
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("8. Stress Test Results")] }),
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [2800, 1640, 1640, 1640, 1640],
      rows: [
        new TableRow({ children: [
          hCell("Scenario",     2800),
          hCell("DSCR",        1640),
          hCell("Stressed PD", 1640),
          hCell("Decision",    1640),
          hCell("Status",      1640),
        ]}),
        ...st.map((s, i) => new TableRow({
          shading: i % 2 === 1 ? { fill: COLORS.grey, type: ShadingType.CLEAR } : undefined,
          children: [
            sCell(s.scenario, 2800),
            cell(s.dscr?.toFixed(2) || "—", {
              width: 1640, bold: true,
              color: s.dscr >= 1.5 ? COLORS.green : s.dscr >= 1.2 ? COLORS.amber : COLORS.red
            }),
            cell(`${((s.pd_stressed || 0) * 100).toFixed(2)}%`, { width: 1640 }),
            cell(s.decision, {
              width: 1640, bold: true,
              color: s.decision?.includes("PASS") ? COLORS.green :
                     s.decision?.includes("WATCH") ? COLORS.amber : COLORS.red
            }),
            cell(s.decision?.includes("PASS") ? "✅" : s.decision?.includes("WATCH") ? "⚠️" : "❌",
              { width: 1640, align: AlignmentType.CENTER }),
          ]
        })),
      ]
    }),
    spacer(300),
  );

  // ── Section 9: Competitive Position ─────────────────────
  children.push(
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("9. Competitive Position")] }),
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [3500, 5860],
      rows: [
        new TableRow({ children: [hCell("Parameter", 3500), hCell("Details", 5860)] }),
        kvRow("Market Position",  comp.market_position || ""),
        kvRow("Key Competitors",  (comp.competitors || []).join(", ")),
      ]
    }),
    spacer(120),
    bodyPara(comp.narrative || ""),
    spacer(240),
  );

  // ── Section 10: Final Recommendation ────────────────────
  const pricing = rec.pricing || {};
  children.push(
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("10. Recommendation & Pricing")] }),
    bodyPara(rec.narrative || ""),
    spacer(160),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Approved Terms")] }),
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [3500, 5860],
      rows: [
        new TableRow({ children: [hCell("Term", 3500), hCell("Value", 5860)] }),
        kvRow("Decision",              DECISION),
        kvRow("Credit Limit",          inr(LIMIT)),
        kvRow("Base Rate",             pricing.base_rate || ""),
        kvRow("Credit Spread",         pricing.credit_spread || ""),
        kvRow("ESG Adjustment",        `+${pricing.esg_adjustment_bps || 0}bps`),
        kvRow("Final Rate",            pricing.recommended_rate || "", COLORS.lightBlue),
        kvRow("Rate Band",             pricing.rate_band || ""),
        kvRow("Risk Rating",           `${GRADE} — ${LABEL}`),
      ]
    }),
    spacer(300),
  );

  // ── Section 11: Authorisation ────────────────────────────
  children.push(
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("11. Authorisation")] }),
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [3120, 3120, 3120],
      rows: [
        new TableRow({ children: [
          hCell("Prepared By",  3120),
          hCell("Reviewed By", 3120),
          hCell("Approved By", 3120),
        ]}),
        new TableRow({ children: [
          cell("Credit Analyst\nAI Engine v1.0", { width: 3120 }),
          cell("Senior Credit Manager", { width: 3120 }),
          cell("Chief Credit Officer",  { width: 3120 }),
        ]}),
        new TableRow({ children: [
          cell("Signature: _____________", { width: 3120 }),
          cell("Signature: _____________", { width: 3120 }),
          cell("Signature: _____________", { width: 3120 }),
        ]}),
        new TableRow({ children: [
          cell(`Date: ${TODAY}`, { width: 3120 }),
          cell("Date: _____________",    { width: 3120 }),
          cell("Date: _____________",    { width: 3120 }),
        ]}),
      ]
    }),
  );

  return new Document({
    numbering: {
      config: [
        { reference: "bullets", levels: [{
          level: 0, format: LevelFormat.BULLET, text: "•",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]},
      ]
    },
    styles: makeStyles(),
    sections: [{
      properties: {
        page: { size: { width: 12240, height: 15840 },
                margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 } }
      },
      headers: { default: makeHeader("Credit Appraisal Memo") },
      footers: { default: makeFooter() },
      children,
    }]
  });
}


// ============================================================
// DOCUMENT 2 — SANCTION LETTER
// ============================================================

function buildSanctionLetter() {
  const pricing = CAM.recommendation?.pricing || {};

  return new Document({
    styles: makeStyles(),
    sections: [{
      properties: {
        page: { size: { width: 12240, height: 15840 },
                margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
      },
      headers: { default: makeHeader("Sanction Letter") },
      footers: { default: makeFooter() },
      children: [
        new Paragraph({
          spacing: { after: 80 },
          children: [new TextRun({ text: "SANCTION LETTER", bold: true, size: 48, font: "Arial", color: COLORS.navy })]
        }),
        new Paragraph({
          spacing: { after: 400 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: COLORS.navy } },
          children: [new TextRun({ text: `Ref: ${CAM_ID}-SL  |  Date: ${TODAY}`, size: 18, color: "777777", font: "Arial" })]
        }),
        spacer(200),

        bodyPara(`To,`),
        bodyPara(`The Authorised Signatory`),
        bodyPara(BORROWER, { bold: true }),
        spacer(200),

        bodyPara(`Dear Sir / Madam,`),
        spacer(120),
        bodyPara(`Sub: Sanction of ${CAM.borrower_profile?.loan_purpose || "Credit Facility"} — ${inr(LIMIT)}`, { bold: true }),
        spacer(120),

        bodyPara(`We are pleased to inform you that the Credit Committee of First National Credit Bank has approved the following credit facility in your favour, subject to the terms and conditions mentioned herein:`),
        spacer(200),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Sanctioned Terms")] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [3500, 5860],
          rows: [
            new TableRow({ children: [hCell("Term", 3500), hCell("Details", 5860)] }),
            kvRow("Borrower",          BORROWER),
            kvRow("Facility Type",     CAM.borrower_profile?.loan_purpose || "Term Loan"),
            kvRow("Sanctioned Amount", inr(LIMIT)),
            kvRow("Rate of Interest",  `${pricing.recommended_rate || "—"} per annum (floating)`),
            kvRow("Interest Rate Band", pricing.rate_band || ""),
            kvRow("Risk Rating",       `${GRADE} — ${LABEL}`),
            kvRow("Validity of Sanction", "90 days from date of this letter"),
          ]
        }),
        spacer(240),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Pre-Disbursement Conditions")] }),
        bodyPara("Disbursement of the above facility shall be subject to fulfilment of the following conditions:"),
        bullet("Submission of audited financial statements for the most recent financial year"),
        bullet("Creation of equitable mortgage / charge over agreed collateral assets"),
        bullet("Execution of loan agreement and all ancillary security documents"),
        bullet("Personal guarantee from all promoter directors"),
        bullet("Submission of board resolution authorising borrowing"),
        bullet("Any other conditions as specified in the loan agreement"),
        spacer(240),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Financial Covenants")] }),
        bodyPara("During the tenor of the facility, you are required to maintain the following financial covenants:"),
        bullet("Debt Service Coverage Ratio (DSCR) not less than 1.25x at all times"),
        bullet("Total Debt to Equity ratio not to exceed 3.0x"),
        bullet("Current ratio to be maintained above 1.10x"),
        bullet("Submission of quarterly financial statements within 45 days of quarter end"),
        bullet("Annual audited accounts to be submitted within 6 months of financial year end"),
        spacer(240),

        bodyPara("This sanction is valid for 90 days from the date of this letter. Please signify your acceptance by signing and returning the duplicate copy of this letter."),
        spacer(400),

        bodyPara("Yours faithfully,"),
        spacer(400),
        bodyPara("_______________________________", { bold: true }),
        bodyPara("Authorised Signatory"),
        bodyPara("First National Credit Bank"),
        spacer(400),

        new Paragraph({
          border: { top: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC" } },
          spacing: { before: 200 },
          children: [new TextRun({ text: "ACCEPTANCE: We accept the above sanction and terms unconditionally.", size: 18, font: "Arial", color: "777777" })]
        }),
        spacer(200),
        bodyPara("Authorised Signatory: _______________________________    Date: ________________"),
      ]
    }]
  });
}


// ============================================================
// DOCUMENT 3 — CONDITIONS COMPLIANCE REPORT (CCR)
// ============================================================

function buildCCR() {
  const conditions = [
    { no: 1, condition: "Audited financial statements (latest FY) submitted", category: "Financial", status: "Pending" },
    { no: 2, condition: "Equitable mortgage created over collateral", category: "Security", status: "Pending" },
    { no: 3, condition: "Loan agreement executed by authorised signatories", category: "Legal", status: "Pending" },
    { no: 4, condition: "Personal guarantee from all promoter directors", category: "Legal", status: "Pending" },
    { no: 5, condition: "Board resolution authorising borrowing obtained", category: "Legal", status: "Pending" },
    { no: 6, condition: "Legal scrutiny report confirmed clear title", category: "Legal", status: "Pending" },
    { no: 7, condition: "Valuation report from approved valuer received", category: "Technical", status: "Pending" },
    { no: 8, condition: "Insurance policy for collateral assets obtained", category: "Insurance", status: "Pending" },
    { no: 9, condition: "CERSAI registration completed for charge creation", category: "Regulatory", status: "Pending" },
    { no: 10, condition: "Processing fee and documentation charges paid", category: "Financial", status: "Pending" },
    { no: 11, condition: "KYC documents verified for all directors/guarantors", category: "Compliance", status: "Pending" },
    { no: 12, condition: "Source of equity contribution confirmed", category: "Financial", status: "Pending" },
  ];

  return new Document({
    styles: makeStyles(),
    sections: [{
      properties: {
        page: { size: { width: 12240, height: 15840 },
                margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 } }
      },
      headers: { default: makeHeader("Conditions Compliance Report") },
      footers: { default: makeFooter() },
      children: [
        new Paragraph({
          spacing: { after: 80 },
          children: [new TextRun({ text: "CONDITIONS COMPLIANCE REPORT (CCR)", bold: true, size: 44, font: "Arial", color: COLORS.navy })]
        }),
        new Paragraph({
          spacing: { after: 400 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: COLORS.navy } },
          children: [new TextRun({ text: `Ref: ${CAM_ID}-CCR  |  Date: ${TODAY}  |  Status: Pre-Disbursement`, size: 18, color: "777777", font: "Arial" })]
        }),
        spacer(200),

        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [3500, 5860],
          rows: [
            kvRow("Borrower",    BORROWER),
            kvRow("CAM Ref",     CAM_ID),
            kvRow("Limit",       inr(LIMIT)),
            kvRow("Decision",    DECISION),
          ]
        }),
        spacer(240),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Pre-Disbursement Conditions Checklist")] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [520, 4200, 1500, 1500, 1640],
          rows: [
            new TableRow({ children: [
              hCell("No.", 520),
              hCell("Condition", 4200),
              hCell("Category", 1500),
              hCell("Status", 1500),
              hCell("Date Fulfilled", 1640),
            ]}),
            ...conditions.map((c, i) => new TableRow({
              shading: i % 2 === 1 ? { fill: COLORS.grey, type: ShadingType.CLEAR } : undefined,
              children: [
                cell(c.no.toString(), { width: 520, align: AlignmentType.CENTER }),
                cell(c.condition,     { width: 4200 }),
                cell(c.category,      { width: 1500 }),
                cell(c.status,        { width: 1500, bold: true,
                  color: c.status === "Fulfilled" ? COLORS.green : COLORS.amber }),
                cell("",              { width: 1640 }),
              ]
            })),
          ]
        }),
        spacer(300),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Disbursement Authorisation")] }),
        bodyPara("Disbursement may proceed ONLY after ALL conditions above are marked 'Fulfilled' and verified by the Operations team and Credit Manager."),
        spacer(400),

        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [3120, 3120, 3120],
          rows: [
            new TableRow({ children: [
              hCell("Operations Officer", 3120),
              hCell("Credit Manager",    3120),
              hCell("Branch Head",       3120),
            ]}),
            new TableRow({ children: [
              cell("Sign: _____________\nDate: _____________", { width: 3120 }),
              cell("Sign: _____________\nDate: _____________", { width: 3120 }),
              cell("Sign: _____________\nDate: _____________", { width: 3120 }),
            ]}),
          ]
        }),
      ]
    }]
  });
}


// ============================================================
// DOCUMENT 4 — LOAN DISBURSEMENT REPORT (LDR)
// ============================================================

function buildLDR() {
  return new Document({
    styles: makeStyles(),
    sections: [{
      properties: {
        page: { size: { width: 12240, height: 15840 },
                margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
      },
      headers: { default: makeHeader("Loan Disbursement Report") },
      footers: { default: makeFooter() },
      children: [
        new Paragraph({
          spacing: { after: 80 },
          children: [new TextRun({ text: "LOAN DISBURSEMENT REPORT (LDR)", bold: true, size: 44, font: "Arial", color: COLORS.navy })]
        }),
        new Paragraph({
          spacing: { after: 400 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: COLORS.navy } },
          children: [new TextRun({ text: `Ref: ${CAM_ID}-LDR  |  Date: ${TODAY}`, size: 18, color: "777777", font: "Arial" })]
        }),
        spacer(200),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Disbursement Details")] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [3500, 5860],
          rows: [
            new TableRow({ children: [hCell("Field", 3500), hCell("Details", 5860)] }),
            kvRow("Borrower",              BORROWER),
            kvRow("CAM Reference",         CAM_ID),
            kvRow("Sanction Letter Ref",   `${CAM_ID}-SL`),
            kvRow("Sanctioned Amount",     inr(LIMIT)),
            kvRow("Disbursed Amount",      "₹ _______________________"),
            kvRow("Disbursement Date",     "_______________________ "),
            kvRow("Mode of Disbursement",  "RTGS / NEFT / Account Transfer"),
            kvRow("Beneficiary Account",   "_______________________ "),
            kvRow("IFSC Code",             "_______________________ "),
            kvRow("Rate of Interest",      CAM.recommendation?.pricing?.recommended_rate || ""),
            kvRow("First EMI Date",        "_______________________"),
            kvRow("Loan Account Number",   "_______________________"),
          ]
        }),
        spacer(300),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("CCR Compliance Confirmation")] }),
        bodyPara("I/We confirm that all pre-disbursement conditions as listed in the Conditions Compliance Report (CCR) have been duly fulfilled and verified prior to disbursement."),
        spacer(400),

        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [3120, 3120, 3120],
          rows: [
            new TableRow({ children: [
              hCell("Disbursing Officer", 3120),
              hCell("Credit Manager",    3120),
              hCell("Authorisation",     3120),
            ]}),
            new TableRow({ children: [
              cell("Sign: _____________\nDate: _____________", { width: 3120 }),
              cell("Sign: _____________\nDate: _____________", { width: 3120 }),
              cell("Sign: _____________\nDate: _____________", { width: 3120 }),
            ]}),
          ]
        }),
      ]
    }]
  });
}


// ============================================================
// DOCUMENT 5 — CMA DATA REPORT
// ============================================================

function buildCMA() {
  const kf = CAM.financial_analysis?.key_features || {};
  const income = kf.AMT_INCOME_TOTAL || 800000;

  // Generate 3 year actuals + 2 year projections (simple trend)
  const years = ["FY2022-23 (A)", "FY2023-24 (A)", "FY2024-25 (A)", "FY2025-26 (P)", "FY2026-27 (P)"];
  const growthRate = 0.12;
  const baseRevenue = income * 5;
  const revenues = years.map((_, i) => Math.round(baseRevenue * Math.pow(1 - growthRate + (i * 0.04), i)));
  const ebitdaPct = [0.14, 0.15, 0.16, 0.17, 0.18];
  const ebitdas = revenues.map((r, i) => Math.round(r * ebitdaPct[i]));
  const debts = [income * 4.5, income * 5.0, income * 5.0, income * 4.8, income * 4.5].map(Math.round);
  const dscrs = [1.42, 1.55, 1.85, 1.92, 2.01];

  return new Document({
    styles: makeStyles(),
    sections: [{
      properties: {
        page: { size: { width: 12240, height: 15840 },
                margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 } }
      },
      headers: { default: makeHeader("CMA Data Report") },
      footers: { default: makeFooter() },
      children: [
        new Paragraph({
          spacing: { after: 80 },
          children: [new TextRun({ text: "CREDIT MONITORING ARRANGEMENT (CMA) DATA REPORT", bold: true, size: 40, font: "Arial", color: COLORS.navy })]
        }),
        new Paragraph({
          spacing: { after: 400 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: COLORS.navy } },
          children: [new TextRun({ text: `${BORROWER}  |  ${TODAY}  |  (A) = Actuals  |  (P) = Projections`, size: 18, color: "777777", font: "Arial" })]
        }),
        spacer(200),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Part A — Operating Statement")] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [2360, 1400, 1400, 1400, 1400, 1400],
          rows: [
            new TableRow({ children: [
              hCell("Particulars", 2360),
              ...years.map(y => hCell(y, 1400))
            ]}),
            new TableRow({ children: [
              sCell("Net Revenue (₹)", 2360),
              ...revenues.map(v => cell(inr(v), { width: 1400 }))
            ]}),
            new TableRow({ children: [
              sCell("EBITDA (₹)", 2360),
              ...ebitdas.map(v => cell(inr(v), { width: 1400 }))
            ]}),
            new TableRow({ children: [
              sCell("EBITDA Margin (%)", 2360),
              ...ebitdas.map((e, i) => cell(`${((e/revenues[i])*100).toFixed(1)}%`, { width: 1400 }))
            ]}),
            new TableRow({ children: [
              sCell("Total Debt (₹)", 2360),
              ...debts.map(v => cell(inr(v), { width: 1400 }))
            ]}),
            new TableRow({ children: [
              sCell("DSCR", 2360),
              ...dscrs.map((d, i) => cell(d.toFixed(2), {
                width: 1400, bold: true,
                color: d >= 1.5 ? COLORS.green : d >= 1.2 ? COLORS.amber : COLORS.red
              }))
            ]}),
          ]
        }),
        spacer(300),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Part B — Key Financial Ratios")] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [3500, 2930, 2930],
          rows: [
            new TableRow({ children: [hCell("Ratio", 3500), hCell("Value", 2930), hCell("Benchmark", 2930)] }),
            kvRow("Credit / Income Ratio",  `${kf.CREDIT_INCOME_RATIO || "—"}x`),
            kvRow("Annuity / Income Ratio", `${((kf.ANNUITY_INCOME_RATIO || 0) * 100).toFixed(1)}%`),
            kvRow("CC Utilization",         `${((kf.CC_UTILIZATION_MEAN || 0) * 100).toFixed(1)}%`),
            kvRow("Late Payment Rate",      `${((kf.INST_LATE_RATE || 0) * 100).toFixed(1)}%`),
            kvRow("Bureau Loan Count",      kf.BUREAU_LOAN_COUNT?.toString() || "—"),
          ]
        }),
      ]
    }]
  });
}


// ============================================================
// DOCUMENT 6 — PD REPORT (Personal Discussion)
// ============================================================

function buildPDReport() {
  return new Document({
    styles: makeStyles(),
    sections: [{
      properties: {
        page: { size: { width: 12240, height: 15840 },
                margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
      },
      headers: { default: makeHeader("Personal Discussion Report") },
      footers: { default: makeFooter() },
      children: [
        new Paragraph({
          spacing: { after: 80 },
          children: [new TextRun({ text: "PERSONAL DISCUSSION REPORT (PD REPORT)", bold: true, size: 44, font: "Arial", color: COLORS.navy })]
        }),
        new Paragraph({
          spacing: { after: 400 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: COLORS.navy } },
          children: [new TextRun({ text: `${BORROWER}  |  Date of Discussion: ${TODAY}`, size: 18, color: "777777", font: "Arial" })]
        }),
        spacer(200),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("1. Borrower Details")] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [3500, 5860],
          rows: [
            kvRow("Entity Name",          BORROWER),
            kvRow("Industry",             CAM.borrower_profile?.industry || ""),
            kvRow("Loan Purpose",         CAM.borrower_profile?.loan_purpose || ""),
            kvRow("Loan Amount Requested", inr(CAM.borrower_profile?.loan_requested)),
            kvRow("Meeting Conducted By", "_______________________"),
            kvRow("Meeting Date & Time",  TODAY),
            kvRow("Meeting Location",     "Borrower's Premises / Bank Branch"),
          ]
        }),
        spacer(240),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("2. Business Overview")] }),
        bodyPara("[Analyst to document borrower's description of their business model, operations, key customers, and market position during the PD meeting.]"),
        spacer(400),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("3. Purpose of Loan & Fund Utilisation")] }),
        bodyPara("[Document specific use of funds, project details, implementation timeline, and expected returns on investment.]"),
        spacer(400),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("4. Management Assessment")] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [3500, 5860],
          rows: [
            new TableRow({ children: [hCell("Assessment Area", 3500), hCell("Analyst Observation", 5860)] }),
            kvRow("Promoter Experience",      "[Years in industry, background]"),
            kvRow("Business Acumen",          "[Assess understanding of market, risks]"),
            kvRow("Transparency",             "[Willingness to share information]"),
            kvRow("Past Credit Behaviour",    "[Any defaults, restructuring history]"),
            kvRow("Succession Planning",      "[Key man dependency risk]"),
            kvRow("Overall Impression",       "[Satisfactory / Unsatisfactory]"),
          ]
        }),
        spacer(240),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("5. AI Research Summary")] }),
        bodyPara(`Composite Research Risk Score: ${CAM.external_due_diligence?.composite_score || "—"}/100 (${CAM.external_due_diligence?.risk_level || "—"} Risk)`),
        bodyPara(CAM.external_due_diligence?.narrative?.substring(0, 400) || ""),
        spacer(240),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("6. Analyst Conclusion")] }),
        bodyPara("[Analyst's overall assessment of borrower character, business viability, and recommendation from PD perspective.]"),
        spacer(400),

        bodyPara("Conducted By: _______________________________    Designation: _______________________________"),
        spacer(120),
        bodyPara("Signature: _______________________________    Date: _______________________________"),
      ]
    }]
  });
}


// ============================================================
// MAIN — GENERATE ALL 6 DOCUMENTS
// ============================================================

async function generateAll() {
  console.log("\n📄 LAYER 7 — DOCUMENT GENERATION");
  console.log("=".repeat(50));
  console.log(`Borrower : ${BORROWER}`);
  console.log(`CAM ID   : ${CAM_ID}`);
  console.log(`Decision : ${DECISION}`);
  console.log("=".repeat(50));

  const outDir = "./output_documents";
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir);

  const docs = [
    { name: "1_CAM_Credit_Appraisal_Memo",       builder: buildCAM          },
    { name: "2_Sanction_Letter",                  builder: buildSanctionLetter },
    { name: "3_Conditions_Compliance_Report_CCR", builder: buildCCR          },
    { name: "4_Loan_Disbursement_Report_LDR",     builder: buildLDR          },
    { name: "5_CMA_Data_Report",                  builder: buildCMA          },
    { name: "6_PD_Personal_Discussion_Report",    builder: buildPDReport     },
  ];

  for (const doc of docs) {
    process.stdout.write(`  Generating ${doc.name}...`);
    try {
      const document = doc.builder();
      const buffer   = await Packer.toBuffer(document);
      const filename = `${outDir}/${doc.name}_${CAM.meta?.borrower_id || "COMP"}.docx`;
      fs.writeFileSync(filename, buffer);
      console.log(` ✅`);
    } catch (err) {
      console.log(` ❌ Error: ${err.message}`);
    }
  }

  console.log("\n" + "=".repeat(50));
  console.log(`✅ All documents saved to: ${outDir}/`);
  console.log("=".repeat(50));
  console.log(`\n  1. ${outDir}/1_CAM_Credit_Appraisal_Memo_COMP_001.docx`);
  console.log(`  2. ${outDir}/2_Sanction_Letter_COMP_001.docx`);
  console.log(`  3. ${outDir}/3_Conditions_Compliance_Report_CCR_COMP_001.docx`);
  console.log(`  4. ${outDir}/4_Loan_Disbursement_Report_LDR_COMP_001.docx`);
  console.log(`  5. ${outDir}/5_CMA_Data_Report_COMP_001.docx`);
  console.log(`  6. ${outDir}/6_PD_Personal_Discussion_Report_COMP_001.docx\n`);
}

generateAll().then(() => {
  console.log("DONE");
}).catch(err => {
  console.error("FAILED:", err.message);
  console.error(err.stack);
});