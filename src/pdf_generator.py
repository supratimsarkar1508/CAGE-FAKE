"""
CAGE-Fake: Publication-Quality PDF Report Generator
Generates comprehensive diagnostic audit reports using ReportLab.
Features modern executive layout, dynamic 2-page budgeting, running headers/footers
with 'Page X of Y', multi-model consensus meter, credibility scorecard, claim-level
evidence grounding, local feature attributions, empirical benchmarks, and embedded plots.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
    PageBreak,
)
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from src.utils import setup_logger

logger = setup_logger("pdf_generator")

TRIGGER_LABELS = {
    "missing_evidence": "Missing Corroborating Evidence",
    "conflicting_evidence": "Conflicting Evidence Across Claims",
    "signal_disagreement": "Disagreement: Signals vs Evidence",
    "both_triggers": "Dual Safety Triggers (Confidence & Disagreement)",
    "low_confidence": "Low Calibrated Confidence",
    "high_disagreement": "High Base Disagreement",
    "evidence_conflict": "External Evidence Conflict",
    "none": "Verified Evidence & Model Consensus",
}


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas that accumulates total page count and renders
    sleek running headers and footers with 'Page X of Y'.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running header on pages > 1
        if self._pageNumber > 1:
            self.drawString(36, 762, "CAGE-Fake: Article Trustworthiness Audit Report")
            self.drawRightString(576, 762, "Calibration-Aware AI Diagnostic Audit")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.6)
            self.line(36, 756, 576, 756)

        # Running footer on all pages
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.6)
        self.line(36, 26, 576, 26)

        self.drawString(36, 15, "Confidential AI Diagnostic Audit • Powered by CAGE-Fake Multi-Model Ensemble")
        self.drawRightString(576, 15, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def generate_pdf_report(
    output_path: Path | str,
    article_meta: Dict[str, Any],
    gate_result: Dict[str, Any],
    disagreement_info: Dict[str, Any],
    credibility_info: Dict[str, Any],
    counterfactual_info: Dict[str, Any],
    why_explanation: str,
    local_features: Dict[str, Any],
    comparison_table: Optional[List[Dict[str, Any]]] = None,
    figures: Optional[Dict[str, Path]] = None,
    claim_verifications: Optional[List[Dict[str, Any]]] = None,
    linguistic_red_flags: Optional[List[str]] = None,
    reasoning_summary: Optional[str] = None,
) -> Path:
    """
    Builds and saves a publication-quality 2-page PDF audit report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 540 pt printable width (letter = 612 x 792 pt, margins = 36 pt left/right, 40 pt top/bottom)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=40,
        bottomMargin=38,
    )

    styles = getSampleStyleSheet()

    # Typography & Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#475569"),
    )
    meta_box_style = ParagraphStyle(
        "MetaBox",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9.5,
        alignment=2,
        textColor=colors.HexColor("#64748b"),
    )
    h2_style = ParagraphStyle(
        "Heading2Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=6,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor("#854d0e"),
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9,
        textColor=colors.HexColor("#334155"),
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9,
        textColor=colors.HexColor("#0f172a"),
    )
    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9,
        alignment=1,
        textColor=colors.HexColor("#334155"),
    )
    table_cell_center_bold = ParagraphStyle(
        "TableCellCenterBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9,
        alignment=1,
        textColor=colors.HexColor("#0f172a"),
    )

    story = []

    # =========================================================================
    # PAGE 1: EXECUTIVE DIAGNOSTIC OVERVIEW
    # =========================================================================

    # 1. Header Banner
    timestamp_str = article_meta.get("timestamp") or time.strftime("%Y-%m-%d %H:%M:%S")
    audit_id = f"AUD-{int(time.time()) % 1000000:06d}"

    header_table = Table(
        [[
            Paragraph(
                "<b>CAGE-Fake: Article Trustworthiness Audit Report</b><br/>"
                "<font color='#475569'>Calibration-Aware Gating & Evidence-Grounded Reasoning for Trustworthy Fake News Triage</font>",
                title_style,
            ),
            Paragraph(
                f"<b>AUDIT ID:</b> {audit_id}<br/>"
                f"<b>DATE:</b> {timestamp_str}<br/>"
                "<b>FRAMEWORK:</b> CAGE-Fake v2.0",
                meta_box_style,
            )
        ]],
        colWidths=[390, 150]
    )
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.8, color=colors.HexColor("#2563eb"), spaceAfter=5, spaceBefore=0))

    # 2. Disclaimer Card
    disclaimer_text = (
        "<b>LEGAL & FACT-CHECKING POLICY:</b> A trained fake-news classifier predicts linguistic, stylometric, "
        "and semantic patterns learned from benchmark datasets; it does <b>NOT</b> independently verify factual reality. "
        "URL extraction, pattern recognition, and external evidence verification remain strictly separate processes. "
        "The pipeline abstains (<b>UNCERTAIN</b>) whenever evidence is missing, conflicting, or model consensus is low."
    )
    disclaimer_table = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[540])
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#fde68a")),
        ("LINELEFT", (0, 0), (0, -1), 3.0, colors.HexColor("#f59e0b")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(disclaimer_table)
    story.append(Spacer(1, 6))

    # 3. Article Metadata & Verdict Hero Card
    decision = gate_result.get("decision", "UNCERTAIN")
    conf = gate_result.get("confidence", 0.0)
    disagreement = gate_result.get("disagreement", 0.0)
    trigger = gate_result.get("trigger", "none")
    tau_val = gate_result.get("tau", 0.80)
    delta_val = gate_result.get("delta", 0.20)

    if decision == "REAL":
        v_bg = colors.HexColor("#ecfdf5")
        v_border = colors.HexColor("#059669")
        v_title_color = "#047857"
    elif decision == "FAKE":
        v_bg = colors.HexColor("#fef2f2")
        v_border = colors.HexColor("#dc2626")
        v_title_color = "#b91c1c"
    else:
        v_bg = colors.HexColor("#fffbeb")
        v_border = colors.HexColor("#d97706")
        v_title_color = "#b45309"

    title_text = article_meta.get("title", "Untitled Article")
    if len(title_text) > 110:
        title_text = title_text[:107] + "..."
    url_text = article_meta.get("url") or "Manual Text Submission"
    if len(url_text) > 65:
        url_text = url_text[:62] + "..."

    source_text = (
        f"<b>Title:</b> {title_text}<br/>"
        f"<b>Source:</b> <font color='#2563eb'>{url_text}</font><br/>"
        f"<b>Word Count:</b> {article_meta.get('word_count', 0):,} words | "
        f"<b>Extraction:</b> {article_meta.get('extraction_status', 'OK')}<br/>"
        f"<b>Triage Policy:</b> Evidence-Grounded Three-Way Gating"
    )

    trigger_display = TRIGGER_LABELS.get(trigger, trigger.replace("_", " ").title())
    verdict_text = (
        f"<font size=13 color='{v_title_color}'><b>TRIAGE VERDICT: {decision}</b></font><br/>"
        f"<b>Calibrated Confidence:</b> {conf:.1%} <font color='#64748b'>(&tau; = {tau_val:.1%})</font><br/>"
        f"<b>Model Disagreement (std):</b> {disagreement:.3f} <font color='#64748b'>(&delta; = {delta_val:.2f})</font><br/>"
        f"<b>Safety Trigger:</b> {trigger_display}"
    )

    meta_table = Table(
        [[
            Paragraph(source_text, body_style),
            Paragraph(verdict_text, body_style)
        ]],
        colWidths=[325, 215]
    )
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#f8fafc")),
        ("BACKGROUND", (1, 0), (1, 0), v_bg),
        ("BOX", (0, 0), (0, 0), 0.8, colors.HexColor("#cbd5e1")),
        ("BOX", (1, 0), (1, 0), 1.5, v_border),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(meta_table)

    if article_meta.get("short_text_warning"):
        warn_p = Paragraph(f"<b>CAUTION: Short Text Length ({article_meta.get('word_count', 0)} words):</b> {article_meta['short_text_warning']}", disclaimer_style)
        warn_table = Table([[warn_p]], colWidths=[540])
        warn_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
            ("BORDER", (0, 0), (-1, -1), 0.5, colors.HexColor("#f59e0b")),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(Spacer(1, 3))
        story.append(warn_table)

    story.append(Spacer(1, 5))

    # 4. Section 1: Multi-Model Disagreement Analysis
    story.append(Paragraph("1. Multi-Model Disagreement Analysis (5 Models)", h2_style))
    meter_header = [
        Paragraph("<b>Model Name</b>", table_cell_bold),
        Paragraph("<b>P(FAKE)</b>", table_cell_center_bold),
        Paragraph("<b>Vote</b>", table_cell_center_bold),
        Paragraph("<b>Confidence</b>", table_cell_center_bold),
    ]
    meter_rows = [meter_header]
    for idx, m in enumerate(disagreement_info.get("models", [])):
        v_col = "#dc2626" if m["vote"] == "FAKE" else "#16a34a"
        meter_rows.append([
            Paragraph(m["name"], table_cell_style),
            Paragraph(f"{m['prob_fake']:.1%}", table_cell_center),
            Paragraph(f"<font color='{v_col}'><b>{m['vote']}</b></font>", table_cell_center),
            Paragraph(f"{m['confidence']:.1%}", table_cell_center),
        ])

    disagree_table = Table(meter_rows, colWidths=[120, 60, 65, 75])
    disagree_table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 2.8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in range(1, len(meter_rows)):
        if i % 2 == 0:
            disagree_table_style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8fafc")))
    disagree_table.setStyle(TableStyle(disagree_table_style))

    disagree_std_val = disagreement_info.get("gate_base_disagreement", disagreement_info.get("std_dev", 0.0))
    five_std_val = disagreement_info.get("five_model_std", disagreement_info.get("std_dev", 0.0))
    consensus_level = disagreement_info.get("consensus_level", "Evaluated")
    c_status_color = "#16a34a" if "Consensus" in consensus_level else ("#d97706" if "Moderate" in consensus_level else "#dc2626")

    meter_bar_str = disagreement_info.get("meter_bar", "")
    bar_display = f"<font color='#2563eb'><b>{meter_bar_str}</b></font><br/>" if meter_bar_str else ""

    disagree_summary = (
        f"<b>Consensus Status:</b> <font color='{c_status_color}'><b>{consensus_level}</b></font><br/>"
        f"<b>Vote Distribution:</b> <b>{disagreement_info.get('fake_votes', 0)} FAKE</b> vs <b>{disagreement_info.get('real_votes', 0)} REAL</b><br/>"
        f"<b>Base Disagreement (4-base std):</b> {disagree_std_val:.3f} <font color='#64748b'>(&delta; = {delta_val:.2f})</font><br/>"
        f"<b>5-Model Consensus Spread:</b> {five_std_val:.3f}<br/>"
        f"{bar_display}"
        f"<b>Safety Assessment:</b> {disagreement_info.get('status_desc', 'Evaluation completed.')}"
    )
    disagree_summary_box = Table([[Paragraph(disagree_summary, body_style)]], colWidths=[215])
    disagree_summary_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    story.append(Table([[disagree_table, disagree_summary_box]], colWidths=[320, 220], style=[
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(Spacer(1, 5))

    # 5. Section 2: News Credibility Fingerprint
    story.append(Paragraph("2. News Credibility Fingerprint (Linguistic & Stylistic Scorecard)", h2_style))
    fp_header = [
        Paragraph("<b>Credibility Dimension</b>", table_cell_bold),
        Paragraph("<b>Score (0-100)</b>", table_cell_center_bold),
        Paragraph("<b>Journalistic Assessment</b>", table_cell_style),
    ]
    fp_rows = [fp_header]
    for dim, score in credibility_info.get("scores", {}).items():
        assessment = "Excellent / Restrained" if score >= 80 else ("Adequate / Neutral" if score >= 60 else "Deceptive / Alert")
        score_color = "#16a34a" if score >= 80 else ("#d97706" if score >= 60 else "#dc2626")
        fp_rows.append([
            Paragraph(dim, table_cell_style),
            Paragraph(f"<font color='{score_color}'><b>{score:.1f}/100</b></font>", table_cell_center),
            Paragraph(assessment, table_cell_style),
        ])

    fp_table = Table(fp_rows, colWidths=[150, 65, 105])
    fp_table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 2.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in range(1, len(fp_rows)):
        if i % 2 == 0:
            fp_table_style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8fafc")))
    fp_table.setStyle(TableStyle(fp_table_style))

    grade_val = credibility_info.get("credibility_grade", "Grade B")
    grade_color = "#16a34a" if "A" in grade_val else ("#d97706" if "B" in grade_val else "#dc2626")
    overall_idx = credibility_info.get("overall_credibility_index", 75.0)

    fp_summary = (
        f"<font size=7 color='#64748b'>OVERALL CREDIBILITY INDEX</font><br/>"
        f"<font size=15 color='{grade_color}'><b>{overall_idx:.1f} / 100</b></font><br/>"
        f"<b>Credibility Tier:</b> <font color='{grade_color}'><b>{grade_val}</b></font><br/><br/>"
        f"<font size=7 color='#64748b'>Scores are standardized against verified newsroom copy standards: high factual attribution, balanced syntax, and low sensationalism maximize the index.</font>"
    )
    fp_summary_box = Table([[Paragraph(fp_summary, body_style)]], colWidths=[215])
    fp_summary_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    story.append(Table([[fp_table, fp_summary_box]], colWidths=[320, 220], style=[
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(Spacer(1, 5))

    # 6. Section 3: Reasoning & Grounded Diagnostic Explanation
    story.append(Paragraph("3. Grounded Diagnostic Reasoning & Counterfactual Sensitivity", h2_style))

    # Executive reasoning summary callout (if available)
    if reasoning_summary:
        r_box = Table([[Paragraph(f"<b>Executive Synthesis:</b> {reasoning_summary}", body_style)]], colWidths=[540])
        r_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), v_bg),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("LINELEFT", (0, 0), (0, -1), 3.0, v_border),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(r_box)
        story.append(Spacer(1, 4))

    formatted_why = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", why_explanation).replace("\n", "<br/>")
    story.append(Paragraph(f"<b>Diagnostic Explanation:</b> {formatted_why}", body_style))
    story.append(Spacer(1, 3))

    shift_val = counterfactual_info.get("signed_shift_pct", counterfactual_info.get("prob_shift", 0.0) * 100.0)
    shift_col = "#dc2626" if shift_val > 0 else "#16a34a"
    cf_summary = (
        f"<b>Counterfactual Simulation:</b> {counterfactual_info.get('summary', 'Analysis completed.')} "
        f"Original P(FAKE): <b>{counterfactual_info.get('original_prob', 0.0):.1%}</b> &rarr; "
        f"Transformed P(FAKE): <b>{counterfactual_info.get('simulated_prob', 0.0):.1%}</b> "
        f"<font color='{shift_col}'><b>(Signed Shift: {shift_val:+.1f}%)</b></font>"
    )
    story.append(Paragraph(cf_summary, body_style))

    # Linguistic Red Flags
    if linguistic_red_flags:
        rf_items = [f"<font color='#dc2626'><b>[!]</b></font> {rf}" for rf in linguistic_red_flags[:3]]
        rf_text = "<b>Linguistic Red Flags Detected:</b> " + " &nbsp;|&nbsp; ".join(rf_items)
        story.append(Spacer(1, 3))
        story.append(Paragraph(rf_text, body_style))
    else:
        clean_rf_text = "<font color='#16a34a'><b>[v] Clean Stylometrics:</b></font> No significant sensational exaggeration, clickbait patterns, or manipulative phrasing detected."
        story.append(Spacer(1, 3))
        story.append(Paragraph(clean_rf_text, body_style))

    # Force PageBreak to guarantee pristine 2-page pagination
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: EVIDENCE GROUNDING, BENCHMARKS & DIAGNOSTIC PLOTS
    # =========================================================================

    # 7. Section 4: Claim-Level Semantic Evidence Grounding
    story.append(Paragraph("4. Claim-Level Semantic Evidence Grounding (Triage Verification)", h2_style))
    if claim_verifications:
        ev_header = [
            Paragraph("<b>Extracted Factual Claim</b>", table_cell_bold),
            Paragraph("<b>Verification Status</b>", table_cell_center_bold),
            Paragraph("<b>Authoritative Source / Match</b>", table_cell_bold),
            Paragraph("<b>Evidence Context Quote</b>", table_cell_style),
        ]
        ev_rows = [ev_header]
        # Display top 4 claims to ensure layout stability
        for c in claim_verifications[:4]:
            st = c.get("status", "Unverified")
            sim = c.get("similarity", 0.0)
            if st == "Supported":
                st_html = f"<font color='#16a34a'><b>SUPPORTED</b></font><br/><font size=6.5 color='#64748b'>(Sim: {sim:.2f})</font>"
            elif st == "Contradicted":
                st_html = f"<font color='#dc2626'><b>CONTRADICTED</b></font><br/><font size=6.5 color='#64748b'>(Sim: {sim:.2f})</font>"
            else:
                st_html = f"<font color='#d97706'><b>UNVERIFIED</b></font><br/><font size=6.5 color='#64748b'>(Sim: {sim:.2f})</font>"

            claim_text = c.get("claim", "")
            if len(claim_text) > 85:
                claim_text = claim_text[:82] + "..."
            ev_snippet = c.get("evidence_text", "No corroborating evidence located.")
            if len(ev_snippet) > 85:
                ev_snippet = ev_snippet[:82] + "..."

            ev_rows.append([
                Paragraph(claim_text, table_cell_style),
                Paragraph(st_html, table_cell_center),
                Paragraph(c.get("source", "Retrieved Document"), table_cell_style),
                Paragraph(ev_snippet, table_cell_style),
            ])

        ev_table = Table(ev_rows, colWidths=[170, 75, 135, 160])
        ev_table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("PADDING", (0, 0), (-1, -1), 2.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
        for i in range(1, len(ev_rows)):
            if i % 2 == 0:
                ev_table_style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8fafc")))
        ev_table.setStyle(TableStyle(ev_table_style))
        story.append(ev_table)
    else:
        no_claims_p = Paragraph(
            "<i>No external claim-level verification data required for this article. Pipeline relied on linguistic stylometrics and calibrated model consensus.</i>",
            body_style
        )
        no_claims_table = Table([[no_claims_p]], colWidths=[540])
        no_claims_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(no_claims_table)

    story.append(Spacer(1, 5))

    # 8. Section 5: Local Signal Attributions
    story.append(Paragraph("5. Local Signal Attributions (Meta-Ensemble Impacts)", h2_style))
    top_fake = local_features.get("fake_signals") or local_features.get("top_fake_driving", [])
    top_real = local_features.get("real_signals") or local_features.get("top_real_driving", [])
    top_fake = top_fake[:3]
    top_real = top_real[:3]

    feat_header = [
        Paragraph("<b>Top FAKE-Driving Indicators</b>", table_cell_bold),
        Paragraph("<b>Impact</b>", table_cell_center_bold),
        Paragraph("<b>Top REAL-Driving Anchors</b>", table_cell_bold),
        Paragraph("<b>Impact</b>", table_cell_center_bold),
    ]
    feat_rows = [feat_header]
    max_len = max(len(top_fake), len(top_real), 1)
    for i in range(max_len):
        f_fake = (top_fake[i].get("name") or top_fake[i].get("feature", "-")) if i < len(top_fake) else "-"
        i_fake = f"{top_fake[i]['impact']:+.3f}" if i < len(top_fake) else "-"
        f_real = (top_real[i].get("name") or top_real[i].get("feature", "-")) if i < len(top_real) else "-"
        i_real = f"{top_real[i]['impact']:+.3f}" if i < len(top_real) else "-"
        feat_rows.append([
            Paragraph(f_fake, table_cell_style),
            Paragraph(f"<font color='#dc2626'><b>{i_fake}</b></font>", table_cell_center),
            Paragraph(f_real, table_cell_style),
            Paragraph(f"<font color='#16a34a'><b>{i_real}</b></font>", table_cell_center),
        ])

    feat_table = Table(feat_rows, colWidths=[170, 80, 210, 80])
    feat_table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 2.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in range(1, len(feat_rows)):
        if i % 2 == 0:
            feat_table_style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8fafc")))
    feat_table.setStyle(TableStyle(feat_table_style))
    story.append(feat_table)
    story.append(Spacer(1, 5))

    # 9. Section 6: Global Benchmark Comparison Table
    if comparison_table:
        story.append(Paragraph("6. Empirical Benchmark Model Comparison (Held-out Test Split)", h2_style))
        comp_header = [
            Paragraph("<b>Model Name</b>", table_cell_bold),
            Paragraph("<b>Accuracy</b>", table_cell_center_bold),
            Paragraph("<b>Macro-F1</b>", table_cell_center_bold),
            Paragraph("<b>ECE</b>", table_cell_center_bold),
            Paragraph("<b>Coverage</b>", table_cell_center_bold),
            Paragraph("<b>Selective Acc</b>", table_cell_center_bold),
        ]
        comp_rows = [comp_header]
        cage_row_idx = None
        for idx, row in enumerate(comparison_table, start=1):
            is_cage = "CAGE" in str(row.get("model_name", ""))
            if is_cage:
                cage_row_idx = idx
            row_style = table_cell_bold if is_cage else table_cell_style
            row_center = table_cell_center_bold if is_cage else table_cell_center
            comp_rows.append([
                Paragraph(str(row.get("model_name", "")), row_style),
                Paragraph(f"{float(row.get('accuracy', 0)):.4f}", row_center),
                Paragraph(f"{float(row.get('macro_f1', 0)):.4f}", row_center),
                Paragraph(f"{float(row.get('ece', 0)):.4f}", row_center),
                Paragraph(f"{float(row.get('coverage', 1.0)):.1%}", row_center),
                Paragraph(f"{float(row.get('selective_accuracy', row.get('accuracy', 0))):.4f}", row_center),
            ])

        comp_t = Table(comp_rows, colWidths=[175, 70, 70, 65, 75, 85])
        comp_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 2.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
        if cage_row_idx:
            comp_style.extend([
                ("BACKGROUND", (0, cage_row_idx), (-1, cage_row_idx), colors.HexColor("#eff6ff")),
                ("LINEABOVE", (0, cage_row_idx), (-1, cage_row_idx), 1.0, colors.HexColor("#3b82f6")),
                ("LINEBELOW", (0, cage_row_idx), (-1, cage_row_idx), 1.0, colors.HexColor("#3b82f6")),
            ])
        comp_t.setStyle(TableStyle(comp_style))
        story.append(comp_t)
        story.append(Spacer(1, 5))

    # 10. Section 7: Embedded Diagnostic Visualizations
    if figures:
        plot_flowables = []
        # Target compact height: 1.65 inch if 4 figures, 2.0 inch if 2 figures
        img_h = 1.62 * inch if len(figures) >= 3 else 1.95 * inch
        img_w = 3.45 * inch

        # Prioritize key figures in optimal visual pairing order
        fig_order = ["meter", "radar", "reliability", "risk_coverage"]
        ordered_keys = [k for k in fig_order if k in figures] + [k for k in figures if k not in fig_order]

        for name in ordered_keys:
            fig_path = figures[name]
            if fig_path and Path(fig_path).exists():
                try:
                    img = Image(str(fig_path), width=img_w, height=img_h)
                    plot_flowables.append(img)
                except Exception as e:
                    logger.warning(f"Could not load figure {fig_path}: {e}")

        if plot_flowables:
            story.append(Paragraph("7. Research Calibration & Disagreement Visualizations", h2_style))
            # Pair plots side by side
            for i in range(0, len(plot_flowables), 2):
                pair = plot_flowables[i:i+2]
                if len(pair) == 2:
                    t = Table([[pair[0], pair[1]]], colWidths=[270, 270])
                else:
                    t = Table([[pair[0]]], colWidths=[540])
                t.setStyle(TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 1),
                ]))
                story.append(t)
                story.append(Spacer(1, 3))

    # Build PDF with two-pass NumberedCanvas for 'Page X of Y'
    doc.build(story, canvasmaker=NumberedCanvas)
    logger.info(f"PDF audit report successfully generated at {output_path}")
    return output_path
