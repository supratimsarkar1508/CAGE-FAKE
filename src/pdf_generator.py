"""
CAGE-Fake: Publication-Quality PDF Report Generator
Generates comprehensive diagnostic audit reports using ReportLab.
Includes verdict, confidence, uncertainty gating, 5-model disagreement meter,
credibility fingerprint, counterfactual analysis, global benchmark comparison, and embedded plots.
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
)
from reportlab.lib.units import inch

from src.utils import setup_logger

logger = setup_logger("pdf_generator")


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
) -> Path:
    """
    Builds and saves a multi-section PDF audit report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e293b"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
    )
    h2_style = ParagraphStyle(
        "Heading2Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerCustom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#92400e"),
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
    )

    story = []

    # 1. Header
    story.append(Paragraph("CAGE-Fake: Article Trustworthiness Audit Report", title_style))
    story.append(Paragraph(
        "Calibration-Aware Gating and Evidence-Grounded Reasoning for Trustworthy Fake News Triage",
        subtitle_style
    ))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=8))

    # 2. Disclaimer Box (Prominent & Mandatory)
    disclaimer_text = (
        "<b>LEGAL & METHODOLOGICAL DISCLAIMER:</b> A trained fake-news classifier predicts linguistic, stylometric, "
        "and semantic patterns learned from benchmark datasets; it does <b>NOT</b> independently verify whether a news story "
        "is factually true. URL extraction, model pattern recognition, and external fact verification remain strictly separate processes. "
        "High uncertainty cases are flagged for professional human editorial review."
    )
    disclaimer_table = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[540])
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef3c7")),
        ("BORDER", (0, 0), (-1, -1), 1, colors.HexColor("#f59e0b")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(disclaimer_table)
    story.append(Spacer(1, 10))

    # 3. Article Metadata & Verdict Card
    decision = gate_result["decision"]
    conf = gate_result["confidence"]
    disagreement = gate_result["disagreement"]
    trigger = gate_result["trigger"]

    if decision == "REAL":
        v_bg = colors.HexColor("#dcfce7")
        v_border = colors.HexColor("#16a34a")
        v_color = "#166534"
    elif decision == "FAKE":
        v_bg = colors.HexColor("#fee2e2")
        v_border = colors.HexColor("#dc2626")
        v_color = "#991b1b"
    else:
        v_bg = colors.HexColor("#fef3c7")
        v_border = colors.HexColor("#d97706")
        v_color = "#92400e"

    title_text = article_meta.get("title", "Untitled Article")[:100]
    url_text = article_meta.get("url") or "Manual Text Submission"
    source_text = f"<b>Source / URL:</b> {url_text}<br/><b>Title:</b> {title_text}<br/><b>Word Count:</b> {article_meta.get('word_count', 0):,} words | <b>Date:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}"

    tau_val = gate_result.get("tau", 0.80)
    delta_val = gate_result.get("delta", 0.20)
    verdict_text = (
        f"<font size=16 color='{v_color}'><b>VERDICT: {decision}</b></font><br/>"
        f"<b>Calibrated Confidence:</b> {conf:.1%} (Threshold tau = {tau_val:.1%})<br/>"
        f"<b>Base Disagreement (std):</b> {disagreement:.3f} (Threshold delta = {delta_val:.2f})<br/>"
        f"<b>Gating Trigger:</b> {trigger.replace('_', ' ').title()}"
    )

    meta_table = Table(
        [[
            Paragraph(source_text, body_style),
            Paragraph(verdict_text, body_style)
        ]],
        colWidths=[330, 210]
    )
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#f8fafc")),
        ("BACKGROUND", (1, 0), (1, 0), v_bg),
        ("BOX", (0, 0), (0, 0), 1, colors.HexColor("#cbd5e1")),
        ("BOX", (1, 0), (1, 0), 1.5, v_border),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(meta_table)

    if article_meta.get("short_text_warning"):
        warn_p = Paragraph(f"<b>CAUTION: Short Text Length ({article_meta.get('word_count', 0)} words)</b><br/>{article_meta['short_text_warning']}", disclaimer_style)
        warn_table = Table([[warn_p]], colWidths=[540])
        warn_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
            ("BORDER", (0, 0), (-1, -1), 1, colors.HexColor("#f59e0b")),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(Spacer(1, 4))
        story.append(warn_table)

    story.append(Spacer(1, 10))

    # 4. Model Disagreement Meter (All 5 Models)
    story.append(Paragraph("1. Multi-Model Disagreement Analysis (5 Models)", h2_style))
    meter_header = [
        Paragraph("<b>Model Name</b>", table_cell_bold),
        Paragraph("<b>P(FAKE)</b>", table_cell_bold),
        Paragraph("<b>Model Vote</b>", table_cell_bold),
        Paragraph("<b>Confidence</b>", table_cell_bold),
    ]
    meter_rows = [meter_header]
    for m in disagreement_info["models"]:
        vote_color = "#dc2626" if m["vote"] == "FAKE" else "#16a34a"
        meter_rows.append([
            Paragraph(m["name"], table_cell_style),
            Paragraph(f"{m['prob_fake']:.1%}", table_cell_style),
            Paragraph(f"<font color='{vote_color}'><b>{m['vote']}</b></font>", table_cell_style),
            Paragraph(f"{m['confidence']:.1%}", table_cell_style),
        ])

    disagree_table = Table(meter_rows, colWidths=[150, 100, 100, 100])
    disagree_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))

    disagree_std_val = disagreement_info.get("gate_base_disagreement", disagreement_info.get("std_dev", 0.0))
    five_std_val = disagreement_info.get("five_model_std", disagreement_info.get("std_dev", 0.0))
    delta_val = disagreement_info.get("threshold_delta", 0.20)
    disagree_summary = (
        f"<b>Consensus Status:</b> {disagreement_info.get('consensus_level', 'Evaluated')}<br/>"
        f"<b>Base Disagreement (4-base std):</b> {disagree_std_val:.3f} (Delta = {delta_val:.2f})<br/>"
        f"<b>5-Model Consensus Spread:</b> {five_std_val:.3f}<br/>"
        f"<b>Vote Distribution:</b> {disagreement_info.get('fake_votes', 0)} FAKE vs {disagreement_info.get('real_votes', 0)} REAL<br/>"
        f"<b>Diagnostic:</b> {disagreement_info.get('status_desc', 'Evaluation completed.')}"
    )

    story.append(Table([[
        disagree_table,
        Paragraph(disagree_summary, body_style)
    ]], colWidths=[290, 250], style=[
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(Spacer(1, 10))

    # 5. News Credibility Fingerprint & Scorecard
    story.append(Paragraph("2. News Credibility Fingerprint (Linguistic & Stylistic Scorecard)", h2_style))
    fp_header = [
        Paragraph("<b>Credibility Dimension</b>", table_cell_bold),
        Paragraph("<b>Score (0-100)</b>", table_cell_bold),
        Paragraph("<b>Journalistic Assessment</b>", table_cell_bold),
    ]
    fp_rows = [fp_header]
    for dim, score in credibility_info["scores"].items():
        assessment = "Excellent" if score >= 80 else ("Adequate" if score >= 60 else "Deceptive / Alert")
        score_color = "#16a34a" if score >= 80 else ("#d97706" if score >= 60 else "#dc2626")
        fp_rows.append([
            Paragraph(dim, table_cell_style),
            Paragraph(f"<font color='{score_color}'><b>{score}/100</b></font>", table_cell_style),
            Paragraph(assessment, table_cell_style),
        ])

    fp_table = Table(fp_rows, colWidths=[170, 90, 130])
    fp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))

    fp_summary = (
        f"<b>Overall Credibility Index:</b> {credibility_info['overall_credibility_index']}/100<br/>"
        f"<b>Credibility Tier:</b> <b>{credibility_info['credibility_grade']}</b><br/><br/>"
        f"Scores are normalized against verified newsroom style guides: high attribution, balanced syntax, and low sensationalism maximize the index."
    )

    story.append(Table([[
        fp_table,
        Paragraph(fp_summary, body_style)
    ]], colWidths=[290, 250], style=[
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(Spacer(1, 10))

    # 6. 'Why this prediction?' & Counterfactual Analysis
    story.append(Paragraph("3. Transparent Reasoning & Counterfactual Sensitivity", h2_style))
    formatted_why = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", why_explanation).replace("\n", "<br/>")
    story.append(Paragraph(f"<b>Diagnostic Explanation:</b><br/>{formatted_why}", body_style))
    story.append(Spacer(1, 6))

    shift_val = counterfactual_info.get("signed_shift_pct", counterfactual_info.get("prob_shift", 0.0) * 100.0)
    cf_summary = (
        f"<b>Counterfactual Simulation:</b> {counterfactual_info['summary']}<br/>"
        f"<b>Original P(FAKE):</b> {counterfactual_info['original_prob']:.1%} | "
        f"<b>Transformed P(FAKE):</b> {counterfactual_info['simulated_prob']:.1%} "
        f"<b>(Signed Shift: {shift_val:+.1f} percentage points)</b>"
    )
    story.append(Paragraph(cf_summary, body_style))
    story.append(Spacer(1, 10))

    # 7. Local Feature Attributions (Top FAKE vs REAL Driving Markers)
    story.append(Paragraph("4. Local Signal Attributions (Meta-Ensemble Impacts)", h2_style))
    top_fake = local_features.get("fake_signals") or local_features.get("top_fake_driving", [])
    top_real = local_features.get("real_signals") or local_features.get("top_real_driving", [])
    top_fake = top_fake[:4]
    top_real = top_real[:4]

    feat_header = [
        Paragraph("<b>Top FAKE-Driving Indicators</b>", table_cell_bold),
        Paragraph("<b>Weight / Impact</b>", table_cell_bold),
        Paragraph("<b>Top REAL-Driving Anchors</b>", table_cell_bold),
        Paragraph("<b>Weight / Impact</b>", table_cell_bold),
    ]
    feat_rows = [feat_header]
    max_len = max(len(top_fake), len(top_real))
    for i in range(max_len):
        f_fake = (top_fake[i].get("name") or top_fake[i].get("feature", "-")) if i < len(top_fake) else "-"
        i_fake = f"{top_fake[i]['impact']:+.3f}" if i < len(top_fake) else "-"
        f_real = (top_real[i].get("name") or top_real[i].get("feature", "-")) if i < len(top_real) else "-"
        i_real = f"{top_real[i]['impact']:+.3f}" if i < len(top_real) else "-"
        feat_rows.append([
            Paragraph(f_fake, table_cell_style),
            Paragraph(f"<font color='#dc2626'>{i_fake}</font>", table_cell_style),
            Paragraph(f_real, table_cell_style),
            Paragraph(f"<font color='#16a34a'>{i_real}</font>", table_cell_style),
        ])

    feat_table = Table(feat_rows, colWidths=[160, 90, 180, 90])
    feat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 10))

    # 8. Global Benchmark Comparison Table (if provided)
    if comparison_table:
        story.append(Paragraph("5. Empirical Benchmark Model Comparison (Held-out Test Split)", h2_style))
        comp_header = [
            Paragraph("<b>Model</b>", table_cell_bold),
            Paragraph("<b>Accuracy</b>", table_cell_bold),
            Paragraph("<b>Macro-F1</b>", table_cell_bold),
            Paragraph("<b>ECE</b>", table_cell_bold),
            Paragraph("<b>Coverage</b>", table_cell_bold),
            Paragraph("<b>Selective Acc</b>", table_cell_bold),
        ]
        comp_rows = [comp_header]
        for row in comparison_table:
            is_cage = "CAGE" in str(row.get("model_name", ""))
            row_style = table_cell_bold if is_cage else table_cell_style
            comp_rows.append([
                Paragraph(str(row.get("model_name", "")), row_style),
                Paragraph(f"{float(row.get('accuracy', 0)):.4f}", row_style),
                Paragraph(f"{float(row.get('macro_f1', 0)):.4f}", row_style),
                Paragraph(f"{float(row.get('ece', 0)):.4f}", row_style),
                Paragraph(f"{float(row.get('coverage', 1.0)):.1%}", row_style),
                Paragraph(f"{float(row.get('selective_accuracy', row.get('accuracy', 0))):.4f}", row_style),
            ])

        comp_t = Table(comp_rows, colWidths=[180, 70, 70, 60, 70, 80])
        comp_t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(comp_t)
        story.append(Spacer(1, 10))

    # 9. Embedded Diagnostic Plots (if figures exist)
    if figures:
        plot_flowables = []
        for name, fig_path in figures.items():
            if fig_path and Path(fig_path).exists():
                try:
                    img = Image(str(fig_path), width=3.5 * inch, height=2.3 * inch)
                    plot_flowables.append(img)
                except Exception as e:
                    logger.warning(f"Could not load figure {fig_path}: {e}")

        if plot_flowables:
            story.append(Paragraph("6. Research Calibration & Disagreement Visualizations", h2_style))
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
                    ("PADDING", (0, 0), (-1, -1), 2),
                ]))
                story.append(t)
                story.append(Spacer(1, 6))

    # Build PDF
    doc.build(story)
    logger.info(f"PDF audit report successfully generated at {output_path}")
    return output_path
