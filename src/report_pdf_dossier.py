"""
CAGE-Fake: Per-Article Multi-Page PDF Dossier Generator
Generates individual phase/expert PDFs (00_*.pdf to 10_*.pdf) under outputs/reports/<timestamp>/
and compiles them into a merged full_report.pdf using pypdf.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import time
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
    HRFlowable,
)
from reportlab.pdfgen import canvas
from pypdf import PdfWriter

from src.utils import setup_logger

logger = setup_logger("report_dossier")


def _create_single_page_pdf(path: Path, title: str, subtitle: str, content_flowables: List[Any]):
    """Helper to build a clean single-page phase PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    t_style = ParagraphStyle(
        "TStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
    )
    sub_style = ParagraphStyle(
        "SubStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#475569"),
    )

    story = [
        Paragraph(title, t_style),
        Paragraph(subtitle, sub_style),
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=8),
    ]
    story.extend(content_flowables)
    doc.build(story)


class ArticleReportDossier:
    """
    Generates the modular 11-part diagnostic audit dossier and merges it into full_report.pdf.
    """

    def __init__(self, base_reports_dir: Path):
        self.base_reports_dir = Path(base_reports_dir)

    def generate_dossier(
        self,
        timestamp_str: str,
        article_meta: Dict[str, Any],
        decision_res: Dict[str, Any],
        multi_view_res: Dict[str, Any],
        evidence_res: Dict[str, Any],
        ood_res: Dict[str, Any],
        credibility_info: Dict[str, Any],
        figures: Optional[Dict[str, Path]] = None,
    ) -> Path:
        report_dir = self.base_reports_dir / timestamp_str
        report_dir.mkdir(parents=True, exist_ok=True)

        styles = getSampleStyleSheet()
        body_style = ParagraphStyle("BStyle", parent=styles["Normal"], fontName="Helvetica", fontSize=8, leading=11, textColor=colors.HexColor("#334155"))
        cell_style = ParagraphStyle("CStyle", parent=styles["Normal"], fontName="Helvetica", fontSize=7.5, leading=10, textColor=colors.HexColor("#334155"))
        cell_bold = ParagraphStyle("CBStyle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=7.5, leading=10, textColor=colors.HexColor("#0f172a"))

        generated_files = []

        # 00_verdict_summary.pdf
        f0 = report_dir / "00_verdict_summary.pdf"
        dec = decision_res["decision"]
        v_col = "#047857" if dec == "REAL" else ("#b91c1c" if dec == "FAKE" else "#b45309")
        v_bg = "#ecfdf5" if dec == "REAL" else ("#fef2f2" if dec == "FAKE" else "#fffbeb")
        v_bd = "#059669" if dec == "REAL" else ("#dc2626" if dec == "FAKE" else "#d97706")

        v_text = (
            f"<font size=16 color='{v_col}'><b>VERDICT: {dec}</b></font><br/>"
            f"<b>Confidence:</b> {decision_res['confidence']:.1%} | <b>Gating Trigger:</b> {decision_res['trigger']}<br/>"
            f"<b>Policy Category:</b> {decision_res['qualifier']}<br/>"
            f"<b>Explanation:</b> {decision_res['explanation']}"
        )
        if decision_res.get("style_evidence_conflict"):
            v_text += f"<br/><br/><font color='#dc2626'><b>{decision_res['conflict_explanation']}</b></font>"

        v_table = Table([[Paragraph(v_text, body_style)]], colWidths=[540])
        v_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(v_bg)),
            ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor(v_bd)),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        meta_p = Paragraph(
            f"<b>Title:</b> {article_meta.get('title')}<br/>"
            f"<b>Source URL:</b> {article_meta.get('url') or 'Manual Input'}<br/>"
            f"<b>Word Count:</b> {article_meta.get('word_count', 0):,} words | <b>Date:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}",
            body_style
        )
        _create_single_page_pdf(f0, "Phase 0: Triage Verdict Summary", "CAGE-Fake Final Decision & Policy Arbitration", [meta_p, Spacer(1, 8), v_table])
        generated_files.append(f0)

        # 01_claims_evidence.pdf
        f1 = report_dir / "01_claims_evidence.pdf"
        claims = evidence_res.get("claim_verifications", [])
        c_rows = [[Paragraph("<b>Claim</b>", cell_bold), Paragraph("<b>Status</b>", cell_bold), Paragraph("<b>Source</b>", cell_bold), Paragraph("<b>Evidence Context</b>", cell_bold)]]
        for c in claims[:6]:
            st = c.get("status", "Unverified")
            s_col = "#16a34a" if st == "Supported" else ("#dc2626" if st == "Contradicted" else "#d97706")
            c_rows.append([
                Paragraph(c.get("claim", "")[:120], cell_style),
                Paragraph(f"<font color='{s_col}'><b>{st}</b></font>", cell_style),
                Paragraph(c.get("source", ""), cell_style),
                Paragraph(c.get("evidence_text", "")[:150], cell_style),
            ])
        c_tab = Table(c_rows, colWidths=[170, 75, 125, 170])
        c_tab.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        _create_single_page_pdf(f1, "Phase 1: Claim-Level Evidence Grounding", "Extracted Factual Assertions & Authoritative Corroboration", [c_tab])
        generated_files.append(f1)

        # 02_entity_number_match.pdf
        f2 = report_dir / "02_entity_number_match.pdf"
        match_p = Paragraph(
            f"<b>Numeric Conflicts Detected:</b> {'Yes' if evidence_res.get('has_numeric_conflict') else 'None'}<br/>"
            f"<b>Corroborated Trusted Domains:</b> {', '.join(evidence_res.get('unique_trusted_domains', [])) or 'None'}<br/>"
            f"<b>Cross-Consistency Summary:</b> {evidence_res.get('summary')}",
            body_style
        )
        _create_single_page_pdf(f2, "Phase 2: Entity & Numerical Consistency", "Symbolic Comparison of Cardinal Figures, Dates, and Named Entities", [match_p])
        generated_files.append(f2)

        # 03_expert_lexical.pdf
        f3 = report_dir / "03_expert_lexical.pdf"
        lex = multi_view_res.get("lexical_explanation", {})
        lex_p = Paragraph(f"<b>Model:</b> Logistic Regression on Raw Sparse TF-IDF (No SVD)<br/><b>P(FAKE):</b> {lex.get('p_fake', 0.5):.1%}<br/><b>Top Driving Tokens:</b><br/>" + "<br/>".join([f"• {w}" for w in lex.get("why", [])]), body_style)
        _create_single_page_pdf(f3, "Phase 3: Lexical Expert Diagnosis", "Sparse Vocabulary Distribution & Linear Feature Coefficients", [lex_p])
        generated_files.append(f3)

        # 04_expert_style.pdf
        f4 = report_dir / "04_expert_style.pdf"
        sty = multi_view_res.get("style_explanation", {})
        sty_p = Paragraph(f"<b>Model:</b> XGBoost on 19 Named Stylometric Features<br/><b>P(FAKE):</b> {sty.get('p_fake', 0.5):.1%}<br/><b>Top SHAP Impact Features:</b><br/>" + "<br/>".join([f"• {w}" for w in sty.get("why", [])]), body_style)
        _create_single_page_pdf(f4, "Phase 4: Style Expert Diagnosis", "Tree-Based Stylometric & Readability Analysis with SHAP TreeExplainer", [sty_p])
        generated_files.append(f4)

        # 05_expert_semantic.pdf
        f5 = report_dir / "05_expert_semantic.pdf"
        sem = multi_view_res.get("semantic_explanation", {})
        sem_p = Paragraph(f"<b>Model:</b> Fine-Tuned DistilRoBERTa / MuRIL Transformer<br/><b>P(FAKE):</b> {sem.get('p_fake', 0.5):.1%}<br/><b>Token Attributions (Integrated Gradients):</b><br/>" + "<br/>".join([f"• {w}" for w in sem.get("why", [])]), body_style)
        _create_single_page_pdf(f5, "Phase 5: Semantic Expert Diagnosis", "Deep Contextual Representation & Gradient-Based Token Highlights", [sem_p])
        generated_files.append(f5)

        # 06_expert_evidence.pdf
        f6 = report_dir / "06_expert_evidence.pdf"
        ev_p = Paragraph(
            f"<b>Total Claims Evaluated:</b> {evidence_res.get('total_claims', 0)}<br/>"
            f"<b>Supported Claims:</b> {evidence_res.get('supported_count', 0)} ({evidence_res.get('percent_supported', 0.0):.1%})<br/>"
            f"<b>Contradicted Claims:</b> {evidence_res.get('contradicted_count', 0)}<br/>"
            f"<b>Unverified Claims:</b> {evidence_res.get('unverified_count', 0)}<br/>"
            f"<b>Independent Trusted Domains:</b> {evidence_res.get('trusted_domains_count', 0)}",
            body_style
        )
        _create_single_page_pdf(f6, "Phase 6: Evidence Verification Metrics", "Authoritative Whitelist Verification & External Truth Grounding", [ev_p])
        generated_files.append(f6)

        # 07_consensus_disagreement.pdf
        f7 = report_dir / "07_consensus_disagreement.pdf"
        dis_p = Paragraph(
            f"<b>Multi-View Disagreement (delta_views = std):</b> {multi_view_res.get('view_disagreement', 0.0):.4f}<br/>"
            f"<b>Mean Multi-View P(FAKE):</b> {multi_view_res.get('mean_p_fake', 0.5):.1%}<br/>"
            f"<b>Lexical P(FAKE):</b> {multi_view_res.get('p_lexical', 0.5):.1%}<br/>"
            f"<b>Style P(FAKE):</b> {multi_view_res.get('p_style', 0.5):.1%}<br/>"
            f"<b>Semantic P(FAKE):</b> {multi_view_res.get('p_semantic', 0.5):.1%}",
            body_style
        )
        content_07 = [dis_p]
        if figures and figures.get("meter") and Path(figures["meter"]).exists():
            content_07.extend([Spacer(1, 8), Image(str(figures["meter"]), width=5.5 * 72, height=2.8 * 72)])
        _create_single_page_pdf(f7, "Phase 7: Multi-View Consensus Meter", "Inter-Expert Disagreement & Variance Decomposition", content_07)
        generated_files.append(f7)

        # 08_fingerprint.pdf
        f8 = report_dir / "08_fingerprint.pdf"
        fp_p = Paragraph(
            f"<b>Overall Credibility Index:</b> {credibility_info.get('overall_credibility_index', 0.0):.1f} / 100<br/>"
            f"<b>Credibility Tier:</b> {credibility_info.get('credibility_grade', 'Grade B')}",
            body_style
        )
        content_08 = [fp_p]
        if figures and figures.get("radar") and Path(figures["radar"]).exists():
            content_08.extend([Spacer(1, 8), Image(str(figures["radar"]), width=5.0 * 72, height=3.5 * 72)])
        _create_single_page_pdf(f8, "Phase 8: News Credibility Fingerprint", "Stylistic & Journalistic Norm Scorecard", content_08)
        generated_files.append(f8)

        # 09_gate_decision.pdf
        f9 = report_dir / "09_gate_decision.pdf"
        gate_p = Paragraph(
            f"<b>Tri-Trigger Gate Evaluation:</b><br/>"
            f"• Confidence Threshold tau: {decision_res.get('tau', 0.80):.1%}<br/>"
            f"• View Disagreement Threshold delta: {decision_res.get('delta', 0.20):.2f}<br/>"
            f"• Domain Shift Threshold rho: {decision_res.get('rho', 0.50):.2f}<br/>"
            f"• Trigger Result: <b>{decision_res.get('trigger')}</b><br/>"
            f"• Decision Output: <b>{decision_res.get('decision')}</b>",
            body_style
        )
        _create_single_page_pdf(f9, "Phase 9: Tri-Trigger Gate Arbitration", "Confidence (tau), Multi-View Disagreement (delta), and Domain Shift (rho)", [gate_p])
        generated_files.append(f9)

        # 10_ood_check.pdf
        f10 = report_dir / "10_ood_check.pdf"
        ood_p = Paragraph(
            f"<b>OOD Status:</b> {'OUT-OF-DISTRIBUTION' if ood_res.get('is_ood') else 'IN-DISTRIBUTION'}<br/>"
            f"<b>Composite OOD Score:</b> {ood_res.get('ood_score', 0.0):.4f} (Threshold rho = {ood_res.get('threshold_rho', 0.5):.4f})<br/>"
            f"<b>TF-IDF Vocabulary Coverage:</b> {ood_res.get('vocab_coverage', 1.0):.1%}<br/>"
            f"<b>SBERT Mahalanobis Distance:</b> {ood_res.get('mahalanobis_dist', 0.0):.2f}<br/>"
            f"<b>Diagnostic Note:</b> {ood_res.get('explanation')}",
            body_style
        )
        _create_single_page_pdf(f10, "Phase 10: Out-Of-Distribution Domain Shift Check", "Lexical Vocabulary Coverage & Semantic Mahalanobis Distance", [ood_p])
        generated_files.append(f10)

        # Merge all into full_report.pdf using pypdf
        merged_pdf = report_dir / "full_report.pdf"
        writer = PdfWriter()
        for pdf_file in generated_files:
            writer.append(str(pdf_file))
        with open(merged_pdf, "wb") as f_out:
            writer.write(f_out)

        logger.info(f"Successfully generated 11 phase PDFs and merged full_report.pdf at {merged_pdf}")
        return merged_pdf
