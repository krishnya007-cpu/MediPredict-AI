# ============================================================
# utils/pdf_report.py — MEDIPREDICT AI
# Generates a professional one-click PDF prediction report
# Usage: from utils.pdf_report import generate_report
# ============================================================

import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String, Circle
from reportlab.graphics import renderPDF
from reportlab.graphics.charts.barcharts import VerticalBarChart


# ── Brand colours ─────────────────────────────────────────
BLUE        = colors.HexColor("#2563EB")
PURPLE      = colors.HexColor("#7C3AED")
DARK        = colors.HexColor("#0F172A")
SLATE       = colors.HexColor("#1E293B")
MUTED       = colors.HexColor("#64748B")
LIGHT_BG    = colors.HexColor("#F0F4FF")
BORDER      = colors.HexColor("#E2E8F0")
WHITE       = colors.white

RISK_COLORS = {
    "Low":    colors.HexColor("#22C55E"),
    "Medium": colors.HexColor("#F59E0B"),
    "High":   colors.HexColor("#EF4444"),
}
FRAUD_COLORS = {
    "Legitimate": colors.HexColor("#22C55E"),
    "Suspicious": colors.HexColor("#F59E0B"),
    "Fraudulent": colors.HexColor("#EF4444"),
}


# ── Style helpers ──────────────────────────────────────────
def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title",
            fontName="Helvetica-Bold", fontSize=22,
            textColor=WHITE, alignment=TA_LEFT, leading=28),
        "subtitle": ParagraphStyle("subtitle",
            fontName="Helvetica", fontSize=11,
            textColor=colors.HexColor("#CBD5E1"),
            alignment=TA_LEFT, leading=16),
        "section": ParagraphStyle("section",
            fontName="Helvetica-Bold", fontSize=13,
            textColor=DARK, spaceAfter=6, leading=18),
        "label": ParagraphStyle("label",
            fontName="Helvetica-Bold", fontSize=9,
            textColor=MUTED, leading=12),
        "value": ParagraphStyle("value",
            fontName="Helvetica", fontSize=10,
            textColor=SLATE, leading=14),
        "normal": ParagraphStyle("normal",
            fontName="Helvetica", fontSize=9,
            textColor=MUTED, leading=13),
        "badge_text": ParagraphStyle("badge_text",
            fontName="Helvetica-Bold", fontSize=14,
            textColor=WHITE, alignment=TA_CENTER, leading=20),
        "footer": ParagraphStyle("footer",
            fontName="Helvetica", fontSize=8,
            textColor=MUTED, alignment=TA_CENTER),
        "amount": ParagraphStyle("amount",
            fontName="Helvetica-Bold", fontSize=20,
            textColor=BLUE, alignment=TA_CENTER, leading=26),
        "disclaimer": ParagraphStyle("disclaimer",
            fontName="Helvetica-Oblique", fontSize=8,
            textColor=MUTED, leading=11),
    }


def _header_block(story, S, patient_info: dict, report_id: str):
    """Dark gradient header with patient name and report metadata."""

    # Header table: left = titles, right = report meta
    now = datetime.now().strftime("%d %b %Y, %I:%M %p")
    name = patient_info.get("name", "Patient")

    header_data = [[
        Paragraph(f"MEDIPREDICT AI", S["title"]),
        Paragraph(f"Report ID: {report_id}<br/>Generated: {now}", S["subtitle"]),
    ]]
    t = Table(header_data, colWidths=[110*mm, 70*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), DARK),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8*mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8*mm),
        ("TOPPADDING",   (0, 0), (-1, -1), 7*mm),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 7*mm),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",        (1, 0), (1, 0),   "RIGHT"),
    ]))
    story.append(t)
    story.append(Spacer(1, 5*mm))

    # Patient name strip
    name_row = [[Paragraph(f"Patient: {name}", ParagraphStyle("pn",
        fontName="Helvetica-Bold", fontSize=11,
        textColor=SLATE, leading=16))]]
    nt = Table(name_row, colWidths=[180*mm])
    nt.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), LIGHT_BG),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6*mm),
        ("TOPPADDING",   (0, 0), (-1, -1), 3*mm),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3*mm),
        ("ROUNDEDCORNERS", [3]),
    ]))
    story.append(nt)
    story.append(Spacer(1, 6*mm))


def _section_header(story, S, title: str):
    story.append(HRFlowable(width="100%", thickness=1,
                             color=BORDER, spaceAfter=4))
    story.append(Paragraph(title, S["section"]))


def _kv_table(story, rows: list[tuple], col_w=(55*mm, 85*mm)):
    """Two-column key-value info grid."""
    data = [[
        Paragraph(k, ParagraphStyle("k", fontName="Helvetica-Bold",
                  fontSize=9, textColor=MUTED, leading=13)),
        Paragraph(str(v), ParagraphStyle("v", fontName="Helvetica",
                  fontSize=10, textColor=SLATE, leading=14)),
    ] for k, v in rows]
    t = Table(data, colWidths=list(col_w))
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), WHITE),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [WHITE, LIGHT_BG]),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4*mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4*mm),
        ("TOPPADDING",   (0, 0), (-1, -1), 2.5*mm),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2.5*mm),
        ("BOX",          (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID",    (0, 0), (-1, -1), 0.5, BORDER),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(t)
    story.append(Spacer(1, 5*mm))


def _badge(label: str, color) -> Table:
    """Coloured pill badge for risk / fraud labels."""
    data = [[Paragraph(label, ParagraphStyle("bl",
        fontName="Helvetica-Bold", fontSize=13,
        textColor=WHITE, alignment=TA_CENTER, leading=18))]]
    t = Table(data, colWidths=[45*mm], rowHeights=[10*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), color),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ROUNDEDCORNERS", [5]),
    ]))
    return t


def _metric_cards(story, metrics: list[dict]):
    """
    Horizontal row of metric cards.
    Each dict: {label, value, sub}
    """
    card_data = []
    for m in metrics:
        cell = [
            Paragraph(m["label"], ParagraphStyle("ml",
                fontName="Helvetica-Bold", fontSize=8,
                textColor=MUTED, alignment=TA_CENTER, leading=11)),
            Paragraph(str(m["value"]), ParagraphStyle("mv",
                fontName="Helvetica-Bold", fontSize=16,
                textColor=DARK, alignment=TA_CENTER, leading=20)),
            Paragraph(m.get("sub", ""), ParagraphStyle("ms",
                fontName="Helvetica", fontSize=8,
                textColor=MUTED, alignment=TA_CENTER, leading=11)),
        ]
        card_data.append(cell)

    n   = len(metrics)
    w   = 180 / n
    row = [[c[0] for c in card_data],
           [c[1] for c in card_data],
           [c[2] for c in card_data]]
    t = Table(row, colWidths=[w*mm]*n)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), WHITE),
        ("BOX",          (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID",    (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",   (0, 0), (-1, -1), 3*mm),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3*mm),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 5*mm))


# ── Main generator ─────────────────────────────────────────
def generate_report(
    patient_info: dict,
    risk_result:     dict | None = None,
    premium_result:  dict | None = None,
    fraud_result:    dict | None = None,
    claim_result:    dict | None = None,
) -> bytes:
    """
    Build and return a MEDIPREDICT AI PDF report as bytes.

    Parameters
    ----------
    patient_info : dict
        Keys: name, age, gender, region, marital_status, num_dependents,
              bmi, smoker, chronic_disease, prev_hospitalizations,
              num_medications, exercise_frequency, coverage_type,
              policy_duration, deductible, num_prev_claims,
              annual_income, credit_score

    risk_result : dict | None
        Keys: label ('Low'|'Medium'|'High'), confidence (0-100 float),
              probabilities ({'Low':…, 'Medium':…, 'High':…})

    premium_result : dict | None
        Keys: amount (float), monthly (float), range_low (float), range_high (float)

    fraud_result : dict | None
        Keys: label ('Legitimate'|'Suspicious'|'Fraudulent'),
              probability (0-100 float), flags (list[str])

    claim_result : dict | None
        Keys: amount (float), range_low (float), range_high (float),
              category ('Low'|'Medium'|'High')

    Returns
    -------
    bytes  — PDF binary (pass directly to st.download_button)
    """
    import random, string
    report_id = "MP-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=12*mm,  bottomMargin=15*mm,
    )

    S     = _styles()
    story = []

    # ── Header ────────────────────────────────────────────
    _header_block(story, S, patient_info, report_id)

    # ── Patient Profile ───────────────────────────────────
    _section_header(story, S, "Patient Profile")
    _kv_table(story, [
        ("Full Name",           patient_info.get("name", "—")),
        ("Age",                 f"{patient_info.get('age', '—')} years"),
        ("Gender",              patient_info.get("gender", "—")),
        ("Region",              patient_info.get("region", "—")),
        ("Marital Status",      patient_info.get("marital_status", "—")),
        ("Dependents",          patient_info.get("num_dependents", "—")),
        ("BMI",                 f"{patient_info.get('bmi', '—'):.1f}" if isinstance(patient_info.get('bmi'), float) else patient_info.get('bmi','—')),
        ("Smoker",              patient_info.get("smoker", "—")),
        ("Chronic Diseases",    patient_info.get("chronic_disease", "—")),
        ("Prior Hospitalisations", patient_info.get("prev_hospitalizations", "—")),
        ("Medications",         patient_info.get("num_medications", "—")),
        ("Exercise Frequency",  patient_info.get("exercise_frequency", "—")),
        ("Coverage Type",       patient_info.get("coverage_type", "—")),
        ("Policy Duration",     f"{patient_info.get('policy_duration','—')} yrs"),
        ("Deductible",          f"${patient_info.get('deductible', '—'):,}"),
        ("Prior Claims",        patient_info.get("num_prev_claims", "—")),
        ("Annual Income",       f"${patient_info.get('annual_income', 0):,}"),
        ("Credit Score",        patient_info.get("credit_score", "—")),
    ])

    # ── Risk Prediction ───────────────────────────────────
    if risk_result:
        _section_header(story, S, "Health Risk Assessment")
        label = risk_result.get("label", "Unknown")
        conf  = risk_result.get("confidence", 0)
        probs = risk_result.get("probabilities", {})
        color = RISK_COLORS.get(label, MUTED)

        badge_cell  = _badge(f"{label} Risk", color)
        conf_cell   = Paragraph(
            f"Model Confidence: <b>{conf:.1f}%</b>",
            ParagraphStyle("cc", fontName="Helvetica", fontSize=10,
                           textColor=SLATE, alignment=TA_CENTER, leading=14))

        row = [[badge_cell, conf_cell]]
        t2  = Table(row, colWidths=[55*mm, 125*mm])
        t2.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("ALIGN",  (1,0), (1,0),   "LEFT"),
            ("LEFTPADDING",  (0,0), (-1,-1), 0),
            ("RIGHTPADDING", (0,0), (-1,-1), 0),
            ("TOPPADDING",   (0,0), (-1,-1), 0),
            ("BOTTOMPADDING",(0,0), (-1,-1), 3*mm),
        ]))
        story.append(t2)

        if probs:
            prob_rows = [["Risk Level", "Probability"]] + [
                [k, f"{v:.1f}%"] for k, v in probs.items()
            ]
            pt = Table(prob_rows, colWidths=[60*mm, 40*mm])
            pt.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0), SLATE),
                ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
                ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",     (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LIGHT_BG]),
                ("BOX",          (0, 0), (-1, -1), 0.5, BORDER),
                ("INNERGRID",    (0, 0), (-1, -1), 0.5, BORDER),
                ("TOPPADDING",   (0, 0), (-1, -1), 2.5*mm),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 2.5*mm),
                ("LEFTPADDING",  (0, 0), (-1, -1), 4*mm),
            ]))
            story.append(pt)
            story.append(Spacer(1, 5*mm))

    # ── Premium Prediction ────────────────────────────────
    if premium_result:
        _section_header(story, S, "Estimated Insurance Premium")
        amt     = premium_result.get("amount", 0)
        monthly = premium_result.get("monthly", amt / 12)
        lo      = premium_result.get("range_low",  amt * 0.9)
        hi      = premium_result.get("range_high", amt * 1.1)

        _metric_cards(story, [
            {"label": "ANNUAL PREMIUM",  "value": f"${amt:,.0f}",     "sub": "per year"},
            {"label": "MONTHLY",         "value": f"${monthly:,.0f}", "sub": "per month"},
            {"label": "EXPECTED RANGE",  "value": f"${lo:,.0f} – ${hi:,.0f}", "sub": "95% confidence"},
        ])

    # ── Fraud Assessment ──────────────────────────────────
    if fraud_result:
        _section_header(story, S, "Fraud Risk Assessment")
        label = fraud_result.get("label", "Legitimate")
        prob  = fraud_result.get("probability", 0)
        flags = fraud_result.get("flags", [])
        color = FRAUD_COLORS.get(label, RISK_COLORS["Low"])

        badge_cell = _badge(label, color)
        prob_cell  = Paragraph(
            f"Fraud Probability: <b>{prob:.1f}%</b>",
            ParagraphStyle("fp", fontName="Helvetica", fontSize=10,
                           textColor=SLATE, alignment=TA_LEFT, leading=14))
        row = [[badge_cell, prob_cell]]
        t3  = Table(row, colWidths=[55*mm, 125*mm])
        t3.setStyle(TableStyle([
            ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
            ("LEFTPADDING",  (0,0),(-1,-1), 0),
            ("RIGHTPADDING", (0,0),(-1,-1), 0),
            ("TOPPADDING",   (0,0),(-1,-1), 0),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3*mm),
        ]))
        story.append(t3)

        if flags:
            story.append(Paragraph("Risk Flags Detected:", ParagraphStyle("fl",
                fontName="Helvetica-Bold", fontSize=9, textColor=MUTED, leading=13)))
            for f in flags:
                story.append(Paragraph(f"• {f}", ParagraphStyle("fi",
                    fontName="Helvetica", fontSize=9,
                    textColor=colors.HexColor("#EF4444"), leading=13,
                    leftIndent=4*mm)))
            story.append(Spacer(1, 4*mm))

    # ── Claim Prediction ──────────────────────────────────
    if claim_result:
        _section_header(story, S, "Predicted Claim Amount")
        amt = claim_result.get("amount", 0)
        lo  = claim_result.get("range_low",  amt * 0.8)
        hi  = claim_result.get("range_high", amt * 1.2)
        cat = claim_result.get("category", "Medium")

        _metric_cards(story, [
            {"label": "PREDICTED CLAIM", "value": f"${amt:,.0f}",           "sub": "expected value"},
            {"label": "CLAIM RANGE",     "value": f"${lo:,.0f}–${hi:,.0f}", "sub": "80% confidence"},
            {"label": "CLAIM CATEGORY",  "value": cat,                       "sub": "severity"},
        ])

    # ── Summary table ─────────────────────────────────────
    _section_header(story, S, "Prediction Summary")
    summary_rows = [["Module", "Result", "Status"]]
    checks = [
        ("Health Risk",     risk_result,    risk_result.get("label","—") if risk_result else "—"),
        ("Premium Estimate",premium_result, f"${premium_result.get('amount',0):,.0f}/yr" if premium_result else "—"),
        ("Fraud Check",     fraud_result,   fraud_result.get("label","—") if fraud_result else "—"),
        ("Claim Prediction",claim_result,   f"${claim_result.get('amount',0):,.0f}" if claim_result else "—"),
    ]
    for module, res, val in checks:
        status = "✓ Complete" if res else "— Skipped"
        summary_rows.append([module, val, status])

    st_tbl = Table(summary_rows, colWidths=[60*mm, 70*mm, 50*mm])
    st_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), SLATE),
        ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LIGHT_BG]),
        ("BOX",          (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID",    (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",   (0, 0), (-1, -1), 2.5*mm),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2.5*mm),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4*mm),
    ]))
    story.append(st_tbl)
    story.append(Spacer(1, 6*mm))

    # ── Disclaimer ────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceBefore=4))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI system for informational purposes only. "
        "It does not constitute medical advice, diagnosis, or treatment. "
        "All predictions carry inherent uncertainty and must be reviewed by a qualified professional "
        "before any clinical or financial decisions are made. "
        f"MEDIPREDICT AI · Report {report_id} · {datetime.now().strftime('%Y')}",
        S["disclaimer"]
    ))

    doc.build(story)
    return buf.getvalue()
