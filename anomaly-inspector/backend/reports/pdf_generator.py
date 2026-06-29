import io
from datetime import datetime, timezone
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_report(
    start_date: datetime,
    end_date: datetime,
    total: int,
    ok_count: int,
    not_ok_count: int,
    pass_rate: float,
    sku_names: list[str],
    threshold_min: float,
    threshold_max: float,
    model_version: str,
    app_version: str,
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=18, spaceAfter=6
    )
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], fontSize=12, spaceAfter=4, spaceBefore=8
    )
    normal = styles["Normal"]

    story = []

    # ── Header ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Anomaly Detection Inspection Report", title_style))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(f"Software Version: {app_version}", normal))
    story.append(Paragraph(f"Model Version: {model_version}", normal))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(f"Generated: {generated}", normal))
    story.append(Spacer(1, 6 * mm))

    # ── Date Range ──────────────────────────────────────────────────────────
    story.append(Paragraph("Date Range", heading_style))
    story.append(Paragraph(f"From: {start_date.strftime('%Y-%m-%d')}", normal))
    story.append(Paragraph(f"To:   {end_date.strftime('%Y-%m-%d')}", normal))
    story.append(Spacer(1, 6 * mm))

    # ── SKU ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("SKU", heading_style))
    sku_text = ", ".join(sorted(set(sku_names))) if sku_names else "N/A"
    story.append(Paragraph(sku_text, normal))
    story.append(Spacer(1, 6 * mm))

    # ── Summary Statistics ──────────────────────────────────────────────────
    story.append(Paragraph("Summary Statistics", heading_style))
    data = [
        ["Metric", "Value"],
        ["Total Inspected", str(total)],
        ["OK", str(ok_count)],
        ["NOT OK", str(not_ok_count)],
        ["Pass Rate", f"{pass_rate:.1f}%"],
    ]
    table = Table(data, colWidths=[85 * mm, 85 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4f8"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 6 * mm))

    # ── Threshold ───────────────────────────────────────────────────────────
    story.append(Paragraph("Threshold Settings", heading_style))
    if threshold_min == threshold_max:
        threshold_text = f"Threshold: {threshold_min:.2f}"
    else:
        threshold_text = f"Threshold range: {threshold_min:.2f} – {threshold_max:.2f}"
    story.append(Paragraph(threshold_text, normal))

    doc.build(story)
    return buffer.getvalue()
