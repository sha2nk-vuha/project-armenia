import io
import json
from collections import defaultdict
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from inference.verdict import NOT_OK, OK, cascade_nok_reason

# Feature label + threshold-label lookup, mirroring config.FEATURES. Imported
# lazily so pdf_generator stays import-safe in tests without the app config.
try:
    from config import FEATURES as _FEATURE_CATALOG
except (ImportError, ModuleNotFoundError):  # pragma: no cover - config always present in-app
    _FEATURE_CATALOG = {}


def _feature_label(feature: str) -> str:
    """Human label for a Feature.

    Falls back to a title-cased form of the id when the Feature is unknown.

    Args:
        feature: Feature name from config.FEATURES.

    Returns:
        Operator-facing label.
    """
    entry = _FEATURE_CATALOG.get(feature, {})
    return entry.get("label", feature.replace("_", " ").title())


def _parse_metrics(raw: str | None) -> dict:
    """Best-effort parse of the metrics JSON column.

    Args:
        raw: Raw JSON string from the inspections.metrics column, a parsed
            dict, or None (legacy rows).

    Returns:
        Parsed dict when it is a dict; {} otherwise. Never raises.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _fmt_score(score) -> str:
    """Format a score for the report table, tolerating None/truncation.

    Args:
        score: The score value, possibly None or a non-numeric placeholder.

    Returns:
        Two-decimal string, or "—" when not formattable.
    """
    if score is None:
        return "—"
    try:
        return f"{float(score):.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_threshold(value) -> str:
    """Format a threshold for the report table.

    Args:
        value: Threshold value, possibly missing or non-numeric.

    Returns:
        Two-decimal string, or "—" when not formattable.
    """
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_threshold_range(values) -> str:
    """Format a per-group threshold as a single value or min–max range.

    Args:
        values: Thresholds of inspections belonging to one group.

    Returns:
        "—" when none are numeric, one value when all equal, otherwise
        "lo–hi".
    """
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return "—"
    lo, hi = min(nums), max(nums)
    if lo == hi:
        return f"{lo:.2f}"
    return f"{lo:.2f}–{hi:.2f}"


def generate_report(
    start_date: datetime,
    end_date: datetime,
    inspections: list[dict],
    app_version: str,
    customer_name: str = "",
) -> bytes:
    """Build a self-contained inspection report PDF.

    ``inspections`` is a list of plain dicts, each carrying the columns needed
    for reporting: ``timestamp`` (datetime), ``sku_name``, ``feature``, ``score``
    (float|None), ``threshold`` (float), ``verdict`` (str), ``model_version``
    (str), ``decision_rule`` (str|None), and ``metrics`` (raw JSON text, a parsed
    dict, or None). Grouping, stats, and layout all happen here so the caller
    (main.py) stays thin.

    Args:
        start_date: Inclusive start of the requested date range.
        end_date: Inclusive end of the requested date range.
        inspections: Flat inspection row dicts (from _report_row).
        app_version: Version string stamped into the report header.
        customer_name: Optional customer to title the report for.

    Returns:
        Complete PDF file contents as bytes (consumed by the endpoint's
        Response).
    """
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
    customer_style = ParagraphStyle(
        "CustomerHeading",
        parent=styles["Heading1"],
        fontSize=15,
        spaceAfter=2,
        spaceBefore=2,
        textColor=colors.HexColor("#1e3a5f"),
    )
    normal = styles["Normal"]
    cell_style = ParagraphStyle(
        "CellText", parent=normal, fontSize=7.5, leading=9
    )
    cell_bold = ParagraphStyle(
        "CellBold", parent=cell_style, fontName="Helvetica-Bold"
    )
    stage_cell_style = ParagraphStyle(
        "StageCell", parent=normal, fontSize=7, leading=8.5, textColor=colors.HexColor("#475569")
    )

    def _cell(text: str, style=cell_style) -> Paragraph:
        return Paragraph(str(text), style)

    def _stats_table(data: list[list], col_widths: list[float], header_bg="#1e3a5f") -> Table:
        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4f8"), colors.white]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        return t

    story = []

    # ── Header ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Anomaly Detection Inspection Report", title_style))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(f"Software Version: {app_version}", normal))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(f"Generated: {generated}", normal))
    story.append(Spacer(1, 6 * mm))

    # ── Customer ────────────────────────────────────────────────────────────
    story.append(Paragraph("Customer", heading_style))
    customer_text = customer_name if customer_name else "N/A"
    story.append(Paragraph(f"<b>{customer_text}</b>", customer_style))
    story.append(Spacer(1, 6 * mm))

    # ── Date Range ──────────────────────────────────────────────────────────
    story.append(Paragraph("Date Range", heading_style))
    story.append(Paragraph(f"From: {start_date.strftime('%Y-%m-%d')}", normal))
    story.append(Paragraph(f"To:   {end_date.strftime('%Y-%m-%d')}", normal))
    story.append(Spacer(1, 6 * mm))

    # ── SKU ─────────────────────────────────────────────────────────────────
    sku_names = sorted({i["sku_name"] for i in inspections})
    story.append(Paragraph("SKU", heading_style))
    sku_text = ", ".join(sku_names) if sku_names else "N/A"
    story.append(Paragraph(sku_text, normal))
    story.append(Spacer(1, 6 * mm))

    total = len(inspections)
    ok_count = sum(1 for i in inspections if i["verdict"] == OK)
    not_ok_count = total - ok_count
    pass_rate = round(ok_count / total * 100, 1) if total else 0.0

    # ── Per-SKU × Feature Statistics ─────────────────────────────────────────
    # Group by (SKU, feature) so results from different features are not conflated.
    story.append(Paragraph("Per-SKU Statistics", heading_style))
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for i in inspections:
        groups[(i["sku_name"], i["feature"])].append(i)

    per_sku_data = [[
        _cell("SKU", cell_bold), _cell("Feature", cell_bold),
        _cell("Total", cell_bold), _cell("OK", cell_bold),
        _cell("NOT OK", cell_bold), _cell("Pass Rate", cell_bold),
        _cell("Threshold", cell_bold),
    ]]
    for (sku, feature) in sorted(groups):
        items = groups[(sku, feature)]
        s_total = len(items)
        s_ok = sum(1 for x in items if x["verdict"] == OK)
        s_not_ok = s_total - s_ok
        s_thr = _fmt_threshold_range([x["threshold"] for x in items])
        per_sku_data.append([
            _cell(sku), _cell(_feature_label(feature)),
            _cell(str(s_total)), _cell(str(s_ok)), _cell(str(s_not_ok)),
            _cell(f"{round(s_ok / s_total * 100, 1) if s_total else 0.0:.1f}%"),
            _cell(s_thr),
        ])
    story.append(_stats_table(
        per_sku_data, [32 * mm, 30 * mm, 16 * mm, 14 * mm, 16 * mm, 20 * mm, 22 * mm]
    ))
    story.append(Spacer(1, 6 * mm))

    # ── Aggregate Statistics (only when more than one group) ─────────────────
    if len(groups) > 1:
        story.append(Paragraph("Aggregate Statistics (All SKUs)", heading_style))
        agg_data = [
            [_cell("Metric", cell_bold), _cell("Value", cell_bold)],
            [_cell("SKUs Scanned"), _cell(str(len(sku_names)))],
            [_cell("Features Used"), _cell(str(len({i['feature'] for i in inspections})))],
            [_cell("Total Inspected"), _cell(str(total))],
            [_cell("OK"), _cell(str(ok_count))],
            [_cell("NOT OK"), _cell(str(not_ok_count))],
            [_cell("Pass Rate"), _cell(f"{pass_rate:.1f}%")],
        ]
        story.append(_stats_table(agg_data, [85 * mm, 85 * mm]))
        story.append(Spacer(1, 6 * mm))

    # ── Models Used ─────────────────────────────────────────────────────────
    # One row per distinct (feature, model_version) so the report states which
    # model produced which scans -- not just the currently-loaded model.
    story.append(Paragraph("Models Used", heading_style))
    model_groups: dict[tuple[str, str], int] = defaultdict(int)
    for i in inspections:
        model_groups[(i["feature"], i["model_version"])] += 1
    models_data = [[
        _cell("Feature", cell_bold), _cell("Model Version", cell_bold), _cell("Scans", cell_bold),
    ]]
    for (feature, version), count in sorted(model_groups.items()):
        models_data.append([
            _cell(_feature_label(feature)), _cell(version), _cell(str(count)),
        ])
    story.append(_stats_table(models_data, [50 * mm, 85 * mm, 35 * mm]))
    story.append(Spacer(1, 6 * mm))

    # ── Threshold Settings ──────────────────────────────────────────────────
    # Per (feature, decision_rule); cascade appears as one "Per-stage" row.
    story.append(Paragraph("Threshold Settings", heading_style))
    thr_groups: dict[tuple[str, str | None], list[float]] = defaultdict(list)
    for i in inspections:
        thr_groups[(i["feature"], i["decision_rule"])].append(i["threshold"])
    thr_data = [[
        _cell("Feature", cell_bold), _cell("Decision Rule", cell_bold),
        _cell("Threshold", cell_bold), _cell("Scans", cell_bold),
    ]]
    for (feature, rule), values in sorted(thr_groups.items()):
        if feature == "cascade":
            label = "Per-stage"
        else:
            label = _fmt_threshold_range(values)
        thr_data.append([
            _cell(_feature_label(feature)),
            _cell(rule or "—"),
            _cell(label),
            _cell(str(len(values))),
        ])
    story.append(_stats_table(thr_data, [40 * mm, 45 * mm, 45 * mm, 30 * mm]))
    story.append(Spacer(1, 6 * mm))

    def _reason_for_case(metrics: dict | None) -> str:
        """Resolve a NOK case's human reason, handling cascades and legacy rows."""
        if isinstance(metrics, dict) and metrics.get("stages"):
            return cascade_nok_reason(metrics) or "cascade NOK (see stage breakdown below)"
        if not metrics:
            return "details not recorded"
        return ""

    def _stage_subtable(metrics: dict | None) -> Table | None:
        """Indented per-stage breakdown for a cascade NOK case, if any."""
        if not isinstance(metrics, dict) or not metrics.get("stages"):
            return None
        stage_rows = [[
            _cell("Stage", stage_cell_style),
            _cell("Model", stage_cell_style),
            _cell("Score", stage_cell_style),
            _cell("Threshold", stage_cell_style),
            _cell("Verdict", stage_cell_style),
            _cell("Reason", stage_cell_style),
        ]]
        for s in metrics["stages"]:
            stage_rows.append([
                _cell(f"  {s.get('feature', '?')}", stage_cell_style),
                _cell(s.get("model_version", "—"), stage_cell_style),
                _cell(_fmt_score(s.get("score")), stage_cell_style),
                _cell(_fmt_threshold(s.get("threshold")), stage_cell_style),
                _cell(s.get("verdict", "—"), stage_cell_style),
                _cell(s.get("reason", "") or "—", stage_cell_style),
            ])
        sub = Table(stage_rows, colWidths=[28 * mm, 30 * mm, 14 * mm, 16 * mm, 16 * mm, 56 * mm])
        sub.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        return sub

    # ── NOK Cases ───────────────────────────────────────────────────────────
    nok_cases = [i for i in inspections if i["verdict"] == NOT_OK]
    story.append(Paragraph(f"NOK Cases ({len(nok_cases)})", heading_style))

    if not nok_cases:
        story.append(Paragraph("No NOK cases in the selected range.", normal))
        doc.build(story)
        return buffer.getvalue()

    nok_rows = [[
        _cell("Time (UTC)", cell_bold), _cell("SKU", cell_bold),
        _cell("Feature", cell_bold), _cell("Decision Rule", cell_bold),
        _cell("Score", cell_bold), _cell("Threshold", cell_bold),
        _cell("Reason", cell_bold),
    ]]
    col_widths = [24 * mm, 20 * mm, 22 * mm, 24 * mm, 14 * mm, 16 * mm, 50 * mm]

    for case in nok_cases:
        ts = case["timestamp"]
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)
        score = case["score"]
        threshold = case["threshold"]
        feature = case["feature"]
        rule = case["decision_rule"] or "—"
        metrics = case.get("metrics")
        if isinstance(metrics, str):
            metrics = _parse_metrics(metrics)
        elif metrics is None:
            metrics = {}

        reason = _reason_for_case(metrics)

        nok_rows.append([
            _cell(ts_str), _cell(case["sku_name"]),
            _cell(_feature_label(feature)), _cell(rule),
            _cell(_fmt_score(score)), _cell(_fmt_threshold(threshold)),
            _cell(reason),
        ])

        sub = _stage_subtable(metrics)
        if sub is not None:
            nok_rows.append([sub, "", "", "", "", "", ""])

    nok_table = Table(nok_rows, colWidths=col_widths, repeatRows=1)
    nok_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7f1d1d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#fef2f2"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#fca5a5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(nok_table)

    doc.build(story)
    return buffer.getvalue()
