import io
from datetime import datetime, timezone
from pypdf import PdfReader
from reports.pdf_generator import generate_report


def _pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "".join(page.extract_text() or "" for page in reader.pages)


def _make_report(**overrides):
    defaults = dict(
        start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 1, 31, tzinfo=timezone.utc),
        total=100,
        ok_count=90,
        not_ok_count=10,
        pass_rate=90.0,
        sku_names=["WIDGET-A", "WIDGET-B"],
        threshold_min=0.5,
        threshold_max=0.5,
        model_version="v1.0-patchcore",
        app_version="1.0.0",
        customer_name="Acme Corp",
    )
    defaults.update(overrides)
    return generate_report(**defaults)


def test_returns_bytes():
    result = _make_report()
    assert isinstance(result, bytes)


def test_is_pdf_magic_bytes():
    result = _make_report()
    assert result[:4] == b"%PDF"


def test_non_empty_output():
    result = _make_report()
    assert len(result) > 1000


def test_varied_threshold_range():
    result = _make_report(threshold_min=0.3, threshold_max=0.7)
    assert result[:4] == b"%PDF"


def test_empty_sku_list():
    result = _make_report(sku_names=[])
    assert result[:4] == b"%PDF"


def test_customer_name_appears_in_pdf():
    result = _make_report(customer_name="Acme Corp")
    text = _pdf_text(result)
    assert "Acme Corp" in text


def test_customer_name_na_when_empty():
    result = _make_report(customer_name="")
    text = _pdf_text(result)
    assert "N/A" in text
