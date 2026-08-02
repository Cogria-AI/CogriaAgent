"""DefaultDocumentExtractor against real files.

The guards (zip bomb, unknown type, image passthrough) need no optional
dependencies and always run. The PDF/DOCX cases build a genuine file and skip
unless the `attachments` extra is installed — "Word documents work" is the kind
of claim that has to be proven on a real .docx, not a stub.
"""

from __future__ import annotations

import io
import zipfile

import pytest
from cogria_agent.attachments.extract import DefaultDocumentExtractor

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def extractor(**kwargs) -> DefaultDocumentExtractor:
    return DefaultDocumentExtractor(**kwargs)


async def test_images_are_passed_through_not_extracted():
    result = await extractor().extract(data=b"\x89PNG...", mime="image/png", filename="a.png")
    assert result["kind"] == "image" and result["text"] is None


async def test_text_is_decoded_leniently():
    result = await extractor().extract(
        data="héllo, wörld".encode() + b"\xff", mime="text/plain", filename="a.txt"
    )
    assert result["kind"] == "text" and "héllo" in result["text"]


async def test_unknown_mime_is_reported_not_raised():
    result = await extractor().extract(
        data=b"\x00\x01", mime="application/x-thing", filename="a.bin"
    )
    assert result["kind"] == "unsupported" and "No extractor" in result["error"]


async def test_zip_bomb_is_refused_before_any_parser_runs():
    """A docx is a zip; 50 MB of zeros compresses to nothing."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", b"\x00" * (50 * 1024 * 1024))
    result = await extractor(max_uncompressed_bytes=1024 * 1024).extract(
        data=buf.getvalue(), mime=DOCX_MIME, filename="bomb.docx"
    )
    assert result["kind"] == "unsupported" and "too much data" in result["error"]


async def test_too_many_zip_entries_is_refused():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(50):
            zf.writestr(f"f{i}.xml", b"x")
    result = await extractor(max_zip_entries=10).extract(
        data=buf.getvalue(), mime=DOCX_MIME, filename="many.docx"
    )
    assert result["kind"] == "unsupported" and "too many" in result["error"]


async def test_corrupt_office_file_is_refused_cleanly():
    result = await extractor().extract(data=b"not a zip at all", mime=DOCX_MIME, filename="x.docx")
    assert result["kind"] == "unsupported" and "not a readable" in result["error"].lower()


async def test_real_pdf_text_is_extracted():
    pymupdf = pytest.importorskip("pymupdf", reason="needs the `attachments` extra")
    pytest.importorskip("pymupdf4llm", reason="needs the `attachments` extra")

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 144), "Revenue for March was 42,000 EUR")
    data = doc.tobytes()
    doc.close()

    result = await extractor(timeout_seconds=120).extract(
        data=data, mime="application/pdf", filename="report.pdf"
    )
    assert result["kind"] == "text"
    assert "42,000 EUR" in result["text"]
    assert result["page_count"] == 1


async def test_pdf_without_text_reports_that_it_needs_ocr():
    pymupdf = pytest.importorskip("pymupdf", reason="needs the `attachments` extra")
    pytest.importorskip("pymupdf4llm", reason="needs the `attachments` extra")

    doc = pymupdf.open()
    doc.new_page()  # blank page — the shape a scan arrives in
    data = doc.tobytes()
    doc.close()

    result = await extractor(timeout_seconds=120).extract(
        data=data, mime="application/pdf", filename="scan.pdf"
    )
    assert result["kind"] == "unsupported"
    assert "scanned" in result["error"].lower()


async def test_real_docx_text_is_extracted():
    docx = pytest.importorskip("docx", reason="needs python-docx to author the fixture")
    pytest.importorskip("markitdown", reason="needs the `attachments` extra")

    buf = io.BytesIO()
    document = docx.Document()
    document.add_heading("Spring Menu", 1)
    document.add_paragraph("Pasta al limone — 12 EUR")
    document.save(buf)

    result = await extractor(timeout_seconds=120).extract(
        data=buf.getvalue(), mime=DOCX_MIME, filename="menu.docx"
    )
    assert result["kind"] == "text"
    assert "Spring Menu" in result["text"]
    assert "Pasta al limone" in result["text"]
