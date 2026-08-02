"""DefaultDocumentExtractor — bytes to text the model can read.

Routing by (sniffed) mime:
  image/*   -> no extraction; the bytes go to a vision model as-is
  pdf       -> pymupdf4llm markdown (fast, layout-aware, no model weights)
  ooxml     -> markitdown (docx/xlsx/pptx in one path, tables preserved)
  text/*    -> decoded

Needs the `attachments` extra; imports are lazy so the kernel still starts
without it and the error names the fix.

Hardening that belongs here rather than in the caller:
  - office files are zip containers -> inflated-size and entry-count caps
    (zip bomb) before any parser sees them,
  - parsers are a known CVE surface -> each runs in a worker thread under a
    timeout, and their chatty stdout is swallowed so it can't pollute logs.

A timeout frees the request, not the thread: CPython can't kill a thread
mid-parse. A deployment that must hard-kill a runaway parse should supply an
extractor that shells out to a subprocess.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import zipfile
from typing import Any

logger = logging.getLogger("cogria.attachments")

IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
TEXT_MIMES = {"text/plain", "text/markdown", "text/csv", "application/json"}
PDF_MIME = "application/pdf"
OOXML_EXTENSIONS = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
}

_MISSING_EXTRA = (
    'attachment extraction needs the "attachments" extra: '
    'pip install "cogria-agentserv[attachments]"'
)


def _result(
    kind: str, *, text: str | None = None, pages: int | None = None, error: str | None = None
):
    return {"kind": kind, "text": text, "page_count": pages, "error": error}


class DefaultDocumentExtractor:
    """Implements protocols.DocumentExtractor."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        max_uncompressed_bytes: int = 200 * 1024 * 1024,
        max_zip_entries: int = 5_000,
    ) -> None:
        self._timeout = timeout_seconds
        self._max_uncompressed = max_uncompressed_bytes
        self._max_entries = max_zip_entries

    async def extract(self, *, data: bytes, mime: str, filename: str) -> dict[str, Any]:
        if mime in IMAGE_MIMES:
            return _result("image")

        if mime in TEXT_MIMES:
            return _result("text", text=data.decode("utf-8", errors="replace"))

        if mime == PDF_MIME:
            worker, args = _extract_pdf, (data,)
        elif mime in OOXML_EXTENSIONS:
            guard = self._check_zip(data)
            if guard:
                return _result("unsupported", error=guard)
            worker, args = _extract_office, (data, OOXML_EXTENSIONS[mime])
        else:
            return _result("unsupported", error=f"No extractor for {mime}.")

        try:
            text, pages = await asyncio.wait_for(
                asyncio.to_thread(worker, *args), timeout=self._timeout
            )
        except TimeoutError:
            return _result("unsupported", error="Extraction timed out.")
        except ImportError as e:
            logger.error("extractor dependency missing: %s", e)
            return _result("unsupported", error=_MISSING_EXTRA)
        except Exception as e:  # noqa: BLE001 — a malformed file must not 500
            logger.warning("extraction failed for %s (%s): %s", filename, mime, e)
            return _result("unsupported", error=f"Could not read this file ({type(e).__name__}).")

        if not (text or "").strip():
            # Almost always a scan/photo saved as a document.
            return _result(
                "unsupported",
                pages=pages,
                error="No extractable text — this looks like a scanned document, and OCR is not configured.",
            )
        return _result("text", text=text, pages=pages)

    def _check_zip(self, data: bytes) -> str | None:
        """Zip-bomb guard. Returns an error message, or None when it's sane."""
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                infos = zf.infolist()
                if len(infos) > self._max_entries:
                    return "This file has too many internal entries to open safely."
                if sum(i.file_size for i in infos) > self._max_uncompressed:
                    return "This file expands to too much data to open safely."
        except zipfile.BadZipFile:
            return "This file is not a readable Office document."
        return None


def _extract_pdf(data: bytes) -> tuple[str, int | None]:
    import pymupdf  # noqa: PLC0415 — lazy: optional extra
    import pymupdf4llm  # noqa: PLC0415

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        pages = doc.page_count
        # to_markdown narrates its progress on stdout; keep it out of the logs.
        with contextlib.redirect_stdout(io.StringIO()):
            text = pymupdf4llm.to_markdown(doc)
    return text, pages


def _extract_office(data: bytes, extension: str) -> tuple[str, int | None]:
    from markitdown import MarkItDown, StreamInfo  # noqa: PLC0415 — lazy: optional extra

    converter = MarkItDown(enable_plugins=False)
    with contextlib.redirect_stdout(io.StringIO()):
        result = converter.convert_stream(
            io.BytesIO(data), stream_info=StreamInfo(extension=extension)
        )
    return result.text_content, None
