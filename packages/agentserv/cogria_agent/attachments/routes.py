"""The /attachments endpoints, mounted by build_app when the seams are wired.

    POST   /attachments          multipart upload -> metadata
    GET    /attachments/{id}     metadata
    GET    /attachments/{id}/raw the bytes (preview/download)
    DELETE /attachments/{id}     forget it

Auth is the same JWT the rest of the service uses, and `sub` is the owner. There
is no unauthenticated or signed-URL path: the BFF proxies previews with the
caller's own token, so a leaked id is not enough to read a file.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile

from .repo import public_view
from .service import AttachmentRejected, AttachmentService

logger = logging.getLogger("cogria.attachments")

# Read caps in chunks so an oversized upload is refused as it arrives rather
# than after it has all been buffered.
_CHUNK = 512 * 1024


def build_attachment_router(service: AttachmentService, verify_jwt) -> APIRouter:
    router = APIRouter(prefix="/attachments", tags=["attachments"])

    @router.post("")
    async def upload(file: UploadFile = File(...), claims: dict = Depends(verify_jwt)) -> dict:
        limit = service.config.max_file_bytes
        buffer = bytearray()
        while chunk := await file.read(_CHUNK):
            buffer.extend(chunk)
            if len(buffer) > limit:
                raise HTTPException(
                    413, f"That file is larger than the {limit / (1024 * 1024):.0f} MB limit."
                )
        try:
            return await service.upload(
                data=bytes(buffer),
                filename=file.filename or "file",
                declared_mime=file.content_type or "",
                owner_sub=str(claims.get("sub") or ""),
            )
        except AttachmentRejected as e:
            raise HTTPException(e.status_code, e.message) from e

    @router.get("/{attachment_id}")
    async def meta(attachment_id: str, claims: dict = Depends(verify_jwt)) -> dict:
        record = await service.repo.get(attachment_id, owner_sub=str(claims.get("sub") or ""))
        if record is None:
            raise HTTPException(404, "attachment not found")
        return public_view(record)

    @router.get("/{attachment_id}/raw")
    async def raw(attachment_id: str, claims: dict = Depends(verify_jwt)) -> Response:
        record = await service.repo.get(attachment_id, owner_sub=str(claims.get("sub") or ""))
        if record is None:
            raise HTTPException(404, "attachment not found")
        try:
            data = await service.load_bytes(record)
        except FileNotFoundError as e:
            raise HTTPException(410, "attachment bytes are no longer stored") from e
        # Never inline: a user-supplied file rendered on our origin is an XSS.
        return Response(
            content=data,
            media_type=record["mime"],
            headers={
                "Content-Disposition": f'attachment; filename="{_safe_filename(record["filename"])}"',
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private, max-age=300",
            },
        )

    @router.delete("/{attachment_id}")
    async def remove(attachment_id: str, claims: dict = Depends(verify_jwt)) -> dict:
        ok = await service.delete(attachment_id, owner_sub=str(claims.get("sub") or ""))
        if not ok:
            raise HTTPException(404, "attachment not found")
        return {"ok": True}

    return router


def _safe_filename(name: str) -> str:
    """Header-safe rendition: drop quotes, control characters and path parts."""
    cleaned = "".join(c for c in (name or "file") if c.isprintable() and c not in '"\\/')
    return cleaned[:200] or "file"
