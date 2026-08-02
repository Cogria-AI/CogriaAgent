"""LocalDiskAttachmentStore — the zero-config AttachmentStore (bytes on disk).

Content-addressed: the key is derived from the sha256 of the bytes, so the same
file uploaded twice costs one blob, and **no part of the path ever comes from a
client-supplied filename** (that is the path-traversal defence). Keys are
re-validated on read, so even a corrupted metadata row can't walk the tree.

A deployment that wants S3/GCS implements the same three methods.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

_KEY_RE = re.compile(r"^[0-9a-f]{2}/[0-9a-f]{2}/[0-9a-f]{64}$")


class LocalDiskAttachmentStore:
    """Implements protocols.AttachmentStore."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()

    @property
    def root(self) -> Path:
        return self._root

    async def put(self, *, data: bytes, sha256: str, mime: str) -> str:
        key = f"{sha256[:2]}/{sha256[2:4]}/{sha256}"
        await asyncio.to_thread(self._write, key, data)
        return key

    async def get(self, storage_key: str) -> bytes:
        return await asyncio.to_thread(self._read, storage_key)

    async def delete(self, storage_key: str) -> None:
        await asyncio.to_thread(self._unlink, storage_key)

    def _path(self, storage_key: str) -> Path:
        if not _KEY_RE.match(storage_key or ""):
            raise ValueError(f"refusing malformed storage key: {storage_key!r}")
        return self._root / storage_key

    def _write(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return  # content-addressed: identical bytes, nothing to do
        # Write-then-rename so a crash can't leave a half-written blob.
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)

    def _read(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def _unlink(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)
