"""Persistent registry of sucai (素材) template images.

Each sucai is a small reference picture (button / icon / widget crop) with a
user-facing id and a free-text describe. Registered sucai are stored on disk:

    <sucai_dir>/meta.json        id -> metadata (describe, size, created_at)
    <sucai_dir>/images/<id>.png  normalized PNG bytes

so the library survives service restarts and can be copied between hosts.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2

from .imaging import decode_image, encode_png

logger = logging.getLogger("yocr.sucai")

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MAX_SIDE = 2048


class SucaiError(ValueError):
    """Invalid sucai payload (bad id, oversized image, ...)."""


class SucaiConflict(KeyError):
    """Sucai id already registered."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SucaiStore:
    """Thread-safe CRUD + on-disk persistence for sucai template images."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.meta_path = self.root / "meta.json"
        self._lock = threading.Lock()
        self._meta: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------- io ----
    def _load(self) -> None:
        if not self.meta_path.is_file():
            return
        try:
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("sucai meta.json unreadable (%s); starting empty", exc)
            return
        if not isinstance(data, dict):
            logger.error("sucai meta.json has unexpected layout; starting empty")
            return
        meta: dict[str, dict] = {}
        for key, value in data.items():
            # Tolerate hand-edited/corrupt files: skip malformed records
            # instead of poisoning every later list/find call.
            if not isinstance(value, dict) or not isinstance(value.get("id"), str):
                logger.warning("sucai meta.json: skipping malformed record %r", key)
                continue
            meta[str(key)] = value
        self._meta = meta

    def _save_locked(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.meta_path)

    def _image_path(self, sid: str) -> Path:
        return self.images_dir / f"{sid}.png"

    # ------------------------------------------------------------ crud ---
    @staticmethod
    def validate_id(sid: str) -> str:
        if not ID_PATTERN.match(sid or ""):
            raise SucaiError("id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
        return sid

    def create(self, image_bytes: bytes, *, describe: str = "", sid: str | None = None) -> dict:
        """Register a new sucai; returns its metadata record."""
        image = decode_image(image_bytes)  # raises ValueError on corrupt input
        height, width = image.shape[:2]
        if max(width, height) > MAX_SIDE:
            raise SucaiError(f"picture too large ({width}x{height}); max side is {MAX_SIDE}px")
        png = encode_png(image)

        with self._lock:
            if sid:
                self.validate_id(sid)
                if sid in self._meta:
                    raise SucaiConflict(f"sucai id '{sid}' already exists")
            else:
                while True:
                    sid = f"s-{uuid.uuid4().hex[:8]}"
                    if sid not in self._meta:
                        break
            self.images_dir.mkdir(parents=True, exist_ok=True)
            self._image_path(sid).write_bytes(png)
            record = {
                "id": sid,
                "describe": (describe or "").strip(),
                "width": int(width),
                "height": int(height),
                "size_bytes": len(png),
                "created_at": _now_iso(),
            }
            self._meta[sid] = record
            self._save_locked()
        logger.info("sucai registered: %s (%dx%d) %s", sid, width, height, describe[:40])
        return dict(record)

    def list(self) -> list[dict]:
        with self._lock:
            items = [dict(v) for v in self._meta.values()]
        # Newest first; id tiebreak keeps order stable within the same second.
        items.sort(key=lambda r: (r.get("created_at") or "", r.get("id") or ""), reverse=True)
        return items

    def get(self, sid: str) -> dict:
        with self._lock:
            record = self._meta.get(sid)
        if record is None:
            raise KeyError(f"sucai '{sid}' not found")
        return dict(record)

    def update(self, sid: str, *, describe: str | None = None,
               image_bytes: bytes | None = None) -> dict:
        """Update describe and/or replace the picture."""
        if image_bytes is None and describe is None:
            raise SucaiError("nothing to update (provide describe and/or file)")
        with self._lock:
            if sid not in self._meta:
                raise KeyError(f"sucai '{sid}' not found")
            record = self._meta[sid]
            if image_bytes is not None:
                image = decode_image(image_bytes)
                height, width = image.shape[:2]
                if max(width, height) > MAX_SIDE:
                    raise SucaiError(
                        f"picture too large ({width}x{height}); max side is {MAX_SIDE}px"
                    )
                png = encode_png(image)
                self.images_dir.mkdir(parents=True, exist_ok=True)
                self._image_path(sid).write_bytes(png)
                record.update(width=int(width), height=int(height),
                              size_bytes=len(png), updated_at=_now_iso())
            if describe is not None:
                record["describe"] = describe.strip()
                record.setdefault("updated_at", _now_iso())
            self._save_locked()
            return dict(record)

    def delete(self, sid: str) -> None:
        with self._lock:
            if sid not in self._meta:
                raise KeyError(f"sucai '{sid}' not found")
            del self._meta[sid]
            self._save_locked()
        try:
            self._image_path(sid).unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - best effort cleanup
            logger.warning("could not remove image for %s: %s", sid, exc)

    def read_image(self, sid: str) -> bytes:
        self.get(sid)  # 404 semantics for unknown ids
        path = self._image_path(sid)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.read_bytes()

    def count(self) -> int:
        with self._lock:
            return len(self._meta)

    def iter_records(self) -> list[dict]:
        """Snapshot of all records in insertion order (used by the finder)."""
        with self._lock:
            return [dict(v) for v in self._meta.values()]


__all__ = ["SucaiStore", "SucaiError", "SucaiConflict", "ID_PATTERN"]
