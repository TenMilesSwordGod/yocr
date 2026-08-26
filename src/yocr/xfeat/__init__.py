"""Lazy-loaded XFeat feature extractor (vendored, Apache-2.0).

The heavy imports (torch) happen inside :func:`xfeat_model`, so a server that
never verifies template matches pays nothing, and a missing weight file
degrades the verifier to a no-op instead of crashing matching.
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger("yocr.xfeat")

DEFAULT_WEIGHTS = Path(".cache/xfeat/xfeat.pt")

_models: dict[str, object] = {}
_failures: dict[str, str] = {}
_lock = Lock()


def xfeat_model(weights_path: Path | str = DEFAULT_WEIGHTS) -> object | None:
    """Return the shared XFeat instance for ``weights_path``, or None.

    Each distinct weights path gets its own cached instance (tests may use
    throwaway copies without disturbing the production model). Once a path
    fails to load it is remembered as failed, so repeated calls stay cheap.
    """
    key = str(weights_path)
    if key in _models or key in _failures:
        return _models.get(key)
    with _lock:
        if key in _models or key in _failures:
            return _models.get(key)
        path = Path(weights_path)
        try:
            from .xfeat import XFeat  # noqa: PLC0415 - lazy heavy import

            if not path.is_file():
                raise FileNotFoundError(
                    f"xfeat weights not found at {path} "
                    "(run `make models-download` or place the file there)"
                )
            _models[key] = XFeat(weights=str(path))
            logger.info("xfeat verifier ready (%s)", path)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash matching
            _failures[key] = f"{type(exc).__name__}: {exc}"
            logger.warning("xfeat verification disabled: %s", _failures[key])
            return None
        return _models[key]


def reset_cache() -> None:
    """Drop every cached instance/failure (test isolation hook)."""
    with _lock:
        _models.clear()
        _failures.clear()