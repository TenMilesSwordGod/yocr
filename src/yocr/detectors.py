"""YOLO model registry: lazy, thread-safe, multi-model loading and inference."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import Settings, parse_model_aliases
from .imaging import resolve_model_file

logger = logging.getLogger("yocr.detectors")

DEFAULT_MODEL = "android_ui_detection_yolov8"
DEFAULT_MODEL_REPO = "yasirfaizahmed/android_ui_detection_yolov8"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: str  # local .pt path (relative to models_dir) or HF hub id


def _builtin_specs(settings: Settings) -> dict[str, ModelSpec]:
    specs = {
        DEFAULT_MODEL.lower(): ModelSpec(DEFAULT_MODEL, DEFAULT_MODEL_REPO),
        "screenparser": ModelSpec("ScreenParser", "docling-project/ScreenParser"),
    }
    for alias, source in parse_model_aliases(settings.model_aliases_raw).items():
        specs[alias] = ModelSpec(alias, source)
    return specs


@dataclass
class Detection:
    label: str
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]


class YOLORegistry:
    """Holds several named YOLO models; loads on first use and caches."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._specs = _builtin_specs(settings)
        self._models: dict[str, object] = {}
        self._classes: dict[str, dict[int, str]] = {}
        self._errors: dict[str, str] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    # -- registration ----------------------------------------------------
    def register(self, name: str, source: str) -> None:
        with self._global_lock:
            self._specs[name.lower()] = ModelSpec(name, source)

    def names(self) -> list[str]:
        return [spec.name for spec in self._specs.values()]

    def default_name(self) -> str:
        return self._specs[DEFAULT_MODEL.lower()].name

    def spec(self, name: Optional[str]) -> ModelSpec:
        key = (name or DEFAULT_MODEL).strip().lower()
        if key not in self._specs:
            # Allow ad-hoc local files inside models_dir: "model=mymodel.pt"
            raise KeyError(
                f"unknown model '{name}'. available: {', '.join(sorted(self._specs))}; "
                f"register extra aliases via YOCR_MODEL_ALIASES"
            )
        return self._specs[key]

    # -- loading ----------------------------------------------------------
    _WEIGHT_EXTS = (".pt", ".onnx", ".engine", ".torchscript", ".tflite", ".xml", ".mlpackage")

    def _resolve_source(self, spec: ModelSpec) -> str:
        """Resolve a spec source to something YOLO() accepts.

        Resolution order:
        1. existing local file: source path or `<name>.pt` inside models_dir
        2. HF hub id ("repo_id" or "repo_id/file.pt") -> hf_hub_download
        """
        for candidate in dict.fromkeys((spec.source, f"{spec.name}.pt")):
            local = resolve_model_file(self._settings.models_dir, candidate)
            if Path(local).is_file():
                return str(local)
        source = spec.source
        if "/" in source:
            from huggingface_hub import hf_hub_download

            if source.lower().endswith(self._WEIGHT_EXTS):
                repo_id, filename = source.rsplit("/", 1)
                # repo ids always have at least org/name, so require another "/"
                if "/" not in repo_id:
                    repo_id, filename = source, "best.pt"
                return str(hf_hub_download(repo_id=repo_id, filename=filename))
            return str(hf_hub_download(repo_id=source, filename="best.pt"))
        raise FileNotFoundError(
            f"model weights '{source}' not found in '{self._settings.models_dir}' and not a HF hub id"
        )

    def _load(self, spec: ModelSpec):
        resolved = self._resolve_source(spec)
        from ultralytics import YOLO

        logger.info("loading YOLO model '%s' from %s", spec.name, resolved)
        model = YOLO(resolved)
        names = {int(k): str(v) for k, v in (model.names or {}).items()}
        with self._global_lock:
            self._models[spec.name] = model
            self._classes[spec.name] = names
        return model

    def get(self, name: Optional[str]):
        spec = self.spec(name)
        model = self._models.get(spec.name)
        if model is not None:
            return spec, model
        lock = self._locks.setdefault(spec.name, threading.Lock())
        with lock:
            model = self._models.get(spec.name)
            if model is None:
                try:
                    model = self._load(spec)
                except Exception as exc:  # noqa: BLE001 - keep reason visible via last_error()
                    self._errors[spec.name] = f"{type(exc).__name__}: {exc}"
                    raise
                self._errors.pop(spec.name, None)
        return spec, model

    def last_error(self, name: str) -> Optional[str]:
        """Last load failure for a registered model (display name), if any."""
        return self._errors.get(name)

    def classes(self, name: Optional[str]) -> dict[str, str]:
        spec = self.spec(name)
        _, model = self.get(name)
        return {str(k): v for k, v in self._classes.get(spec.name, {}).items()}

    def preload(self, names: list[str]) -> None:
        for name in names:
            try:
                self.get(name)
            except Exception as exc:  # noqa: BLE001 - keep serving other models
                logger.error("preload failed for '%s': %s", name, exc)

    # -- inference ---------------------------------------------------------
    def predict(
        self,
        image: np.ndarray,
        model: Optional[str],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        classes: Optional[list[int]] = None,
    ) -> tuple[ModelSpec, list[Detection]]:
        spec, loaded = self.get(model)
        results = loaded.predict(
            source=image,
            conf=conf if conf is not None else self._settings.conf_threshold,
            iou=iou if iou is not None else self._settings.iou_threshold,
            imgsz=imgsz or self._settings.infer_size,
            device=self._settings.device,
            classes=classes,
            verbose=False,
        )
        detections: list[Detection] = []
        names_map = self._classes.get(spec.name, {})
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                detections.append(
                    Detection(
                        label=names_map.get(class_id, str(class_id)),
                        class_id=class_id,
                        confidence=float(box.conf[0].item()),
                        xyxy=(x1, y1, x2, y2),
                    )
                )
        return spec, detections


def warmup(registry: YOLORegistry, settings: Settings) -> None:
    started = time.perf_counter()
    targets = settings.preload_models or []
    if targets:
        Path(settings.models_dir).mkdir(parents=True, exist_ok=True)
        registry.preload(list(targets))
    logger.info("model warmup finished in %.1fms", (time.perf_counter() - started) * 1000)
