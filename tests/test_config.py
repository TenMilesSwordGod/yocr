from pathlib import Path

from yocr.config import Settings, parse_model_aliases


def test_alias_parsing():
    raw = "ui=models/ui.pt, screen=docling-project/ScreenParser, bad-item, =novalue"
    aliases = parse_model_aliases(raw)
    assert aliases == {"ui": "models/ui.pt", "screen": "docling-project/ScreenParser"}


def test_settings_lazy_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("YOCR_PORT", "9999")
    monkeypatch.setenv("YOCR_MODELS_DIR", str(tmp_path))
    s = Settings()
    assert s.port == 9999
    assert s.models_dir == tmp_path
    # defaults still sane
    assert Settings().device == "cpu"
