"""Тесты реестра моделей."""

from cangjie_config import CangjieConfig
from cangjie_model_registry import CangjieModelRegistry


def test_active_structure():
    reg = CangjieModelRegistry(config=CangjieConfig())
    a = reg.active()
    assert set(a) == {"detector", "matcher", "llm"}


def test_switch_matcher(monkeypatch):
    import cangjie_model_registry as mr

    monkeypatch.setattr(mr, "save_config", lambda cfg, path=None: None)  # не писать в реальный конфиг
    cfg = CangjieConfig()
    reg = CangjieModelRegistry(config=cfg)
    r = reg.switch("matcher", "chamfer_hog")
    assert r["ok"] and cfg.matcher.backend == "chamfer_hog"
    r2 = reg.switch("llm", "none")
    assert r2["ok"] and cfg.llm.backend == "none"


def test_switch_unknown_kind(monkeypatch):
    import cangjie_model_registry as mr

    monkeypatch.setattr(mr, "save_config", lambda cfg, path=None: None)
    reg = CangjieModelRegistry(config=CangjieConfig())
    r = reg.switch("bogus", "x")
    assert not r["ok"]
