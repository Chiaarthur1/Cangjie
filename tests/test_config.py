"""Тесты единого конфига."""

from cangjie_config import CangjieConfig, load_config, save_config


def test_defaults():
    c = CangjieConfig()
    assert c.detector.backend == "auto"
    assert c.matcher.backend == "chamfer_hog"
    assert c.llm.backend == "none"
    assert c.machine == "auto"


def test_roundtrip(tmp_path):
    c = CangjieConfig()
    c.detector.backend = "pixel"
    c.llm.backend = "ollama"
    c.llm.model = "qwen2.5:7b"
    c.llm.url = "http://localhost:11434"
    p = str(tmp_path / "cfg.json")
    save_config(c, p)
    c2 = load_config(p)
    assert c2.to_dict() == c.to_dict()


def test_validate_ok():
    assert CangjieConfig().validate() == []


def test_validate_bad_detector():
    c = CangjieConfig()
    c.detector.backend = "bogus"
    assert any("detector" in e for e in c.validate())


def test_validate_llm_needs_url():
    c = CangjieConfig()
    c.llm.backend = "ollama"  # без url
    assert any("url" in e for e in c.validate())


def test_resolve_matcher_embedding_without_index(monkeypatch, tmp_path):
    import cangjie_config as cc
    monkeypatch.setattr(cc, "EMBEDDING_INDEX_PATH", str(tmp_path / "nope.npz"))
    c = CangjieConfig()
    c.matcher.backend = "embedding"
    assert c.resolve_matcher_backend() == "chamfer_hog"  # индекса нет
