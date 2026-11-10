"""Тесты графа знаний (быстрые, без загрузки онтологии)."""

from cangjie_knowledge_graph import _dedup


def test_dedup_preserves_order():
    assert _dedup(["a", "b", "a", "", "c"]) == ["a", "b", "c"]
    assert _dedup([]) == []
    assert _dedup(["x"]) == ["x"]
