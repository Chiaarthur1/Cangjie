"""Тесты round-trip разбора IDS."""

from cangjie_parser import IDSParser
from cangjie_tree_eval import ast_to_ids, is_composite


def test_ast_roundtrip():
    parser = IDSParser(radical_map={}, phonetic_set=set())
    for ids in ["⿰氵寺", "⿱艹⿳雨口巫", "⿺辶首", "⿴囗玉", "⿲木木木"]:
        ast = parser.parse_sequence(ids)
        assert ast is not None, ids
        assert ast_to_ids(ast) == ids, ids


def test_is_composite():
    assert is_composite("⿰氵寺")
    assert is_composite("⿳雨口巫")
    assert not is_composite("水")
    assert not is_composite("我")
