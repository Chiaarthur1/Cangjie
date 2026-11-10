"""Тесты metric learning (эмбеддинги + kNN)."""

import numpy as np

from cangjie_metric_embed import GlyphVectorIndex, glyph_to_embedding


def test_embedding_normalized():
    mask = np.random.RandomState(1).rand(64, 64).astype(np.float32)
    v = glyph_to_embedding(mask)
    assert v.shape[0] > 0
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3


def test_index_query_top1():
    idx = GlyphVectorIndex()
    rng = np.random.RandomState(2)
    vecs = rng.randn(6, 32).astype(np.float32)
    idx.add(vecs, ["a", "b", "c", "d", "e", "f"])
    top = idx.query(vecs[2], k=2)
    assert top[0][0] == "c"
    assert len(top) == 2


def test_index_save_load(tmp_path):
    idx = GlyphVectorIndex()
    idx.add(np.random.RandomState(3).randn(4, 16).astype(np.float32), ["w", "x", "y", "z"])
    p = str(tmp_path / "idx.npz")
    idx.save(p)
    idx2 = GlyphVectorIndex.load(p)
    assert idx2.labels == ["w", "x", "y", "z"]
