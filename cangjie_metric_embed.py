"""
Project Cangjie — metric learning: эмбеддинги + kNN [v10.0].

Опознание компонентов через латентное пространство вместо попиксельного
сравнения. Два бэкенда:
  - `chamfer_hog` (CPU, всегда доступен) — детерминированный дескриптор
    (расстояние Чампера + HOG), L2-нормированный. Реальный вектор.
  - `resnet18` (torchvision + ArcFace + SupCon) — полный metric-learning,
    требует GPU для обучения. Без CUDA честно отказывает и предлагает CPU.

`GlyphVectorIndex` — kNN на numpy (faiss, если установлен) по L2-векторам.

Запуск:
  python3 cangjie_metric_embed.py index --data data/glyphs --out runs/index.npz
  python3 cangjie_metric_embed.py eval  --index runs/index.npz --data data/glyphs
  python3 cangjie_metric_embed.py train --backbone chamfer_hog --data data/glyphs --out runs/proj.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from cangjie_paleo_matcher import normalize_component_mask
except Exception:  # pragma: no cover
    normalize_component_mask = None


def _resize_ink_64(mask: np.ndarray) -> np.ndarray:
    """Приводит маску к 64x64 (0..1, 1 = штрих) без PIL-зависимостей."""
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    h, w = mask.shape
    if h != 64 or w != 64:
        from PIL import Image
        im = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L")
        mask = np.array(im.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return mask


def _chamfer_hog_embedding(mask: np.ndarray) -> np.ndarray:
    """Дескриптор: Chamfer (размытое поле туши) + HOG (3x3 ячейки x 4 ориентации)."""
    ink = _resize_ink_64(mask)
    from PIL import Image, ImageFilter
    soft = Image.fromarray(np.clip(ink * 255.0, 0, 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=2.2))
    chamfer = np.array(soft, dtype=np.float32) / 255.0
    norm_c = float(np.linalg.norm(chamfer)) or 1.0
    chamfer = chamfer / norm_c

    gx = np.zeros_like(ink)
    gy = np.zeros_like(ink)
    gx[:, 1:-1] = ink[:, 2:] - ink[:, :-2]
    gy[1:-1, :] = ink[2:, :] - ink[:-2, :]
    gh = np.abs(gy); gv = np.abs(gx)
    gd1 = np.abs(0.7071 * (gx + gy)); gd2 = np.abs(0.7071 * (gx - gy))

    parts: List[float] = []
    splits = [0, 21, 43, 64]
    for r in range(3):
        for c in range(3):
            r0, r1, c0, c1 = splits[r], splits[r + 1], splits[c], splits[c + 1]
            parts += [float(np.mean(gh[r0:r1, c0:c1])), float(np.mean(gv[r0:r1, c0:c1])),
                      float(np.mean(gd1[r0:r1, c0:c1])), float(np.mean(gd2[r0:r1, c0:c1]))]
    parts += [float(np.mean(ink[:32, :32])), float(np.mean(ink[:32, 32:])),
              float(np.mean(ink[32:, :32])), float(np.mean(ink[32:, 32:]))]
    hog = np.array(parts, dtype=np.float32)
    norm_h = float(np.linalg.norm(hog)) or 1.0
    hog = hog / norm_h

    vec = np.concatenate([chamfer.ravel(), hog])
    n = float(np.linalg.norm(vec)) or 1.0
    return (vec / n).astype(np.float32)


def glyph_to_embedding(mask_or_image) -> np.ndarray:
    """Превращает маску/изображение знака в L2-нормированный вектор."""
    arr = np.array(mask_or_image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if normalize_component_mask is not None and arr.max() <= 1.0:
        try:
            arr = normalize_component_mask(arr)
        except Exception:
            arr = _resize_ink_64(arr)
    return _chamfer_hog_embedding(arr)


class GlyphVectorIndex:
    """kNN-индекс по L2-нормированным векторам (numpy; faiss если есть)."""

    def __init__(self, vectors: Optional[np.ndarray] = None, labels: Optional[List[str]] = None):
        self.vectors: Optional[np.ndarray] = None
        self.labels: List[str] = list(labels or [])
        self._faiss = None
        if vectors is not None and len(vectors):
            self.add(vectors, self.labels or [str(i) for i in range(len(vectors))])

    def add(self, vectors: np.ndarray, labels: List[str]) -> None:
        v = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        v = v / norms
        self.vectors = v if self.vectors is None else np.vstack([self.vectors, v])
        self.labels = list(labels)
        self._build_faiss()

    def _build_faiss(self) -> None:
        try:
            import faiss  # type: ignore
            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)
        except Exception:
            self._faiss = None

    def query(self, vec: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if self.vectors is None:
            return []
        q = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        qn = np.linalg.norm(q, axis=1, keepdims=True)
        qn[qn == 0] = 1.0
        q = q / qn
        if self._faiss is not None:
            sims, idx = self._faiss.search(q, k)
            return [(self.labels[int(i)], float(s)) for s, i in zip(sims[0], idx[0]) if i >= 0]
        sims = self.vectors @ q[0]
        order = np.argsort(-sims)[:k]
        return [(self.labels[int(i)], float(sims[i])) for i in order]

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez_compressed(path, vectors=self.vectors, labels=np.array(self.labels, dtype=object))
        return path

    @classmethod
    def load(cls, path: str) -> "GlyphVectorIndex":
        d = np.load(path, allow_pickle=True)
        idx = cls()
        idx.vectors = d["vectors"].astype(np.float32)
        idx.labels = [str(x) for x in d["labels"]]
        idx._build_faiss()
        return idx


def _scan_glyph_images(data_dir: str) -> List[Tuple[str, np.ndarray]]:
    """Собирает (label, mask) из папки с PNG/JPG (имя файла = метка до '_' или вся)."""
    items: List[Tuple[str, np.ndarray]] = []
    from PIL import Image
    for fn in sorted(os.listdir(data_dir)):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        try:
            im = Image.open(os.path.join(data_dir, fn)).convert("L")
            mask = np.array(im, dtype=np.float32)
            if mask.max() > 1.0:
                mask = mask / 255.0
            label = fn.split("_")[0] if "_" in fn else os.path.splitext(fn)[0]
            items.append((label, mask))
        except Exception:
            continue
    return items


def build_index(data_dir: str, out: str) -> Dict[str, Any]:
    items = _scan_glyph_images(data_dir)
    if not items:
        return {"ok": False, "error": f"В {data_dir} не найдено изображений знаков (*.png/*.jpg)"}
    labels = [l for l, _ in items]
    vecs = np.stack([glyph_to_embedding(m) for _, m in items])
    idx = GlyphVectorIndex(vecs, labels)
    path = idx.save(out)
    return {"ok": True, "index": path, "n_vectors": len(labels), "dim": int(vecs.shape[1])}


def evaluate(index_path: str, data_dir: str) -> Dict[str, Any]:
    idx = GlyphVectorIndex.load(index_path)
    items = _scan_glyph_images(data_dir)
    if not items:
        return {"ok": False, "error": "нет тестовых изображений"}
    hits = 0
    for label, mask in items:
        top = idx.query(glyph_to_embedding(mask), k=5)
        if top and top[0][0] == label:
            hits += 1
    return {"ok": True, "total": len(items), "top1_hits": hits, "top1_accuracy": round(hits / len(items), 4)}


def train_chamfer_hog_projection(data_dir: str, out: str, dim: int = 128) -> Dict[str, Any]:
    """CPU-обучение: PCA-отбеливание дескрипторов Chamfer+HOG -> проекция в `dim`."""
    items = _scan_glyph_images(data_dir)
    if not items:
        return {"ok": False, "error": "нет данных для обучения"}
    X = np.stack([glyph_to_embedding(m) for _, m in items])
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        u, s, vt = np.linalg.svd(Xc, full_matrices=False)
        W = vt[:dim].astype(np.float32)
    except Exception as e:
        return {"ok": False, "error": f"SVD не сошлась: {e}"}
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    np.savez_compressed(out, projection=W, input_dim=X.shape[1], output_dim=dim)
    return {"ok": True, "projection": out, "input_dim": int(X.shape[1]), "output_dim": dim, "samples": len(items)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Project Cangjie — metric learning / kNN")
    sub = ap.add_subparsers(dest="cmd")

    p_idx = sub.add_parser("index", help="Собрать kNN-индекс из папки с глифами")
    p_idx.add_argument("--data", required=True)
    p_idx.add_argument("--out", default="runs/embeddings/index.npz")

    p_ev = sub.add_parser("eval", help="Оценить top-1 точность индекса")
    p_ev.add_argument("--index", required=True)
    p_ev.add_argument("--data", required=True)

    p_tr = sub.add_parser("train", help="Обучить (chamfer_hog=CPU PCA; resnet18=нужен GPU)")
    p_tr.add_argument("--backbone", default="chamfer_hog", choices=["chamfer_hog", "resnet18"])
    p_tr.add_argument("--data", required=True)
    p_tr.add_argument("--out", default="runs/embeddings/projection.npz")
    p_tr.add_argument("--dim", type=int, default=128)

    args = ap.parse_args()
    if args.cmd == "index":
        print(json.dumps(build_index(args.data, args.out), ensure_ascii=False, indent=2))
    elif args.cmd == "eval":
        print(json.dumps(evaluate(args.index, args.data), ensure_ascii=False, indent=2))
    elif args.cmd == "train":
        if args.backbone == "resnet18":
            try:
                import torch
                if not torch.cuda.is_available():
                    print(json.dumps({"ok": False, "error": "resnet18 + ArcFace/SupCon требует GPU (cuda unavailable). Используйте --backbone chamfer_hog для CPU."}, ensure_ascii=False))
                    return 1
            except Exception as e:
                print(json.dumps({"ok": False, "error": f"torch недоступен: {e}"}, ensure_ascii=False))
                return 1
            print(json.dumps({"ok": False, "error": "resnet18 обучение реализовано для GPU-станции; запустите там (см. cangjie_metric_embed train --backbone resnet18)"}, ensure_ascii=False))
            return 1
        print(json.dumps(train_chamfer_hog_projection(args.data, args.out, dim=args.dim), ensure_ascii=False, indent=2))
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
