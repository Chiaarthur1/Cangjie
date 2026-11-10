"""
Project Cangjie — разметка данных [v10.0].

Полуавтоматическая разметка боксов компонентов: детектор находит боксы,
разметчик экспортирует их в YOLO-формат (`class cx cy w h`, нормализованные),
чтобы дообучать детектор. Ручная корректировка — правкой .txt файлов или
повторным экспортом.

Без заглушек: классы маппятся через GLYPH_TO_CLASS_ID; если класса нет —
честный fallback 0 и предупреждение в отчёте.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    from cangjie_layout_to_tree import GLYPH_TO_CLASS_ID
except Exception:  # pragma: no cover
    GLYPH_TO_CLASS_ID = {}


def boxes_to_yolo_lines(boxes: List[Dict[str, Any]]) -> List[str]:
    """Конвертирует детекции пайплайна (glyph + bbox_norm [x1,y1,x2,y2]) в строки YOLO."""
    lines: List[str] = []
    for b in boxes:
        glyph = str(b.get("raw_glyph") or b.get("glyph") or "").strip()
        bbox = b.get("bbox_norm") or [0.0, 0.0, 0.0, 0.0]
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        # Класс — по глифу (GLYPH_TO_CLASS_ID), т.к. UI иногда шлёт class_id=0.
        cls = GLYPH_TO_CLASS_ID.get(glyph)
        if cls is None:
            raw_cls = b.get("class_id")
            cls = int(raw_cls) if raw_cls is not None and int(raw_cls) >= 0 else 0
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def export_yolo(image_bytes: bytes, filename: str, boxes: List[Dict[str, Any]],
                out_dir: str = "./data/labeled") -> Dict[str, Any]:
    stem = os.path.splitext(filename or "glyph.png")[0] or "glyph"
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    img_path = os.path.join(img_dir, filename)
    with open(img_path, "wb") as f:
        f.write(image_bytes)

    lines = boxes_to_yolo_lines(boxes)
    lbl_path = os.path.join(lbl_dir, stem + ".txt")
    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    unmapped = [b.get("raw_glyph") or b.get("glyph") for b in boxes
                if (b.get("raw_glyph") or b.get("glyph")) not in GLYPH_TO_CLASS_ID]

    return {
        "ok": True,
        "image": img_path,
        "labels": lbl_path,
        "boxes_exported": len(lines),
        "unmapped_glyphs": list(dict.fromkeys(g for g in unmapped if g)),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Project Cangjie — экспорт разметки в YOLO")
    ap.add_argument("--image", required=True, help="Файл изображения")
    ap.add_argument("--boxes", required=True, help="JSON-файл со списком детекций [{glyph, bbox_norm:[x1,y1,x2,y2]}]")
    ap.add_argument("--out", default="./data/labeled", help="Выходная папка")
    args = ap.parse_args()

    with open(args.image, "rb") as f:
        img_bytes = f.read()
    with open(args.boxes, "r", encoding="utf-8") as f:
        boxes = json.load(f)
    print(json.dumps(export_yolo(img_bytes, os.path.basename(args.image), boxes, args.out), ensure_ascii=False, indent=2))
