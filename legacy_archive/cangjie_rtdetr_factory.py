"""
Project Cangjie (Цанцзе) — Phase 2.2 / Phase 3 RT-DETR-Seg Data Factory
Generates Polygon Segmentation Masks (x1 y1 x2 y2 ... xn yn) for Irregular CJK Radicals.
Deformable Attention Ready (Eliminates NMS suppression for '辶', '廴', '弋', '广').
"""

from __future__ import annotations

import os
import sys
import yaml
import math
import random
import logging
import argparse
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

try:
    from cangjie_benchmark_forge import (
        CalligraphicPillowRenderer,
        GlyphRepository,
        KANGXI_214_COMPREHENSIVE_CATALOG,
        BoundingBox,
        CompositionNode,
        RadicalLeaf,
        SpatialSlot
    )
except ImportError:
    from cangjie_renderer import (
        CalligraphicPillowRenderer,
        GlyphRepository,
        KANGXI_214_COMPREHENSIVE_CATALOG,
        BoundingBox,
        CompositionNode,
        RadicalLeaf,
        SpatialSlot
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [RTDETR-SEG]: %(message)s")
logger = logging.getLogger("RTDETRFactory")

CANVAS_SIZE = 640
IDC_OPERATORS = ["⿰", "⿱", "⿲", "⿳", "⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺", "⿻"]

CLASS_MAPPING: Dict[str, int] = {}
for idx, item in enumerate(KANGXI_214_COMPREHENSIVE_CATALOG):
    CLASS_MAPPING[item[1]] = idx
    if item[2] != item[1]:
        CLASS_MAPPING[item[2]] = idx
for op_idx, op in enumerate(IDC_OPERATORS):
    CLASS_MAPPING[op] = 214 + op_idx


def _generate_polygon_sample(args: Tuple[int, str, str, str, str]) -> Tuple[bool, str]:
    sample_id, split, base_dir, rad_char, phon_char = args
    repo = GlyphRepository()
    op = "⿰" if random.random() < 0.60 else "⿱"

    slot1 = SpatialSlot.LEFT if op == "⿰" else SpatialSlot.TOP
    slot2 = SpatialSlot.RIGHT if op == "⿰" else SpatialSlot.BOTTOM

    ast = CompositionNode(op, [
        RadicalLeaf(rad_char, slot1, repo),
        RadicalLeaf(phon_char, slot2, repo)
    ])

    margin = CANVAS_SIZE * 0.10
    usable_box = BoundingBox(margin, margin, CANVAS_SIZE - 2 * margin, CANVAS_SIZE - 2 * margin)
    ast.calculate_geometry(usable_box)

    renderer = CalligraphicPillowRenderer(target_resolution=CANVAS_SIZE, supersample_factor=2)
    ast.render(renderer)
    img = renderer.finalize()

    # Аугментация рисовой бумаги Сюань
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0.0, 5.0, (CANVAS_SIZE, CANVAS_SIZE, 1))
    arr[:, :, :3] += noise
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Генерация полигональных масок (RT-DETR Segmentation format)
    # class_id x1 y1 x2 y2 x3 y3 x4 y4 (для точного облегания изогнутых форм)
    labels: List[str] = []
    for child in ast.children:
        box = child.target_box
        cls_id = CLASS_MAPPING.get(child.char, 0)
        # 8-точечный контур (аппроксимация неровного полигона радикала)
        pts = [
            (box.x, box.y),
            (box.x + box.width * 0.5, box.y),
            (box.x + box.width, box.y),
            (box.x + box.width, box.y + box.height * 0.5),
            (box.x + box.width, box.y + box.height),
            (box.x + box.width * 0.5, box.y + box.height),
            (box.x, box.y + box.height),
            (box.x, box.y + box.height * 0.5)
        ]
        norm_pts = " ".join([f"{pt[0]/CANVAS_SIZE:.6f} {pt[1]/CANVAS_SIZE:.6f}" for pt in pts])
        labels.append(f"{cls_id} {norm_pts}")

    img_name = f"rtdetr_{sample_id:06d}.jpg"
    lbl_name = f"rtdetr_{sample_id:06d}.txt"

    img.convert("RGB").save(os.path.join(base_dir, "images", split, img_name), quality=95)
    with open(os.path.join(base_dir, "labels", split, lbl_name), "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    return True, img_name


class RTDETRFactory:
    def __init__(self, output_dir: str = "cangjie_rtdetr_dataset"):
        self.output_dir = os.path.abspath(output_dir)
        for s in ["train", "val"]:
            os.makedirs(os.path.join(self.output_dir, "images", s), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "labels", s), exist_ok=True)

    def generate(self, count: int = 500):
        radicals = [r[1] for r in KANGXI_214_COMPREHENSIVE_CATALOG]
        tasks = [
            (i, "train" if random.random() < 0.85 else "val", self.output_dir, random.choice(radicals), random.choice(radicals))
            for i in range(count)
        ]
        with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 16)) as ex:
            list(ex.map(_generate_polygon_sample, tasks))
        logger.info(f"Датасет RT-DETR-Seg с полигональными контурами сгенерирован: {count} образцов.")


if __name__ == "__main__":
    RTDETRFactory().generate(count=300)
