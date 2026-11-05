"""
Project Cangjie (Цанцзе) — Phase 3 Enterprise Synthetic Data Factory [v4.0]
High-Throughput Parallel Dataset Generator for RT-DETRv2-X / YOLO Architecture.
Synthesizes:
- 214 Kangxi Radicals (Classes 0..213, Zero Allograph Collision on 阝 and 月)
- 12 Spatial IDC Operators (Classes 214..225: ⿰, ⿱, ⿲, ⿳, ⿴, ⿵, ⿶, ⿷, ⿸, ⿹, ⿺, ⿻)
- Fractal Hyper-Complex Multi-Register Glyphs (5..10 radicals per char, 50..172+ strokes)
"""

from __future__ import annotations

import os
import sys
try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False
import math
import random
import logging
import argparse
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageFilter
import numpy as np

try:
    from cangjie_benchmark_forge import (
        KANGXI_214_COMPREHENSIVE_CATALOG,
        PHONETICS_COMPREHENSIVE_DB,
        GlyphRepository,
        CalligraphicPillowRenderer,
        BoundingBox,
        RadicalLeaf,
        SpatialSlot,
        SafeAllographResolver
    )
except ImportError:
    from cangjie_renderer import (
        KANGXI_214_COMPREHENSIVE_CATALOG,
        PHONETICS_COMPREHENSIVE_DB,
        GlyphRepository,
        CalligraphicPillowRenderer,
        BoundingBox,
        RadicalLeaf,
        SpatialSlot,
        SafeAllographResolver
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [YOLO-FACTORY-v4]: %(message)s"
)
logger = logging.getLogger("CangjieYOLOFactory")

IDC_OPERATORS: List[str] = [
    "⿰", "⿱", "⿲", "⿳", "⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺", "⿻"
]

RADICAL_TO_CLASS: Dict[str, int] = {}
CLASS_TO_GLYPH: Dict[int, str] = {}
CLASS_NAMES: List[str] = []

# 1. Регистрируем канонические глифы 214 ключей без перезаписи коллизий (月/肉, 阜/邑)
for idx, item in enumerate(KANGXI_214_COMPREHENSIVE_CATALOG):
    base_glyph = item[1]
    allograph = item[2]
    name = item[5]
    class_name = f"rad_{idx:03d}_{name}"

    CLASS_NAMES.append(class_name)
    RADICAL_TO_CLASS[base_glyph] = idx
    CLASS_TO_GLYPH[idx] = base_glyph

for idx, item in enumerate(KANGXI_214_COMPREHENSIVE_CATALOG):
    allograph = item[2]
    if allograph not in RADICAL_TO_CLASS:
        RADICAL_TO_CLASS[allograph] = idx

for idx, op in enumerate(IDC_OPERATORS):
    op_cls = 214 + idx
    op_name = f"idc_{op}"
    CLASS_NAMES.append(op_name)
    CLASS_TO_GLYPH[op_cls] = op

TOP_RADICALS = {"艹", "⺮", "雨", "⻗", "宀", "冖", "亠", "耂", "罒", "穴", "竹", "艸", "网", "老"}
BOTTOM_RADICALS = {"灬", "氺", "心", "火", "皿", "足"}
LEFT_RADICALS = {"氵", "亻", "扌", "忄", "钅", "讠", "犭", "木", "朩", "礻", "衤", "肉", "日", "土", "女", "阜", "车"}
RIGHT_RADICALS = {"刂", "刀", "攵", "攴", "邑", "鸟", "鳥", "页", "頁", "斤", "殳", "欠"}
ENCLOSURE_OUTERS = {
    "⿴": ["囗"],
    "⿵": ["門", "门", "冂", "冖"],
    "⿶": ["凵"],
    "⿷": ["匚", "匸"],
    "⿸": ["广", "厂", "尸", "戶", "麻", "疒"],
    "⿹": ["气", "勹", "戈"],
    "⿺": ["辵", "辶", "廴", "走"],
    "⿻": ["十", "口", "木"]
}


@dataclass
class BBoxAnnotation:
    class_id: int
    xc: float
    yc: float
    w: float
    h: float

    def to_yolo_line(self) -> str:
        xc = max(0.0001, min(0.9999, self.xc))
        yc = max(0.0001, min(0.9999, self.yc))
        w = max(0.0001, min(0.9999, self.w))
        h = max(0.0001, min(0.9999, self.h))
        return f"{self.class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


class ChineseInkAugmentor:
    PAPERS = [
        (255, 255, 255),      # Чистая цифровая бумага
        (248, 245, 236),      # Рисовая бумага Сюаньчжи
        (244, 238, 222),      # Состаренный свиток
        (238, 232, 218),      # Музейный шелк
        (252, 235, 190),      # Желтая даосская бумага для талисманов (Хуанбяочжи 黄表纸)
    ]

    @classmethod
    def apply_paper_texture(cls, img: Image.Image, noise_std: float = 5.0) -> Image.Image:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, noise_std, arr.shape[:2])
        for c in range(3):
            arr[:, :, c] = np.clip(arr[:, :, c] + noise, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))

    @classmethod
    def apply_ink_bleed(cls, img: Image.Image, p: float = 0.30) -> Image.Image:
        if random.random() < p:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
        return img


class CompositionGeometrySynthesizer:
    def __init__(self, repo: GlyphRepository, imgsz: int = 640):
        self.repo = repo
        self.imgsz = imgsz
        self.all_radicals = [item[1] for item in KANGXI_214_COMPREHENSIVE_CATALOG]
        self.top_list = ["雨", "艸", "竹", "宀", "穴", "网", "冖", "亠"]
        self.bottom_list = ["火", "水", "心", "皿", "足", "土"]

    def _resolve_class(self, origin_char: str) -> Optional[int]:
        return RADICAL_TO_CLASS.get(origin_char)

    def _append_leaf_bbox(self, leaf: RadicalLeaf, annotations: List[BBoxAnnotation]) -> None:
        cid = self._resolve_class(leaf.origin_char)
        if cid is not None:
            annotations.append(BBoxAnnotation(
                class_id=cid,
                xc=(leaf.target_box.x + leaf.target_box.width / 2.0) / self.imgsz,
                yc=(leaf.target_box.y + leaf.target_box.height / 2.0) / self.imgsz,
                w=leaf.target_box.width / self.imgsz,
                h=leaf.target_box.height / self.imgsz
            ))

    def synthesize_character(
        self,
        primary_radical: str,
        augment: bool = True
    ) -> Tuple[Image.Image, List[BBoxAnnotation]]:
        imgsz = self.imgsz
        margin = imgsz * random.uniform(0.05, 0.09)
        usable = imgsz - 2 * margin

        op_roll = random.random()
        # 18% датасета — сверхсложные фрактальные иероглифы и даосские лигатуры (5..10 ключей, 50..172 черты)
        if op_roll >= 0.82:
            op = "HYPER_TALISMAN"
        elif primary_radical in TOP_RADICALS or primary_radical in BOTTOM_RADICALS:
            op = "⿱"
        elif primary_radical in RIGHT_RADICALS:
            op = "⿰"
        elif any(primary_radical in ENCLOSURE_OUTERS.get(k, []) for k in ENCLOSURE_OUTERS):
            op = next(k for k, rad_list in ENCLOSURE_OUTERS.items() if primary_radical in rad_list)
        elif op_roll < 0.34:
            op = "⿰"
        elif op_roll < 0.60:
            op = "⿱"
        elif op_roll < 0.72:
            op = random.choice(["⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺", "⿻"])
        elif op_roll < 0.78:
            op = random.choice(["⿲", "⿳"])
        else:
            op = "SINGLE"

        renderer = CalligraphicPillowRenderer(target_resolution=imgsz, supersample_factor=2)
        annotations: List[BBoxAnnotation] = []
        char_box = BoundingBox(margin, margin, usable, usable)

        # 1. Одиночный радикал
        if op == "SINGLE":
            leaf = RadicalLeaf(primary_radical, SpatialSlot.ROOT, self.repo)
            leaf.calculate_geometry(char_box)
            leaf.render(renderer)
            self._append_leaf_bbox(leaf, annotations)

        # 2. СВЕРХСЛОЖНЫЙ ФРАКТАЛЬНЫЙ ИЕРОГЛИФ / ДАОССКИЙ ТАЛИСМАН (5..10 радикалов, 50..172+ черт)
        elif op == "HYPER_TALISMAN":
            # Глобальный оператор трехъярусной вертикали ⿳
            op_cid = 214 + IDC_OPERATORS.index("⿳")
            annotations.append(BBoxAnnotation(
                class_id=op_cid,
                xc=(char_box.x + char_box.width / 2.0) / imgsz,
                yc=(char_box.y + char_box.height / 2.0) / imgsz,
                w=char_box.width / imgsz,
                h=char_box.height / imgsz
            ))

            # Ярус 1 (Небесный покров сверху — 22% высоты)
            top_h = usable * 0.22
            mid_h = usable * 0.52
            bot_h = usable * 0.22
            v_gap = usable * 0.02

            top_rad = primary_radical if primary_radical in TOP_RADICALS else random.choice(self.top_list)
            leaf_top = RadicalLeaf(top_rad, SpatialSlot.TOP, self.repo)
            leaf_top.calculate_geometry(BoundingBox(margin, margin, usable, top_h))
            leaf_top.render(renderer)
            self._append_leaf_bbox(leaf_top, annotations)

            # Ярус 2 (Центральная плотная матрица: 2 или 3 ряда по 2..3 радикала = 4..8 ключей)
            mid_y = margin + top_h + v_gap
            grid_rows = random.choice([2, 2, 3])
            grid_cols = random.choice([2, 3])
            cell_w = (usable - (grid_cols - 1) * (usable * 0.015)) / float(grid_cols)
            cell_h = (mid_h - (grid_rows - 1) * (mid_h * 0.02)) / float(grid_rows)

            placed_primary = (top_rad == primary_radical)
            for r_idx in range(grid_rows):
                for c_idx in range(grid_cols):
                    if not placed_primary:
                        sub_rad = primary_radical
                        placed_primary = True
                    else:
                        sub_rad = random.choice(self.all_radicals)

                    cx = margin + c_idx * (cell_w + usable * 0.015)
                    cy = mid_y + r_idx * (cell_h + mid_h * 0.02)
                    slot = SpatialSlot.LEFT if c_idx == 0 else SpatialSlot.RIGHT
                    lf = RadicalLeaf(sub_rad, slot, self.repo)
                    lf.calculate_geometry(BoundingBox(cx, cy, cell_w, cell_h))
                    lf.render(renderer)
                    self._append_leaf_bbox(lf, annotations)

            # Ярус 3 (Земное основание снизу — 1 или 2 ключа)
            bot_y = mid_y + mid_h + v_gap
            bot_rad = random.choice(self.bottom_list)
            leaf_bot = RadicalLeaf(bot_rad, SpatialSlot.BOTTOM, self.repo)
            leaf_bot.calculate_geometry(BoundingBox(margin, bot_y, usable, bot_h))
            leaf_bot.render(renderer)
            self._append_leaf_bbox(leaf_bot, annotations)

        # 3. Горизонтальное разделение (⿰)
        elif op == "⿰":
            op_cid = 214 + IDC_OPERATORS.index("⿰")
            annotations.append(BBoxAnnotation(
                class_id=op_cid,
                xc=(char_box.x + char_box.width / 2.0) / imgsz,
                yc=(char_box.y + char_box.height / 2.0) / imgsz,
                w=char_box.width / imgsz,
                h=char_box.height / imgsz
            ))

            if primary_radical in RIGHT_RADICALS:
                r1_char, r2_char = random.choice(self.all_radicals), primary_radical
            else:
                r1_char, r2_char = primary_radical, random.choice(self.all_radicals)

            leaf1 = RadicalLeaf(r1_char, SpatialSlot.LEFT, self.repo)
            leaf2 = RadicalLeaf(r2_char, SpatialSlot.RIGHT, self.repo)

            split = random.uniform(0.30, 0.45)
            gap = usable * random.uniform(0.02, 0.05)
            w1 = (usable - gap) * split
            w2 = usable - w1 - gap

            leaf1.calculate_geometry(BoundingBox(margin, margin, w1, usable))
            leaf2.calculate_geometry(BoundingBox(margin + w1 + gap, margin, w2, usable))
            leaf1.render(renderer)
            leaf2.render(renderer)
            self._append_leaf_bbox(leaf1, annotations)
            self._append_leaf_bbox(leaf2, annotations)

        # 4. Вертикальное разделение (⿱)
        elif op == "⿱":
            op_cid = 214 + IDC_OPERATORS.index("⿱")
            annotations.append(BBoxAnnotation(
                class_id=op_cid,
                xc=(char_box.x + char_box.width / 2.0) / imgsz,
                yc=(char_box.y + char_box.height / 2.0) / imgsz,
                w=char_box.width / imgsz,
                h=char_box.height / imgsz
            ))

            if primary_radical in BOTTOM_RADICALS:
                r1_char, r2_char = random.choice(self.all_radicals), primary_radical
            else:
                r1_char, r2_char = primary_radical, random.choice(self.all_radicals)

            leaf1 = RadicalLeaf(r1_char, SpatialSlot.TOP, self.repo)
            leaf2 = RadicalLeaf(r2_char, SpatialSlot.BOTTOM, self.repo)

            split = random.uniform(0.32, 0.46)
            gap = usable * random.uniform(0.02, 0.05)
            h1 = (usable - gap) * split
            h2 = usable - h1 - gap

            leaf1.calculate_geometry(BoundingBox(margin, margin, usable, h1))
            leaf2.calculate_geometry(BoundingBox(margin, margin + h1 + gap, usable, h2))
            leaf1.render(renderer)
            leaf2.render(renderer)
            self._append_leaf_bbox(leaf1, annotations)
            self._append_leaf_bbox(leaf2, annotations)

        # 5. Объемлющие структуры (⿴, ⿵, ⿶, ⿷, ⿸, ⿹, ⿺, ⿻)
        elif op in ENCLOSURE_OUTERS:
            op_cid = 214 + IDC_OPERATORS.index(op)
            annotations.append(BBoxAnnotation(
                class_id=op_cid,
                xc=(char_box.x + char_box.width / 2.0) / imgsz,
                yc=(char_box.y + char_box.height / 2.0) / imgsz,
                w=char_box.width / imgsz,
                h=char_box.height / imgsz
            ))

            outer_candidates = ENCLOSURE_OUTERS[op]
            outer_char = primary_radical if primary_radical in outer_candidates else random.choice(outer_candidates)
            inner_char = primary_radical if outer_char != primary_radical else random.choice([r for r in self.all_radicals if r != outer_char])

            out_slot = SpatialSlot.SURROUND_BL if op == "⿺" else SpatialSlot.ROOT
            leaf_out = RadicalLeaf(outer_char, out_slot, self.repo)
            leaf_in = RadicalLeaf(inner_char, SpatialSlot.INNER, self.repo)
            leaf_out.calculate_geometry(char_box)

            if op in ("⿴", "⿻"):
                in_box = BoundingBox(char_box.x + usable * 0.22, char_box.y + usable * 0.22, usable * 0.56, usable * 0.56)
            elif op == "⿵":
                in_box = BoundingBox(char_box.x + usable * 0.20, char_box.y + usable * 0.30, usable * 0.60, usable * 0.64)
            elif op == "⿶":
                in_box = BoundingBox(char_box.x + usable * 0.20, char_box.y + usable * 0.10, usable * 0.60, usable * 0.64)
            elif op == "⿷":
                in_box = BoundingBox(char_box.x + usable * 0.28, char_box.y + usable * 0.18, usable * 0.62, usable * 0.64)
            elif op == "⿸":
                in_box = BoundingBox(char_box.x + usable * 0.32, char_box.y + usable * 0.32, usable * 0.62, usable * 0.62)
            elif op == "⿹":
                in_box = BoundingBox(char_box.x + usable * 0.08, char_box.y + usable * 0.30, usable * 0.64, usable * 0.64)
            else:
                in_box = BoundingBox(char_box.x + usable * 0.30, char_box.y + usable * 0.06, usable * 0.64, usable * 0.64)

            leaf_in.calculate_geometry(in_box)
            leaf_out.render(renderer)
            leaf_in.render(renderer)
            self._append_leaf_bbox(leaf_out, annotations)
            self._append_leaf_bbox(leaf_in, annotations)

        # 6. Тройные структуры (⿲, ⿳)
        else:
            is_horiz = (op == "⿲")
            op_cid = 214 + IDC_OPERATORS.index("⿲" if is_horiz else "⿳")
            annotations.append(BBoxAnnotation(
                class_id=op_cid,
                xc=(char_box.x + char_box.width / 2.0) / imgsz,
                yc=(char_box.y + char_box.height / 2.0) / imgsz,
                w=char_box.width / imgsz,
                h=char_box.height / imgsz
            ))

            r_chars = [primary_radical, random.choice(self.all_radicals), random.choice(self.all_radicals)]
            gap = usable * 0.03
            unit_size = (usable - 2 * gap) / 3.0

            for i, r_ch in enumerate(r_chars):
                leaf = RadicalLeaf(r_ch, SpatialSlot.ROOT, self.repo)
                if is_horiz:
                    b = BoundingBox(margin + i * (unit_size + gap), margin, unit_size, usable)
                else:
                    b = BoundingBox(margin, margin + i * (unit_size + gap), usable, unit_size)
                leaf.calculate_geometry(b)
                leaf.render(renderer)
                self._append_leaf_bbox(leaf, annotations)

        final_img = renderer.finalize().convert("RGB")

        if augment:
            try:
                from cangjie_generative_degradation import (
                    XuanPaperInkDiffusionSolver,
                    StoneRubbingTopianSimulator,
                    SubtractiveCinnabarSealOverlay,
                    ThinPlateSplineDeformer
                )
                from cangjie_archaic_epigraphy_engine import ArchaicJiaguwenJinwenSynthesizer
                roll_deg = random.random()
                gray_arr = np.array(final_img.convert("L"), dtype=np.float32)
                ink_density = np.clip((255.0 - gray_arr) / 255.0, 0.0, 1.0)
                if roll_deg < 0.18 and annotations:
                    # До-ханьская архаика: гадательные кости Шан (Цзягувэнь 甲骨文) или бронза Чжоу (Цзиньвэнь 金文)
                    era = "SHANG_JIAGUWEN" if roll_deg < 0.09 else "ZHOU_JINWEN"
                    arch_synth = ArchaicJiaguwenJinwenSynthesizer(repo=self.repo, resolution=imgsz)
                    arch_comps = [
                        (CLASS_TO_GLYPH.get(a.class_id, primary_radical),
                         BoundingBox((a.xc - 0.5*a.w)*imgsz, (a.yc - 0.5*a.h)*imgsz, a.w*imgsz, a.h*imgsz))
                        for a in annotations if a.class_id < 214
                    ]
                    if arch_comps:
                        arch_out = arch_synth.render_archaic_inscription(arch_comps, era_mode=era, seed=random.randint(1, 9999))
                        rgb_deg = arch_out["archaic_rgb_array"]
                    else:
                        rgb_deg = StoneRubbingTopianSimulator.synthesize_topian_rubbing(ink_density, seed=random.randint(1, 9999))
                elif roll_deg < 0.35:
                    # Каменный эстампаж топянь (拓片) с инверсным контрастом и шумом Перлина
                    rgb_deg = StoneRubbingTopianSimulator.synthesize_topian_rubbing(ink_density, seed=random.randint(1, 9999))
                else:
                    # Нестационарная анизотропная PDE-диффузия туши по бумаге Сюаньчжи / Хуанбяочжи
                    pde_solver = XuanPaperInkDiffusionSolver(num_steps=6, dt=0.40)
                    diffused = pde_solver.solve_diffusion(ink_density)
                    paper_c = np.array(random.choice(ChineseInkAugmentor.PAPERS), dtype=np.float32)
                    rgb_deg = np.clip(paper_c[None, None, :] * (1.0 - 0.90 * diffused[:, :, None]), 0.0, 255.0).astype(np.uint8)
                    if random.random() < 0.35:
                        rgb_deg = SubtractiveCinnabarSealOverlay.apply_cinnabar_seal(rgb_deg, seed=random.randint(1, 9999))

                # Нелинейная деформация тонких пластин (TPS) с аналитическим пересчетом всех BBox
                if annotations and random.random() < 0.50:
                    boxes_xyxy = np.array([
                        [a.xc - 0.5 * a.w, a.yc - 0.5 * a.h, a.xc + 0.5 * a.w, a.yc + 0.5 * a.h]
                        for a in annotations
                    ], dtype=np.float64)
                    tps = ThinPlateSplineDeformer(grid_size=4, max_Clean_shift=0.025, seed=random.randint(1, 9999))
                    rgb_deg, warped_xyxy = tps.warp_image_and_bboxes(rgb_deg, boxes_xyxy)
                    for idx_a, a in enumerate(annotations):
                        wx1, wy1, wx2, wy2 = warped_xyxy[idx_a]
                        a.xc = float(0.5 * (wx1 + wx2))
                        a.yc = float(0.5 * (wy1 + wy2))
                        a.w = float(max(1e-4, wx2 - wx1))
                        a.h = float(max(1e-4, wy2 - wy1))

                final_img = Image.fromarray(rgb_deg)
            except Exception:
                paper_color = random.choice(ChineseInkAugmentor.PAPERS)
                if paper_color != (255, 255, 255):
                    bg = Image.new("RGB", final_img.size, paper_color)
                    final_img = Image.blend(bg, final_img, alpha=0.92)
                final_img = ChineseInkAugmentor.apply_paper_texture(final_img, noise_std=random.uniform(3.0, 6.0))
                final_img = ChineseInkAugmentor.apply_ink_bleed(final_img, p=0.30)

        return final_img, annotations


def _generate_chunk_worker(
    task_batch: List[Tuple[int, str, str, str]],
    imgsz: int,
    augment: bool
) -> int:
    repo = GlyphRepository()
    synthesizer = CompositionGeometrySynthesizer(repo=repo, imgsz=imgsz)
    completed = 0

    for sample_idx, split_name, primary_rad, out_dir in task_batch:
        try:
            img, annotations = synthesizer.synthesize_character(
                primary_radical=primary_rad,
                augment=augment
            )

            img_filename = f"cangjie_{split_name}_{sample_idx:07d}.png"
            lbl_filename = f"cangjie_{split_name}_{sample_idx:07d}.txt"

            img_path = os.path.join(out_dir, "images", split_name, img_filename)
            lbl_path = os.path.join(out_dir, "labels", split_name, lbl_filename)

            img.save(img_path, format="PNG")
            with open(lbl_path, "w", encoding="utf-8") as f_lbl:
                for ann in annotations:
                    f_lbl.write(ann.to_yolo_line() + "\n")

            completed += 1
        except Exception as e:
            logger.warning(f"Ошибка генерации сэмпла {sample_idx} ({e})")

    return completed


class CangjieYOLODataFactory:
    def __init__(
        self,
        output_dir: str = "./cangjie_yolo_dataset",
        imgsz: int = 640,
        val_ratio: float = 0.15,
        num_workers: Optional[int] = None
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.imgsz = imgsz
        self.val_ratio = val_ratio
        self.num_workers = num_workers or min(os.cpu_count() or 4, 64)

        self.images_train_dir = os.path.join(self.output_dir, "images", "train")
        self.images_val_dir = os.path.join(self.output_dir, "images", "val")
        self.labels_train_dir = os.path.join(self.output_dir, "labels", "train")
        self.labels_val_dir = os.path.join(self.output_dir, "labels", "val")

        self._create_directories()

    def _create_directories(self) -> None:
        for d in [self.images_train_dir, self.images_val_dir, self.labels_train_dir, self.labels_val_dir]:
            os.makedirs(d, exist_ok=True)

    def generate_data_yaml(self) -> str:
        yaml_path = os.path.join(self.output_dir, "data.yaml")
        yaml_content = {
            "path": self.output_dir,
            "train": "images/train",
            "val": "images/val",
            "nc": len(CLASS_NAMES),
            "names": {i: name for i, name in enumerate(CLASS_NAMES)}
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            if HAS_YAML and yaml is not None:
                yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)
            else:
                f.write(f"path: {self.output_dir}\ntrain: images/train\nval: images/val\nnc: {len(CLASS_NAMES)}\n")

        logger.info(f"Конфигурация dataset data.yaml сохранена: {yaml_path} (Классов: {len(CLASS_NAMES)})")
        return yaml_path

    def run_synthesis(self, total_count: int = 1000, augment: bool = True) -> None:
        logger.info(f"Запуск генерации датасета v4.0: {total_count} изображений (Потоков: {self.num_workers}, ImgSz: {self.imgsz})...")
        t_start = time.time()

        val_count = int(total_count * self.val_ratio)
        all_base_radicals = [item[1] for item in KANGXI_214_COMPREHENSIVE_CATALOG]
        tasks = []

        for idx in range(total_count):
            split = "val" if idx < val_count else "train"
            rad = all_base_radicals[idx % len(all_base_radicals)]
            tasks.append((idx + 1, split, rad, self.output_dir))

        random.shuffle(tasks)

        chunk_size = max(10, total_count // (self.num_workers * 4))
        chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

        generated_total = 0
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(_generate_chunk_worker, chunk, self.imgsz, augment)
                for chunk in chunks
            ]

            for future in as_completed(futures):
                count = future.result()
                generated_total += count
                elapsed = time.time() - t_start
                speed = generated_total / max(0.1, elapsed)
                percent = (generated_total / total_count) * 100.0
                print(
                    f"\r[ГЕНЕРАЦИЯ v4.0]: {generated_total}/{total_count} ({percent:.1f}%) | "
                    f"Скорость: {speed:.1f} изобр/сек | Прошло: {elapsed:.1f}с",
                    end="",
                    flush=True
                )

        print("")
        elapsed_total = time.time() - t_start
        logger.info(
            f"ГЕНЕРАЦИЯ ЗАВЕРШЕНА! Создано: {generated_total} пар (PNG + TXT) за {elapsed_total:.1f} сек."
        )
        self.generate_data_yaml()


def main():
    parser = argparse.ArgumentParser(description="Project Cangjie Enterprise YOLO/RT-DETR Data Factory [v4.0]")
    parser.add_argument("--count", type=int, default=1000, help="Количество генерируемых изображений")
    parser.add_argument("--output", type=str, default="./cangjie_yolo_dataset", help="Путь сохранения датасета")
    parser.add_argument("--imgsz", type=int, default=640, help="Разрешение картинок (по умолчанию 640)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Доля валидационной выборки")
    parser.add_argument("--workers", type=int, default=None, help="Количество процессов CPU")
    parser.add_argument("--no-augment", action="store_true", help="Отключить каллиграфические аугментации")
    args = parser.parse_args()

    factory = CangjieYOLODataFactory(
        output_dir=args.output,
        imgsz=args.imgsz,
        val_ratio=args.val_ratio,
        num_workers=args.workers
    )
    factory.run_synthesis(
        total_count=args.count,
        augment=not args.no_augment
    )


if __name__ == "__main__":
    main()
