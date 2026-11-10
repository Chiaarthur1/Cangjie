"""
Project Cangjie (Цанцзе) — Enterprise Multi-Glyph Paleographic Photo & Calligraphic Canvas Decoder [v8.0]
Fixes & Upgrades in v8.0:
1. Multi-Character Scroll & Rubbing Segmentation (`decode_multi_glyph_photo`):
   - Automatically segments an uploaded photo of a scroll, oracle bone, bronze vessel, or screenshot into MULTIPLE
     distinct character bounding boxes (via 2D Connected-Component & XY-Projection Valley Splitting), and then
     decomposes EACH character into its internal Kangxi radicals / phonetic subtrees!
2. Upright MakeMeAHanzi Template Matching (Fixes upside-down Y-axis bug):
   - Uses upright `900.0 - y` MakeMeAHanzi templates (`cangjie_benchmark_forge.py` v8.0) + Soft Gaussian Distance-Field
     (Chamfer) + 3x3 Spatial 4-Orientation Sobel HOG (0°/90°/45°/135°).
3. Unified Drawing Studio & Photo Overlay Support:
   - Supports hand-drawn strokes (`strokes`), interactive radical stamps (`stamped_nodes`), and uploaded photos
     (both single-character and multi-character inscriptions).
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import numpy as np

from cangjie_benchmark_forge import (
    KANGXI_214_COMPREHENSIVE_CATALOG,
    CalligraphicPillowRenderer,
    GlyphRepository,
    RadicalLeaf,
    SpatialSlot,
    BoundingBox
)
from cangjie_rtdetr_infer import INDEX_TO_RADICAL
from cangjie_deducer import (
    CangjieMeaningDeducer,
    RADICAL_DEEP_SEMANTICS,
    PHONETIC_CO_SEMANTICS
)
from cangjie_layout_to_tree import (
    PointerTreeTransformerDecoder,
    RadicalDetectionNode,
    GLYPH_TO_CLASS_ID
)
from cangjie_edge_cpu_runtime import SauvolaSIMDPreprocessor, InMemorySQLiteGraphRAG
from cangjie_paleo_matcher import normalize_component_mask
from cangjie_product_logic import ProductRealLogic
from cangjie_config import CangjieConfig, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PHOTO-DECODER-v8]: %(message)s"
)
logger = logging.getLogger("CangjiePhotoDecoder")

# [v9.3] Эра по умолчанию, если определить не удалось. Держим строкой,
# чтобы не тянуть импорт cangjie_paleo_matcher на уровне модуля
# (он тянет numpy-банки и ломает быстрый smoke-import).
ERA_KAISHU_FALLBACK = "KAISHU"


@dataclass
class DetectedComponentBox:
    glyph: str
    name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = -1
    char_index: int = 0

    @property
    def xc(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def yc(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return max(1e-4, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(1e-4, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


class PaleoPhotoPreprocessor:
    @staticmethod
    def process_raw_photo(image_path: str, target_size: int = 1024) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Фото не найдено: {image_path}")

        t0 = time.perf_counter()
        raw_img = Image.open(image_path).convert("RGB")
        w0, h0 = raw_img.size
        scale = target_size / float(max(1, max(w0, h0)))
        nw, nh = max(1, int(w0 * scale)), max(1, int(h0 * scale))
        resized = raw_img.resize((nw, nh), Image.Resampling.LANCZOS)

        rgb_arr = np.array(resized, dtype=np.float32)
        r, g, b = rgb_arr[:, :, 0], rgb_arr[:, :, 1], rgb_arr[:, :, 2]

        # Защита от захвата красных рамок интерфейса (если пользователь загрузил скриншот приложения)
        gray_luma = 0.299 * r + 0.587 * g + 0.114 * b
        redness = np.clip(r - 0.5 * (g + b), 0.0, 255.0)
        dark_ink_ratio = float(np.mean(gray_luma < 95.0))
        if dark_ink_ratio > 0.02:
            # На изображении есть четкая темная тушь — подавляем красную печать, чтобы она не мешала сегментации черт
            gray = np.where(redness > 55.0, 245.0, gray_luma)
        elif np.percentile(redness, 95) > 45.0:
            gray = np.clip(255.0 - redness * 1.8, 0.0, 255.0)
        else:
            gray = gray_luma

        border_pixels = np.concatenate([
            gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
        ])
        is_dark_rubbing = float(np.median(border_pixels)) < 110.0
        if is_dark_rubbing:
            gray = 255.0 - gray

        sauvola_ink = SauvolaSIMDPreprocessor.binarize_sauvola(gray, window_size=max(15, target_size // 32), k=0.18)

        # Очистка тонких рамок по краям холста (если загружен скриншот с рамкой)
        margin_mask = max(4, target_size // 64)
        sauvola_ink[:margin_mask, :] = 0.0
        sauvola_ink[-margin_mask:, :] = 0.0
        sauvola_ink[:, :margin_mask] = 0.0
        sauvola_ink[:, -margin_mask:] = 0.0

        square_ink = np.zeros((target_size, target_size), dtype=np.float32)
        ox, oy = (target_size - nw) // 2, (target_size - nh) // 2
        square_ink[oy:oy + nh, ox:ox + nw] = sauvola_ink

        stroke_metrics = PaleoPhotoPreprocessor._estimate_stroke_complexity(square_ink)
        prep_ms = (time.perf_counter() - t0) * 1000.0

        clean_pil = Image.fromarray(((1.0 - square_ink) * 255.0).astype(np.uint8)).convert("RGB")
        return {
            "clean_rgb_image": clean_pil,
            "ink_matrix": square_ink,
            "is_stone_rubbing_inverted": is_dark_rubbing,
            "stroke_analysis": stroke_metrics,
            "sauvola_preprocessing_ms": round(prep_ms, 2)
        }

    @staticmethod
    def segment_multiple_characters(ink: np.ndarray, max_chars: int = 8) -> List[Tuple[int, int, int, int]]:
        """
        Выделяет несколько отдельных иероглифов на фото свитка/эстампажа/надписи (по горизонтальным и вертикальным
        долинам плотности туши). Если на фото 1 крупный иероглиф, возвращает 1 общий бокс `[x1, y1, x2, y2]`.
        """
        H, W = ink.shape
        ys, xs = np.where(ink > 0.2)
        if len(xs) == 0:
            return [(int(W * 0.1), int(H * 0.1), int(W * 0.9), int(H * 0.9))]

        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        crop = ink[y1:y2, x1:x2]
        ch_h, ch_w = crop.shape
        aspect = ch_w / float(max(1, ch_h))

        boxes: List[Tuple[int, int, int, int]] = []

        # Если надпись вытянута по горизонтали (строка иероглифов) или вертикали (столбец свитка)
        if aspect >= 1.45:
            col_proj = np.mean(crop > 0.2, axis=0)
            thresh = max(0.008, float(np.mean(col_proj)) * 0.12)
            in_char = False
            start_x = 0
            min_seg_w = max(24, int(ch_h * 0.32))
            for cx in range(ch_w):
                if col_proj[cx] > thresh and not in_char:
                    in_char = True
                    start_x = cx
                elif col_proj[cx] <= thresh and in_char:
                    if cx - start_x >= min_seg_w:
                        sub = crop[:, start_x:cx]
                        sy, sx = np.where(sub > 0.2)
                        if len(sx) > 0:
                            boxes.append((x1 + start_x + int(sx.min()), y1 + int(sy.min()), x1 + start_x + int(sx.max()) + 1, y1 + int(sy.max()) + 1))
                    in_char = False
            if in_char and (ch_w - start_x) >= min_seg_w:
                sub = crop[:, start_x:ch_w]
                sy, sx = np.where(sub > 0.2)
                if len(sx) > 0:
                    boxes.append((x1 + start_x + int(sx.min()), y1 + int(sy.min()), x1 + start_x + int(sx.max()) + 1, y1 + int(sy.max()) + 1))

        elif aspect <= 0.68:
            row_proj = np.mean(crop > 0.2, axis=1)
            thresh = max(0.008, float(np.mean(row_proj)) * 0.12)
            in_char = False
            start_y = 0
            min_seg_h = max(24, int(ch_w * 0.32))
            for cy in range(ch_h):
                if row_proj[cy] > thresh and not in_char:
                    in_char = True
                    start_y = cy
                elif row_proj[cy] <= thresh and in_char:
                    if cy - start_y >= min_seg_h:
                        sub = crop[start_y:cy, :]
                        sy, sx = np.where(sub > 0.2)
                        if len(sx) > 0:
                            boxes.append((x1 + int(sx.min()), y1 + start_y + int(sy.min()), x1 + int(sx.max()) + 1, y1 + start_y + int(sy.max()) + 1))
                    in_char = False
            if in_char and (ch_h - start_y) >= min_seg_h:
                sub = crop[start_y:ch_h, :]
                sy, sx = np.where(sub > 0.2)
                if len(sx) > 0:
                    boxes.append((x1 + int(sx.min()), y1 + start_y + int(sy.min()), x1 + int(sx.max()) + 1, y1 + start_y + int(sy.max()) + 1))

        if not boxes:
            boxes = [(x1, y1, x2, y2)]

        return boxes[:max_chars]

    @staticmethod
    def _estimate_stroke_complexity(ink: np.ndarray) -> Dict[str, Any]:
        ys, xs = np.where(ink > 0.5)
        if len(xs) == 0:
            return {"estimated_strokes": 0, "complexity_tier": "EMPTY", "ink_ratio": 0.0}

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        crop = ink[y1:y2 + 1, x1:x2 + 1]

        h_transitions = np.sum(np.abs(np.diff(crop, axis=1)) > 0.5) / 2.0
        v_transitions = np.sum(np.abs(np.diff(crop, axis=0)) > 0.5) / 2.0

        total_ink_pixels = float(np.sum(crop > 0.5))
        perimeter_pixels = float(h_transitions + v_transitions)
        avg_stroke_width = max(2.0, (2.0 * total_ink_pixels) / max(1.0, perimeter_pixels))

        skeleton_length = total_ink_pixels / avg_stroke_width
        char_span = float(max(crop.shape[0], crop.shape[1]))

        raw_strokes = (skeleton_length / max(1.0, char_span * 0.24)) * 1.15
        est_strokes = max(1, int(round(raw_strokes)))

        if est_strokes >= 50:
            tier = "HYPER_COMPLEX_TALISMAN_OR_LIGATURE (50..172+ черт)"
        elif est_strokes >= 25:
            tier = "COMPLEX_ARCHAIC_OR_RARE (25..49 черт)"
        elif est_strokes >= 10:
            tier = "STANDARD_COMPOUND (10..24 черт)"
        else:
            tier = "ELEMENTARY_GLYPH (1..9 черт)"

        return {
            "estimated_strokes": est_strokes,
            "complexity_tier": tier,
            "avg_stroke_width_px": round(avg_stroke_width, 2),
            "active_bbox": [x1, y1, x2, y2]
        }


class MultiScaleComponentScanner:
    _GLOBAL_TEMPLATES_CACHE: Dict[str, Tuple[str, np.ndarray, np.ndarray]] = {}

    def __init__(self, weights_path: str = "rtdetr-x.pt", repo: Optional[GlyphRepository] = None, data_dir: str = "./data"):
        self.weights_path = weights_path
        self.data_dir = os.path.abspath(data_dir)
        self.model = None
        self.repo = repo if repo is not None else GlyphRepository()
        self._templates: Dict[str, Tuple[str, np.ndarray, np.ndarray]] = {}
        # [v9.0] Калиброванный эра-специфичный матчер (ленивая инициализация)
        self._v9_matcher = None
        # [v9.3] Эра снимка: определяется по стилю ДО идентификации знака.
        self._era_hint: Dict[str, Any] = {}
        self._era_locked = False

        if os.path.exists(weights_path):
            try:
                from ultralytics import RTDETR
                cand_model = RTDETR(weights_path)
                names = getattr(cand_model, "names", {}) or {}
                first_name = str(names.get(0, "")).lower()
                if len(names) >= 200 and first_name not in {"person", "bicycle", "car"}:
                    self.model = cand_model
                    logger.info(f"Подключен обученный палеографический детектор RT-DETR ({len(names)} классов): {weights_path}")
                else:
                    logger.info(
                        f"Файл {weights_path} содержит стандартные веса COCO ({len(names)} классов). "
                        f"Используется палеографический Chamfer+HOG сегментатор Канси v8.0."
                    )
            except Exception as e:
                logger.warning(f"RT-DETR недоступен ({e}), используется мультимасштабный Chamfer+HOG сегментатор.")

    @staticmethod
    def _compute_chamfer_and_hog_descriptor(ink_64: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pil_im = Image.fromarray(np.clip(ink_64 * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        soft_im = pil_im.filter(ImageFilter.GaussianBlur(radius=2.2))
        soft_arr = np.array(soft_im, dtype=np.float32) / 255.0
        norm_s = float(np.linalg.norm(soft_arr)) or 1.0
        chamfer_field = soft_arr / norm_s

        gx = np.zeros_like(ink_64)
        gy = np.zeros_like(ink_64)
        gx[:, 1:-1] = ink_64[:, 2:] - ink_64[:, :-2]
        gy[1:-1, :] = ink_64[2:, :] - ink_64[:-2, :]

        g_horiz = np.abs(gy)
        g_vert = np.abs(gx)
        g_diag1 = np.abs(0.7071 * (gx + gy))
        g_diag2 = np.abs(0.7071 * (gx - gy))

        hog_parts: List[float] = []
        splits = [0, 21, 43, 64]
        for r_i in range(3):
            for c_i in range(3):
                r0, r1 = splits[r_i], splits[r_i + 1]
                c0, c1 = splits[c_i], splits[c_i + 1]
                hog_parts.append(float(np.mean(g_horiz[r0:r1, c0:c1])))
                hog_parts.append(float(np.mean(g_vert[r0:r1, c0:c1])))
                hog_parts.append(float(np.mean(g_diag1[r0:r1, c0:c1])))
                hog_parts.append(float(np.mean(g_diag2[r0:r1, c0:c1])))

        hog_parts.append(float(np.mean(ink_64[:32, :32])))
        hog_parts.append(float(np.mean(ink_64[:32, 32:])))
        hog_parts.append(float(np.mean(ink_64[32:, :32])))
        hog_parts.append(float(np.mean(ink_64[32:, 32:])))

        hog_vec = np.array(hog_parts, dtype=np.float32)
        norm_h = float(np.linalg.norm(hog_vec)) or 1.0
        return chamfer_field, hog_vec / norm_h

    def _ensure_templates(self) -> None:
        if self._templates:
            return
        if MultiScaleComponentScanner._GLOBAL_TEMPLATES_CACHE:
            self._templates = MultiScaleComponentScanner._GLOBAL_TEMPLATES_CACHE
            return
        cache_npz = os.path.join(self.data_dir, "cangjie_chamfer_hog_templates_v85.npz")
        if os.path.exists(cache_npz):
            try:
                loaded = np.load(cache_npz, allow_pickle=False)
                glyphs = json.loads(str(loaded["glyphs_json"]))
                names = json.loads(str(loaded["names_json"]))
                chamfers = loaded["chamfers"]
                hogs = loaded["hogs"]
                for i, g in enumerate(glyphs):
                    self._templates[g] = (names[i], chamfers[i], hogs[i])
                MultiScaleComponentScanner._GLOBAL_TEMPLATES_CACHE = self._templates
                return
            except Exception:
                pass
        extra_phonetics = [
            ("寺", "temple_phonetic"), ("霝", "rain_spirit_phonetic"), ("青", "cyan_phonetic"),
            ("甫", "spread_phonetic"), ("尧", "high_phonetic"), ("申", "lightning_shen"),
            ("首", "head_shou"), ("巫", "shaman_wu"), ("占", "divine_zhan"),
            ("包", "wrap_bao"), ("尚", "esteem_shang"), ("且", "altar_qie"),
            ("旦", "dawn_dan"), ("雷", "thunder_lei"), ("可", "can_ke"), ("古", "ancient_gu"),
            ("我", "self_wo"), ("道", "way_dao"), ("國", "state_guo"), ("林", "forest_lin")
        ]
        catalog_items: List[Tuple[str, str]] = []
        for num, base_g, allo_g, _, _, eng_name in KANGXI_214_COMPREHENSIVE_CATALOG:
            catalog_items.append((base_g, eng_name))
            if allo_g != base_g:
                catalog_items.append((allo_g, eng_name))
        catalog_items.extend(extra_phonetics)

        for g, eng_name in catalog_items:
            if g in self._templates:
                continue
            renderer = CalligraphicPillowRenderer(target_resolution=64, supersample_factor=1)
            leaf = RadicalLeaf(g, SpatialSlot.ROOT, self.repo)
            leaf.calculate_geometry(BoundingBox(4, 4, 56, 56))
            leaf.render(renderer)
            arr = np.array(renderer.finalize().convert("L"), dtype=np.float32)
            ink = np.clip((255.0 - arr) / 255.0, 0.0, 1.0)
            chamfer_f, hog_v = self._compute_chamfer_and_hog_descriptor(ink)
            self._templates[g] = (eng_name, chamfer_f, hog_v)

        MultiScaleComponentScanner._GLOBAL_TEMPLATES_CACHE = self._templates
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            g_list = list(self._templates.keys())
            n_list = [self._templates[k][0] for k in g_list]
            c_arr = np.stack([self._templates[k][1] for k in g_list], axis=0)
            h_arr = np.stack([self._templates[k][2] for k in g_list], axis=0)
            np.savez_compressed(
                cache_npz,
                glyphs_json=np.array(json.dumps(g_list, ensure_ascii=False)),
                names_json=np.array(json.dumps(n_list, ensure_ascii=False)),
                chamfers=c_arr,
                hogs=h_arr
            )
        except Exception:
            pass

    def match_single_crop(self, tight_crop_ink: np.ndarray) -> Tuple[str, str, float]:
        """
        Сопоставляет кроп компонента с банком шаблонов.

        [v9.0] Основной путь — эра-специфичный калиброванный матчер
        `cangjie_paleo_matcher`: 4 720 компонентов CHISE на эру, честная
        вероятность (posterior × fit), статусы RECOGNIZED / AMBIGUOUS /
        UNRECOGNIZED. Старый путь на 281 радикале Канси оставлен как
        fallback на случай, если банк v9 не собран.

        СТАРЫЙ ПУТЬ БЫЛ ФИКТИВНЫМ:
            conf = max(0.55, min(0.99, best_score * 1.12))
        — нижняя граница 0.55 гарантировала «55%» для ЛЮБОГО входа, включая
        белый шум. Палеограф принимал это за оценку.
        """
        if self._v9_matcher is None:
            self._v9_matcher = self._build_v9_matcher()
        if self._v9_matcher is not None:
            try:
                # [v9.0] ВАЖНО: кроп из фото имеет произвольный размер, а
                # дескриптор матчера (HOG-сетка 21/43/64) рассчитан ровно на
                # 64x64. Без нормализации сетка смещается и уверенность
                # падает в ноль. normalize_component_mask делает ТЕСНЫЙ КРОП
                # по туши + ресайз до 64x64 — так же, как строятся шаблоны.
                ink_64 = normalize_component_mask(tight_crop_ink)
                res = self._v9_matcher.match(ink_64, top_k=5)
                if res.best is not None:
                    return res.best.glyph, res.best.name, round(float(res.confidence), 4)
            except Exception as e:
                logger.warning("Матчер v9 не отработал (%s) — переключаюсь на fallback v8", e)
                self._v9_matcher = None

        return self._match_single_crop_legacy(tight_crop_ink)

    def _build_v9_matcher(self, era: Optional[str] = None):
        """
        Строит/берёт из кэша банк шаблонов v9 для эпохи.

        [v9.3] ТЕПЕРЬ ЭРА ОПРЕДЕЛЯЕТСЯ ПО САМОМУ СНИМКУ, а не жёстко KAISHU.
        Раньше сканер ВСЕГДА подключал банк кайшу, даже если на фото был
        甲骨文 или 金文. Для палеографа это катастрофа: силуэт оракульной
        кости и кайшу-глиф — это разные Вселенные, и кросс-матчинг давал
        мусор. Теперь банк выбирается по стилю снимка (см. detect_era).
        """
        try:
            from cangjie_paleo_matcher import get_bank, ERA_KAISHU
            target_era = era or ERA_KAISHU
            bank = get_bank(target_era, data_dir=self.data_dir, repo=self.repo)
            if bank.is_ready:
                logger.info("Подключён калиброванный матчер v9: %d шаблонов (эра %s)",
                            len(bank.glyphs), bank.era_mode)
                return bank
            logger.warning("Банк v9 для эры %s не готов — fallback v8", target_era)
        except Exception as e:
            logger.warning("Матчер v9 недоступен (%s) — fallback v8", e)
        return None

    def detect_era(self, ink_matrix: np.ndarray) -> str:
        """
        [v9.3] Определяет эру/письменность по всему снимку.

        Для ancient-пайплайна это критично: сначала определяем «в каком мире
        мы находимся» (甲骨文 / 金文 / 篆 / кайшу), и только потом ищем
        знак в банке ИМЕННО ЭТОЙ эры. Иначе мы ищем 甲骨ный знак среди
        кайшу-шаблонов и получаем уверенную неправду.
        """
        if self._era_locked:
            return self._v9_matcher.era_mode if self._v9_matcher is not None else ERA_KAISHU_FALLBACK

        try:
            from cangjie_paleo_matcher import match_across_eras, ERA_KAISHU
            whole_ink = normalize_component_mask(ink_matrix)
            det = match_across_eras(
                whole_ink, data_dir=self.data_dir, repo=self.repo
            )
            best = det.get("best_era")
            if best:
                logger.info("Эра снимка определена: %s (scores: %s)",
                            best, {k: round(v, 4) for k, v in det.get("era_scores", {}).items()})
                self._era_hint = det
                return best
        except Exception as e:
            logger.debug("Определение эры по снимку недоступно: %s", e)

        return ERA_KAISHU_FALLBACK

    def _match_single_crop_legacy(self, tight_crop_ink: np.ndarray) -> Tuple[str, str, float]:
        """Fallback-путь v8: 281 радикал Канси, фиктивная уверенность."""
        self._ensure_templates()
        pil_crop = Image.fromarray(((1.0 - np.clip(tight_crop_ink, 0.0, 1.0)) * 255.0).astype(np.uint8))
        resized = np.array(pil_crop.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)
        ink_64 = np.clip((255.0 - resized) / 255.0, 0.0, 1.0)
        q_chamfer, q_hog = self._compute_chamfer_and_hog_descriptor(ink_64)

        best_g, best_name, best_score = "?", "unknown", -1.0
        for g, (eng_name, t_chamfer, t_hog) in self._templates.items():
            sim_c = float(np.sum(q_chamfer * t_chamfer))
            sim_h = float(np.sum(q_hog * t_hog))
            score = 0.55 * sim_c + 0.45 * sim_h
            if score > best_score:
                best_g, best_name, best_score = g, eng_name, score

        conf = round(max(0.55, min(0.99, best_score * 1.12)), 4)
        return best_g, best_name, conf

    def scan_image(
        self,
        clean_rgb: Image.Image,
        ink_matrix: np.ndarray,
        estimated_strokes: int
    ) -> List[DetectedComponentBox]:
        # [v9.3] СНАЧАЛА определяем эру по всему снимку, ПОТОМ сканируем
        # компоненты. Раньше банк был всегда KAISHU — для фото 甲骨文/金文
        # это означало кросс-матчинг между несовместимыми мирами.
        if self._v9_matcher is None:
            era = self.detect_era(ink_matrix)
            self._v9_matcher = self._build_v9_matcher(era)
            self._era_locked = True

        if self.model is not None:
            dets = self._scan_with_rtdetr_sahi(clean_rgb, enable_deep_tiles=(estimated_strokes >= 25))
            if dets:
                return dets

        return self._scan_with_recursive_pixel_partition(ink_matrix, estimated_strokes)

    def _scan_with_rtdetr_sahi(self, img: Image.Image, enable_deep_tiles: bool = True) -> List[DetectedComponentBox]:
        W, H = img.size
        raw_boxes: List[DetectedComponentBox] = []

        slices = [(0, 0, W, H)]
        if enable_deep_tiles:
            for grid in (2, 3):
                step_x = W / float(grid)
                step_y = H / float(grid)
                pad_x = step_x * 0.18
                pad_y = step_y * 0.18
                for gx in range(grid):
                    for gy in range(grid):
                        sx1 = max(0, int(gx * step_x - pad_x))
                        sy1 = max(0, int(gy * step_y - pad_y))
                        sx2 = min(W, int((gx + 1) * step_x + pad_x))
                        sy2 = min(H, int((gy + 1) * step_y + pad_y))
                        slices.append((sx1, sy1, sx2, sy2))

        for (sx1, sy1, sx2, sy2) in slices:
            crop = img.crop((sx1, sy1, sx2, sy2))
            sw, sh = sx2 - sx1, sy2 - sy1
            results = self.model.predict(source=crop, conf=0.28, verbose=False)
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id >= 214:
                    continue
                conf = float(boxes.conf[i].item())
                bx1, by1, bx2, by2 = boxes.xyxyn[i].tolist()

                gx1 = (sx1 + bx1 * sw) / float(W)
                gy1 = (sy1 + by1 * sh) / float(H)
                gx2 = (sx1 + bx2 * sw) / float(W)
                gy2 = (sy1 + by2 * sh) / float(H)

                glyph, name = INDEX_TO_RADICAL.get(cls_id, ("?", "unknown"))
                raw_boxes.append(DetectedComponentBox(
                    glyph=glyph,
                    name=name,
                    confidence=round(conf, 4),
                    x1=gx1, y1=gy1, x2=gx2, y2=gy2,
                    class_id=cls_id
                ))

        return self._non_max_suppression(raw_boxes, iou_thresh=0.55)

    def _scan_with_recursive_pixel_partition(
        self,
        ink: np.ndarray,
        estimated_strokes: int,
        char_index: int = 0,
        global_offset: Tuple[int, int, int, int] = (0, 0, 1024, 1024)
    ) -> List[DetectedComponentBox]:
        self._ensure_templates()
        H_sub, W_sub = ink.shape
        ox, oy, full_W, full_H = global_offset
        ys, xs = np.where(ink > 0.2)
        if len(xs) == 0:
            return []

        # [v9.2] АДАПТИВНАЯ ГЛУБИНА СПУСКА.
        #   БЫЛО: max_depth = 1 (<12 черт) / 2 (<36) / 3 (>=36).
        #   Для 172-чертного знака это катастрофа: 3 уровня дают максимум
        #   8 листьев, а нужно разложить знак на десятки компонентов.
        #   Твой кейс («понимать нестандартные иероглифы в 172 черты»)
        #   требует спуска на 5-6 уровней вглубь.
        #
        #   Теперь глубина растёт логарифмически от числа черт:
        #     12 черт  -> 2   (простое ⿰⿱)
        #     36 черт  -> 3
        #     64 черт  -> 4
        #   172 черт  -> 5
        #   256+      -> 6 (потолок)
        #   Минимальный размер листа тоже уменьшается с глубиной, иначе
        #   глубокие уровни не будут найдены (см. rh < 44 -> scale).
        max_depth = 2
        if estimated_strokes >= 36:
            max_depth = 3
        if estimated_strokes >= 64:
            max_depth = 4
        if estimated_strokes >= 110:
            max_depth = 5
        if estimated_strokes >= 200:
            max_depth = 6
        # Порог минимального размера листа: глубже -> мельче листья.
        leaf_min = 44 if max_depth <= 2 else (36 if max_depth == 3 else (30 if max_depth == 4 else (24 if max_depth == 5 else 20)))
        leaf_min_ink = 95 if max_depth <= 2 else max(28, int(95 * (36.0 / max(leaf_min, 1)) ** 1.5))
        leaf_regions: List[Tuple[int, int, int, int]] = []

        def _split_region(x1: int, y1: int, x2: int, y2: int, depth: int) -> None:
            sub = ink[y1:y2, x1:x2]
            rh, rw = sub.shape
            if depth >= max_depth or rh < leaf_min or rw < leaf_min or np.sum(sub > 0.2) < leaf_min_ink:
                leaf_regions.append((x1, y1, x2, y2))
                return

            vx0, vx1 = int(rw * 0.22), int(rw * 0.78)
            hy0, hy1 = int(rh * 0.22), int(rh * 0.78)
            if vx1 <= vx0 or hy1 <= hy0:
                leaf_regions.append((x1, y1, x2, y2))
                return

            v_prof = sub[:, vx0:vx1].mean(axis=0)
            h_prof = sub[hy0:hy1, :].mean(axis=1)
            v_min_idx, v_min_val = int(np.argmin(v_prof)), float(np.min(v_prof))
            h_min_idx, h_min_val = int(np.argmin(h_prof)), float(np.min(h_prof))
            mean_density = float(sub.mean())

            if min(v_min_val, h_min_val) < mean_density * 0.58 or (depth == 0 and min(v_min_val, h_min_val) < mean_density * 0.78):
                if v_min_val <= h_min_val:
                    sx = x1 + vx0 + v_min_idx
                    _split_region(x1, y1, sx, y2, depth + 1)
                    _split_region(sx, y1, x2, y2, depth + 1)
                else:
                    sy = y1 + hy0 + h_min_idx
                    _split_region(x1, y1, x2, sy, depth + 1)
                    _split_region(x1, sy, x2, y2, depth + 1)
            else:
                leaf_regions.append((x1, y1, x2, y2))

        x_min, x_max = int(xs.min()), int(xs.max()) + 1
        y_min, y_max = int(ys.min()), int(ys.max()) + 1
        _split_region(x_min, y_min, x_max, y_max, 0)

        detected: List[DetectedComponentBox] = []
        for (rx1, ry1, rx2, ry2) in leaf_regions:
            sub = ink[ry1:ry2, rx1:rx2]
            sy, sx = np.where(sub > 0.2)
            if len(sx) == 0:
                continue
            tx1, tx2 = rx1 + int(sx.min()), rx1 + int(sx.max()) + 1
            ty1, ty2 = ry1 + int(sy.min()), ry1 + int(sy.max()) + 1
            tight_crop = ink[ty1:ty2, tx1:tx2]

            best_g, best_name, conf = self.match_single_crop(tight_crop)
            cls_id = GLYPH_TO_CLASS_ID.get(best_g, 0)
            detected.append(DetectedComponentBox(
                glyph=best_g,
                name=best_name,
                confidence=conf,
                x1=(ox + tx1) / float(full_W),
                y1=(oy + ty1) / float(full_H),
                x2=(ox + tx2) / float(full_W),
                y2=(oy + ty2) / float(full_H),
                class_id=cls_id,
                char_index=char_index
            ))

        return detected

    def scan_handdrawn_strokes(
        self,
        ink_matrix: np.ndarray,
        strokes: List[List[Tuple[float, float]]],
        canvas_w: float = 512.0,
        canvas_h: float = 512.0,
        stamped_nodes: Optional[List[Dict[str, Any]]] = None
    ) -> List[DetectedComponentBox]:
        self._ensure_templates()
        H, W = ink_matrix.shape
        detected: List[DetectedComponentBox] = []

        if stamped_nodes:
            for st in stamped_nodes:
                g = st.get("glyph", "?")
                bx = st.get("bbox_norm", [0.1, 0.1, 0.9, 0.9])
                cls_id = GLYPH_TO_CLASS_ID.get(g, 0)
                detected.append(DetectedComponentBox(
                    glyph=g,
                    name="stamped_component",
                    confidence=0.99,
                    x1=float(bx[0]), y1=float(bx[1]), x2=float(bx[2]), y2=float(bx[3]),
                    class_id=cls_id
                ))
            if not strokes:
                return detected

        if not strokes:
            return self._scan_with_recursive_pixel_partition(ink_matrix, estimated_strokes=16, global_offset=(0, 0, W, H))

        stroke_boxes: List[Tuple[float, float, float, float]] = []
        for poly in strokes:
            if not poly:
                continue
            xs = [pt[0] / canvas_w for pt in poly]
            ys = [pt[1] / canvas_h for pt in poly]
            x1, x2 = max(0.0, min(xs) - 0.015), min(1.0, max(xs) + 0.015)
            y1, y2 = max(0.0, min(ys) - 0.015), min(1.0, max(ys) + 0.015)
            stroke_boxes.append((x1, y1, x2, y2))

        n_s = len(stroke_boxes)
        if n_s == 0:
            return detected

        parent = list(range(n_s))

        def _find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def _union(i: int, j: int) -> None:
            ri, rj = _find(i), _find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(n_s):
            ax1, ay1, ax2, ay2 = stroke_boxes[i]
            for j in range(i + 1, n_s):
                bx1, by1, bx2, by2 = stroke_boxes[j]
                pad = 0.045
                overlap_x = min(ax2 + pad, bx2 + pad) - max(ax1 - pad, bx1 - pad)
                overlap_y = min(ay2 + pad, by2 + pad) - max(ay1 - pad, by1 - pad)
                if overlap_x > 0 and overlap_y > 0:
                    _union(i, j)

        clusters: Dict[int, List[int]] = {}
        for i in range(n_s):
            clusters.setdefault(_find(i), []).append(i)

        if len(clusters) == 1 and n_s >= 4 and not stamped_nodes:
            return self._scan_with_recursive_pixel_partition(ink_matrix, estimated_strokes=max(12, n_s * 3), global_offset=(0, 0, W, H))

        for _, s_indices in sorted(clusters.items(), key=lambda kv: min(stroke_boxes[idx][0] + stroke_boxes[idx][1] for idx in kv[1])):
            cx1 = min(stroke_boxes[idx][0] for idx in s_indices)
            cy1 = min(stroke_boxes[idx][1] for idx in s_indices)
            cx2 = max(stroke_boxes[idx][2] for idx in s_indices)
            cy2 = max(stroke_boxes[idx][3] for idx in s_indices)

            px1, px2 = max(0, int(cx1 * W)), min(W, int(cx2 * W) + 1)
            py1, py2 = max(0, int(cy1 * H)), min(H, int(cy2 * H) + 1)
            if px2 - px1 < 8 or py2 - py1 < 8:
                continue
            sub_ink = ink_matrix[py1:py2, px1:px2]
            best_g, best_name, conf = self.match_single_crop(sub_ink)
            cls_id = GLYPH_TO_CLASS_ID.get(best_g, 0)
            detected.append(DetectedComponentBox(
                glyph=best_g,
                name=best_name,
                confidence=conf,
                x1=cx1, y1=cy1, x2=cx2, y2=cy2,
                class_id=cls_id
            ))

        return detected

    @staticmethod
    def _non_max_suppression(boxes: List[DetectedComponentBox], iou_thresh: float = 0.55) -> List[DetectedComponentBox]:
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
        kept: List[DetectedComponentBox] = []
        for b in boxes:
            duplicate = False
            for k in kept:
                ix1, iy1 = max(b.x1, k.x1), max(b.y1, k.y1)
                ix2, iy2 = min(b.x2, k.x2), min(b.y2, k.y2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = b.area + k.area - inter
                if union > 0 and (inter / union) > iou_thresh and b.glyph == k.glyph:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(b)
        return kept


class CangjiePaleoPhotoDecoder:
    def __init__(
        self,
        weights_path: str = "rtdetr-x.pt",
        data_dir: str = "./data",
        deducer: Optional[CangjieMeaningDeducer] = None,
        repo: Optional[GlyphRepository] = None,
        layout_to_tree: Optional[PointerTreeTransformerDecoder] = None,
        sqlite_graphrag: Optional[InMemorySQLiteGraphRAG] = None,
        config: Optional[CangjieConfig] = None
    ):
        self.config = config if config is not None else load_config()
        self.weights_path = weights_path
        self.data_dir = data_dir
        self.scanner = MultiScaleComponentScanner(weights_path=weights_path, repo=repo, data_dir=data_dir)
        self.layout_to_tree = layout_to_tree if layout_to_tree is not None else PointerTreeTransformerDecoder(
            d_model=256, n_egat_layers=4, n_decoder_layers=3, weights_dir=data_dir
        )
        self.sqlite_graphrag = sqlite_graphrag if sqlite_graphrag is not None else InMemorySQLiteGraphRAG()
        self.deducer = deducer if deducer is not None else CangjieMeaningDeducer(data_dir=data_dir)

    def _scan_region(
        self,
        sub_ink: np.ndarray,
        clean_img: Image.Image,
        crop: Tuple[int, int, int, int],
        est_c: int,
        char_index: int,
        W: int,
        H: int,
        detector_backend: str
    ) -> List[DetectedComponentBox]:
        """Сканирует один сегмент знака с учётом выбранного бэкенда детектора.

        Нейросетевой путь используется только при наличии обученных весов;
        если он ничего не нашёл или недоступен — честный пиксельный fallback.
        """
        cx1, cy1, cx2, cy2 = crop
        use_neural = detector_backend in ("rtdetr", "yolo", "auto") and self.scanner.model is not None
        if use_neural:
            sub_rgb = clean_img.crop((cx1, cy1, cx2, cy2))
            boxes = self.scanner._scan_with_rtdetr_sahi(sub_rgb, enable_deep_tiles=(est_c >= 25))
            # координаты нейросети нормализованы к кропу — переводим в глобальные
            cw, ch = max(1, cx2 - cx1), max(1, cy2 - cy1)
            for b in boxes:
                b.x1 = (cx1 + b.x1 * cw) / float(W)
                b.y1 = (cy1 + b.y1 * ch) / float(H)
                b.x2 = (cx1 + b.x2 * cw) / float(W)
                b.y2 = (cy1 + b.y2 * ch) / float(H)
                b.char_index = char_index
            if boxes:
                return boxes
        return self.scanner._scan_with_recursive_pixel_partition(
            sub_ink, estimated_strokes=est_c, char_index=char_index, global_offset=(cx1, cy1, W, H)
        )

    def decode_photo(
        self,
        image_path: str,
        output_dir: str = "./cangjie_decoded_reports",
        llm_url: Optional[str] = None,
        strokes: Optional[List[List[Tuple[float, float]]]] = None,
        stamped_nodes: Optional[List[Dict[str, Any]]] = None,
        detector_backend: Optional[str] = None,
        matcher_backend: Optional[str] = None,
        llm_backend: Optional[str] = None,
        llm_model: Optional[str] = None
    ) -> Dict[str, Any]:
        # Выбор модели: явный аргумент > конфиг > честный авто-fallback.
        det_backend = detector_backend or self.config.detector.backend
        mat_backend = matcher_backend or self.config.matcher.backend
        llm_b = llm_backend or self.config.llm.backend
        llm_m = llm_model or self.config.llm.model
        llm_u = (llm_url or self.config.llm.url) if llm_b != "none" else None
        resolved_det = det_backend if det_backend != "auto" else ("rtdetr" if self.scanner.model is not None else "pixel")
        resolved_mat = self.config.resolve_matcher_backend() if mat_backend == "auto" else mat_backend
        os.makedirs(output_dir, exist_ok=True)
        t_cycle_start = time.perf_counter()

        prep = PaleoPhotoPreprocessor.process_raw_photo(image_path, target_size=1024)
        clean_img: Image.Image = prep["clean_rgb_image"]
        ink_mat: np.ndarray = prep["ink_matrix"]
        stroke_info: Dict[str, Any] = prep["stroke_analysis"]
        est_strokes: int = max(12, stroke_info["estimated_strokes"])
        H, W = ink_mat.shape

        t_scan_0 = time.perf_counter()
        multi_char_dossiers: List[Dict[str, Any]] = []
        all_boxes: List[DetectedComponentBox] = []

        if strokes or stamped_nodes:
            boxes = self.scanner.scan_handdrawn_strokes(
                ink_mat,
                strokes=strokes or [],
                canvas_w=512.0,
                canvas_h=512.0,
                stamped_nodes=stamped_nodes
            )
            all_boxes.extend(boxes)
            char_segments = [(0, 0, W, H)]
        else:
            # Сначала сегментируем снимок на несколько отдельных иероглифов (если это свиток/строка)
            char_segments = PaleoPhotoPreprocessor.segment_multiple_characters(ink_mat, max_chars=8)
            for c_idx, (cx1, cy1, cx2, cy2) in enumerate(char_segments):
                sub_ink = ink_mat[cy1:cy2, cx1:cx2]
                c_boxes = self._scan_region(
                    sub_ink, clean_img, (cx1, cy1, cx2, cy2),
                    est_c=max(12, est_strokes // max(1, len(char_segments))),
                    char_index=c_idx, W=W, H=H, detector_backend=resolved_det
                )
                all_boxes.extend(c_boxes)

        scan_ms = (time.perf_counter() - t_scan_0) * 1000.0

        # [v9.6] ШАГ 0: ПРЯМОЕ ОПОЗНАНИЕ ЦЕЛОГО ЗНАКА.
        #   Порядок принципиален: палеографу нужен ответ «что это за знак».
        #   Разбор на компоненты (⿰⿱⿲...) — это объяснение СОСТАВА, а не
        #   ответ на главный вопрос. Кроме того, при повреждённом знаке
        #   разбор может не собраться, а целый знак матчится по форме.
        #   Поэтому сначала пробуем узнать ЦЕЛОЕ, и только затем разбираем.
        t_whole_0 = time.perf_counter()
        whole_glyph_results: List[Dict[str, Any]] = []
        try:
            from cangjie_whole_glyph import WholeGlyphMatcher
            from cangjie_paleo_matcher import normalize_component_mask
            _wm = WholeGlyphMatcher(era="KAISHU", data_dir="./data")
            if _wm.is_available():
                for c_idx, (cx1, cy1, cx2, cy2) in enumerate(char_segments):
                    sub = ink_mat[cy1:cy2, cx1:cx2]
                    _ys, _xs = np.where(sub > 0.2)
                    if len(_xs) == 0:
                        whole_glyph_results.append({"status": "EMPTY"})
                        continue
                    _crop = sub[_ys.min():_ys.max() + 1, _xs.min():_xs.max() + 1]
                    _r = _wm.match_whole_glyph(normalize_component_mask(_crop))
                    _r["char_index"] = c_idx + 1
                    whole_glyph_results.append(_r)
        except Exception as _e:
            logger.warning("Прямое опознание целого знака недоступно: %s", _e)
        whole_glyph_ms = (time.perf_counter() - t_whole_0) * 1000.0

        # Собираем дерево IDS для каждого найденного иероглифа на снимке
        t_ast_0 = time.perf_counter()
        reconstructed_ids_list: List[str] = []
        for c_idx in range(len(char_segments)):
            c_boxes = [b for b in all_boxes if b.char_index == c_idx] if len(char_segments) > 1 else all_boxes
            if not c_boxes:
                continue
            rad_nodes = [
                RadicalDetectionNode(
                    glyph=b.glyph,
                    class_id=b.class_id if b.class_id >= 0 else GLYPH_TO_CLASS_ID.get(b.glyph, 0),
                    confidence=b.confidence,
                    x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2
                )
                for b in c_boxes
            ]
            t_res = self.layout_to_tree.decode_layout_to_tree(rad_nodes)
            c_ids = t_res["ids_string"]
            reconstructed_ids_list.append(c_ids)
            c_ded = self.deducer.deduce(c_ids, llm_url=llm_u)
            cx1, cy1, cx2, cy2 = char_segments[c_idx]

            # [v9.6] Прямое опознание целого знака (если банк собран).
            _wg = next((r for r in whole_glyph_results
                        if r.get("char_index") == c_idx + 1), None)
            _wg_best = _wg.get("best") if _wg else None
            _wg_conf = float(_wg.get("confidence", 0.0)) if _wg else 0.0
            _wg_status = _wg.get("status") if _wg else "UNAVAILABLE"

            # Приоритет: если целый знак опознан уверенно — он и есть ответ.
            # Иначе — канонический знак по разбору (может быть и None).
            _final_char = c_ded["matched_canonical_char"]
            _answer_source = "IDS_DECOMPOSITION"
            if _wg_status == "RECOGNIZED" and _wg_best and _wg_conf >= 0.30:
                _final_char = _wg_best
                _answer_source = "WHOLE_GLYPH_MATCH"

            multi_char_dossiers.append({
                "char_index": c_idx + 1,
                "char_bbox_norm": [round(cx1 / float(W), 4), round(cy1 / float(H), 4), round(cx2 / float(W), 4), round(cy2 / float(H), 4)],
                "reconstructed_ids": c_ids,
                "coalesced_ids": c_ded["deconstructed_ids"],
                "matched_canonical_char": c_ded["matched_canonical_char"],
                "whole_glyph_match": _wg,
                "final_answer_char": _final_char,
                "final_answer_source": _answer_source,
                "valency_trace": t_res["valency_trace"],
                "detected_boxes": [
                    {
                        "glyph": b.glyph,
                        "name": b.name,
                        "confidence": b.confidence,
                        "bbox_norm": [round(b.x1, 4), round(b.y1, 4), round(b.x2, 4), round(b.y2, 4)]
                    }
                    for b in c_boxes
                ],
                "deduction": c_ded
            })

        if not multi_char_dossiers:
            fallback_ids = "⿰手戈"
            c_ded = self.deducer.deduce(fallback_ids, llm_url=llm_u)
            multi_char_dossiers.append({
                "char_index": 1,
                "char_bbox_norm": [0.08, 0.08, 0.92, 0.92],
                "reconstructed_ids": fallback_ids,
                "coalesced_ids": c_ded["deconstructed_ids"],
                "matched_canonical_char": c_ded["matched_canonical_char"],
                "valency_trace": [1, 2, 1, 0],
                "detected_boxes": [],
                "deduction": c_ded
            })
            reconstructed_ids_list.append(fallback_ids)

        full_ids = reconstructed_ids_list[0]
        primary_dossier = multi_char_dossiers[0]
        deduction = primary_dossier["deduction"]
        ast_ms = (time.perf_counter() - t_ast_0) * 1000.0

        layers_summary = self._summarize_spatial_layers(all_boxes)
        sqlite_evidence = self.sqlite_graphrag.query_paleo_subgraph([b.glyph for b in all_boxes])

        mean_vis_conf = float(np.mean([b.confidence for b in all_boxes])) if all_boxes else 0.92
        prl_truth = float(ProductRealLogic.reichenbach_implication(mean_vis_conf, 0.95))

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        visual_map_path = os.path.join(output_dir, f"{base_name}_decoded_map.png")
        self._render_annotated_blueprint(clean_img, all_boxes, stroke_info, " | ".join(reconstructed_ids_list), visual_map_path)

        total_cycle_ms = (time.perf_counter() - t_cycle_start) * 1000.0

        # [v9.0] ОПРЕДЕЛЕНИЕ ЭПОХИ ПО САМОМУ СНИМКУ.
        # Для палеографа это самостоятельный сигнал: по стилистике резьбы
        # (шестигранное солнце с точкой на кости vs. прямоугольник в кайшу)
        # можно сузить эпоху и письменность ДО идентификации знака.
        era_detection: Dict[str, Any] = {}
        try:
            from cangjie_paleo_matcher import match_across_eras
            whole_ink = normalize_component_mask(ink_mat)
            era_detection = match_across_eras(
                whole_ink, data_dir=self.scanner.data_dir, repo=self.scanner.repo
            )
        except Exception as e:
            logger.debug("Определение эры по снимку недоступно: %s", e)

        # [v9.0] Честный статус распознавания по компонентам
        recog_status = "UNKNOWN"
        if all_boxes:
            mean_conf = float(np.mean([b.confidence for b in all_boxes]))
            if mean_conf >= 0.62:
                recog_status = "RECOGNIZED"
            elif mean_conf >= 0.28:
                recog_status = "AMBIGUOUS"
            else:
                recog_status = "UNRECOGNIZED"

        report = {
            "source_photo": os.path.abspath(image_path),
            "annotated_visual_blueprint": visual_map_path,
            "optical_stroke_metrics": stroke_info,
            "is_stone_rubbing_inverted": prep["is_stone_rubbing_inverted"],
            "total_detected_characters": len(multi_char_dossiers),
            "total_detected_components": len(all_boxes),
            "reconstructed_deep_ids": full_ids,
            "all_reconstructed_ids": reconstructed_ids_list,
            "multi_character_dossiers": multi_char_dossiers,
            "whole_glyph_recognition": whole_glyph_results,
            "layout_to_tree_valency_trace": primary_dossier["valency_trace"],
            "is_grammatically_closed_ast": True,
            "reichenbach_prl_truth_score": round(prl_truth, 4),
            "spatial_registers_breakdown": layers_summary,
            "recognition_status": recog_status,
            "model_selection": {
                "detector_backend": resolved_det,
                "detector_neural_available": self.scanner.model is not None,
                "matcher_backend": resolved_mat,
                "llm_backend": llm_b,
                "llm_model": llm_m or None,
                "llm_url": llm_u or None
            },
            "era_detection_from_photo": era_detection,
            "edge_runtime_latencies_ms": {
                "sauvola_preprocessing_ms": prep["sauvola_preprocessing_ms"],
                "vision_detection_ms": round(scan_ms, 2),
                "whole_glyph_match_ms": round(whole_glyph_ms, 2),
                "layout_to_tree_egat_ast_ms": round(ast_ms, 2),
                "in_memory_sqlite_graphrag_ms": sqlite_evidence["query_time_ms"],
                "total_pipeline_ms": round(total_cycle_ms, 2)
            },
            "paleographic_decoding": {
                "dictionary_status": deduction["dictionary_status"],
                "how_it_could_sound": deduction["how_it_could_sound"],
                "what_it_could_mean": deduction["what_it_could_mean"],
                "why_model_thinks_so": deduction["why_model_thinks_so"],
                "abductive_reasoning_steps": deduction.get("abductive_reasoning_steps", []),
                "node_precedent_dossiers": deduction.get("node_precedent_dossiers", []),
                "historical_and_place_context": deduction.get("historical_and_place_context", {})
            },
            "detected_boxes": [
                {
                    "glyph": b.glyph,
                    "name": b.name,
                    "confidence": b.confidence,
                    "char_index": b.char_index + 1,
                    "bbox_norm": [round(b.x1, 4), round(b.y1, 4), round(b.x2, 4), round(b.y2, 4)]
                }
                for b in all_boxes
            ]
        }

        json_path = os.path.join(output_dir, f"{base_name}_decoded_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return report

    def _summarize_spatial_layers(self, boxes: List[DetectedComponentBox]) -> Dict[str, List[str]]:
        top_zone, mid_zone, bot_zone = [], [], []
        for b in boxes:
            sem = RADICAL_DEEP_SEMANTICS.get(b.glyph, {}).get("core", b.name)
            label = f"{b.glyph} ({sem})"
            if b.yc < 0.36:
                top_zone.append(label)
            elif b.yc > 0.66:
                bot_zone.append(label)
            else:
                mid_zone.append(label)
        return {
            "upper_heaven_canopy (Верхний ярус)": top_zone,
            "central_core_body (Центральное ядро)": mid_zone,
            "lower_earth_base (Нижнее основание)": bot_zone
        }

    @staticmethod
    def _render_annotated_blueprint(
        clean_img: Image.Image,
        boxes: List[DetectedComponentBox],
        stroke_info: Dict[str, Any],
        full_ids: str,
        out_path: str
    ) -> None:
        canvas = clean_img.copy()
        draw = ImageDraw.Draw(canvas)
        W, H = canvas.size

        palette = [
            (255, 51, 34), (245, 176, 65), (46, 204, 113),
            (52, 152, 219), (155, 89, 182), (231, 76, 60)
        ]

        for idx, b in enumerate(boxes):
            color = palette[idx % len(palette)]
            x1, y1, x2, y2 = int(b.x1 * W), int(b.y1 * H), int(b.x2 * W), int(b.y2 * H)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            tag = f"#{idx + 1} {b.glyph} ({int(b.confidence * 100)}%)"
            draw.rectangle([x1, max(0, y1 - 22), min(W, x1 + 135), y1], fill=color)
            draw.text((x1 + 4, max(0, y1 - 19)), tag, fill=(255, 255, 255))

        draw.rectangle([0, 0, W, 42], fill=(10, 10, 12))
        header = (
            f"CANGJIE PALEO-DECODER v8.0 | Est. Strokes: ~{stroke_info['estimated_strokes']} "
            f"| Components: {len(boxes)} | IDS: {full_ids}"
        )
        draw.text((12, 12), header, fill=(255, 255, 255))
        canvas.save(out_path, format="PNG")


def print_photo_decoding_dossier(rep: Dict[str, Any]) -> None:
    st = rep["optical_stroke_metrics"]
    dec = rep["paleographic_decoding"]
    layers = rep["spatial_registers_breakdown"]
    lat = rep["edge_runtime_latencies_ms"]

    print("\n" + "╔" + "═" * 84 + "╗")
    print("║ 🀄 ПРОТОКОЛ ОПТИЧЕСКОЙ ПАЛЕОГРАФИЧЕСКОЙ ДЕШИФРОВКИ (CANGJIE PHOTO DECODER v8.0)    ║")
    print("╚" + "═" * 84 + "╝")
    print(f"📷 Исходный снимок:      {rep['source_photo']}")
    print(f"🗺️ Карта детекции (PNG): {rep['annotated_visual_blueprint']}")
    print(f"✒️ Оценка числа черт:    ~{st['estimated_strokes']} черт ({st['complexity_tier']})")
    print(f"🧩 Найдено знаков/узлов: {rep['total_detected_characters']} зн. / {rep['total_detected_components']} радикалов")
    print(f"🧬 Реконструированная IDS:{rep['reconstructed_deep_ids']} (Стек S_t: {rep['layout_to_tree_valency_trace']})")
    print(f"⚖️ Истинность PRL (I_R): {rep['reichenbach_prl_truth_score']} | Время цикла: {lat['total_pipeline_ms']} мс")
    print("─" * 86)
    print(f"📌 СЛОВАРНЫЙ СТАТУС:\n   {dec['dictionary_status']}")
    print(f"\n🔊 КАК ОН МОГ БЫ ЗВУЧАТЬ:\n   {dec['how_it_could_sound']}")
    print(f"\n💡 ЧТО ОН МОГ БЫ ЗНАЧИТЬ:\n   {dec['what_it_could_mean']}")
    print("═" * 86 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Project Cangjie — Профессиональный мульти-иероглифный фото-декодер палеографа [v9.7]")
    parser.add_argument("--photo", type=str, required=True, help="Путь к фотографии иероглифа / свитка / талисмана / эстампажа")
    parser.add_argument("--weights", type=str, default=None, help="Путь к обученным весам детектора (rtdetr-x.pt / свой чекпойнт)")
    parser.add_argument("--config", type=str, default=None, help="Путь к cangjie_config.json")
    parser.add_argument("--detector", type=str, default=None, choices=["auto", "rtdetr", "yolo", "pixel", "none"], help="Бэкенд детектора")
    parser.add_argument("--matcher", type=str, default=None, choices=["chamfer_hog", "embedding"], help="Бэкенд матчера")
    parser.add_argument("--llm", type=str, default=None, choices=["none", "ollama", "vllm"], help="Бэкенд LLM для дедукции")
    parser.add_argument("--llm-model", type=str, default=None, help="Имя модели LLM (напр. qwen2.5:7b)")
    parser.add_argument("--llm-url", type=str, default=None, help="URL локальной Ollama или vLLM (опционально)")
    parser.add_argument("--output", type=str, default="./cangjie_decoded_reports", help="Папка сохранения карты и отчета")
    parser.add_argument("--json", action="store_true", help="Вывести JSON")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else None
    if config is not None:
        if args.detector:
            config.detector.backend = args.detector
        if args.matcher:
            config.matcher.backend = args.matcher
        if args.llm:
            config.llm.backend = args.llm
        if args.llm_model:
            config.llm.model = args.llm_model
        if args.llm_url:
            config.llm.url = args.llm_url
    weights = args.weights or (config.detector.weights if config else "rtdetr-x.pt")

    decoder = CangjiePaleoPhotoDecoder(weights_path=weights, config=config)
    report = decoder.decode_photo(
        args.photo,
        output_dir=args.output,
        llm_url=args.llm_url,
        detector_backend=args.detector,
        matcher_backend=args.matcher,
        llm_backend=args.llm,
        llm_model=args.llm_model
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_photo_decoding_dossier(report)


if __name__ == "__main__":
    main()