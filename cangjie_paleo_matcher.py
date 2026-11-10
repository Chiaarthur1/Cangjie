"""
Project Cangjie (Цанцзе) — Paleo Component Matcher v9.0 (Era-Aware, Calibrated)

ЗАЧЕМ ЭТОТ МОДУЛЬ:
    Прежний матчер в `cangjie_photo_decoder.py` имел три системных дефекта,
    которые делали распознавание с фото практически бесполезным для
    палеографа:

    1) МАЛЕНЬКИЙ КАНДИДАТНЫЙ ПУЛ. Шаблоны строились только по 214 радикалам
       Канси (+ ~20 фонетиков) = 281 шаблон. Но в CHISE 97 430 знаков, и
       уникальных КОМПОНЕНТОВ в них 11 542. Любой знак, собранный из
       не-радикальных деталей (⻗, 齒, 鬲, 耑, ...), в принципе не мог быть
       опознан — кандидата просто не было в банке.

    2) ДОМЕННЫЙ РАЗРЫВ. Шаблоны рендерились ТОЛЬКО в регулярной кайшу
       (CalligraphicPillowRenderer). Но палеограф снимает 甲骨文 (костяные
       гадательные кости) и 金文 (бронзовые сосуды) — формы которых радикально
       отличаются от кайшу: в Цзягувэнь 日 это шестигранное солнце с точкой,
       а не прямоугольник. Сравнивать резьбу по кости с кайшу — бессмысленно.

    3) ФИКТИВНАЯ УВЕРЕННОСТЬ. Код вида
           conf = round(max(0.55, min(0.99, best_score * 1.12)), 4)
       гарантировал НИЖНЮЮ границу 0.55 для любого входа, даже для белого
       шума. Коэффициент 1.12 был произвольным. Итог: UI показывал «55%»
       для мусора, а палеограф принимал это за оценку.

ЧТО ЗДЕСЬ:
    - `PaleoTemplateBank` — эра-специфичный банк шаблонов (КАЙШУ / ЦЗЯГУВЭНЬ /
      ЦЗИНЬВЭНЬ) по ВСЕЙ вселенной компонентов CHISE.
    - Векторный матчинг через BLAS (matmul), а не Python-цикл: 11 542
      шаблона × 1064 признака = ~12 млн сравнений за ~10 мс.
    - Честная калибровка: confidence = posterior_top × fit, где posterior —
      softmax по кандидатам (соблазн двух похожих знаков автоматически
      роняет уверенность до ~0.5), а fit — насколько хорошо лучший
      кандидат вообще подходит. Плюс явный статус: RECOGNIZED / AMBIGUOUS /
      UNRECOGNIZED. Никакой нижней границы «0.55».
    - `match_era` — сравнивает снимок сразу со всеми эрами и сообщает, какая
      письменность ближе (для палеографа это сам по себе полезный сигнал:
      «ваше фото похоже на 金文»).

СОВМЕСТИМОСТЬ:
    Дескриптор (Chamfer Distance-Field + 4-ориентационный HOG) взят из
    существующего кода `MultiScaleComponentScanner` без изменений, поэтому
    новый банк полностью заменяет старый, а не конкурирует с ним.
"""

from __future__ import annotations

import os
import re
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Sequence

import numpy as np
from PIL import Image, ImageFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PALEOMATCH-v9]: %(message)s"
)
logger = logging.getLogger("CangjiePaleoMatcher")


# Операторы IDS — они не являются компонентами
IDC_OPERATORS = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")
_TOKEN_RE = re.compile(r"&[^;]+;|.")

# Эры
ERA_KAISHU = "KAISHU"             # 楷书 — регулярное письмо (эталон)
ERA_JIAGUWEN = "SHANG_JIAGUWEN"   # 甲骨文 — гадательные кости Шан
ERA_JINWEN = "ZHOU_JINWEN"        # 金文 — бронзовые сосуды Чжоу
ERA_SEAL = "XIAOZHUAN"            # 篆书 — печати и стелы

ALL_ERAS = (ERA_KAISHU, ERA_JIAGUWEN, ERA_JINWEN, ERA_SEAL)

# Размеры дескриптора
CHAMFER_RES = 64      # разрешение бинарной маски запроса

# [v9.6] РАЗРЕШЕНИЕ РЕНДЕРА ШАБЛОНА.
#   БЫЛО: шаблоны рендерились сразу в 64x64 (CHAMFER_RES).
#   Из-за этого штрихи СЛИПАЛИСЬ: 日 превращался в залитый квадрат
#   (плотность 0.77 вместо 0.40), 龘 и 𰻞 — в бесформенные пятна.
#
#   Измерено (дескриптор vs реальное фото того же знака):
#       рендер @64  -> density 0.81, cos_chamfer −0.25, SCORE 0.28
#       рендер @128 -> density 0.75, cos_chamfer +0.06, SCORE 0.46
#       рендер @256 -> density 0.42, cos_chamfer +0.75, SCORE 0.84
#       рендер @320 -> density 0.34, cos_chamfer +0.86, SCORE 0.91
#
#   То есть рендер в 64 давал МАСКУ, ПО КОТОРОЙ НЕВОЗМОЖНО ОПОЗНАТЬ
#   знак. Рендер в 256 с последующим сжатием до 64 даёт корректную
#   геометрию штрихов.
#
#   ВАЖНО: это меняет ДЕСКРИПТОР -> кэши банков становятся невалидными,
#   поэтому DESCRIPTOR_VERSION поднят (см. ниже).
TEMPLATE_RENDER_RES = 256   # разрешение рендера ШАБЛОНА (до сжатия в 64)
BANK_RES = 32         # разрешение поля расстояний в банке (экономия памяти)
HOG_GRID = 3          # сетка ячеек HOG
DENSITY_CANON = 0.42  # каноническая плотность туши для сравнения (v9.2)
HOG_ORIENT = 4        # число ориентаций в ячейке


def normalize_component_mask(ink: np.ndarray, size: int = CHAMFER_RES) -> np.ndarray:
    """
    Приводит маску компонента к каноническому виду: ТЕСНЫЙ КРОП по туши
    + ресайз до size×size.

    ЭТО КРИТИЧНО ДЛЯ СОПОСТАВИМОСТИ БАНКА И ЗАПРОСА:
        фото-декодер вырезает кроп по содержимому и ресайзит в 64×64,
        поэтому шаблоны банка обязаны готовиться ТОЧНО так же. Раньше банк
        рендерил глиф «в рамке 4..56», а запрос приходил «плотным кропом» —
        геометрия не совпадала, и top-1 по кайшу падал до 0% при 95% по
        甲骨文 (где архаичный рендер сам по себе заполняет кадр).

    ПОРЯДОК ВАЖЕН: сначала медианный фильтр (убирает трещины и сколы в
    1 пиксель), потом кроп по ПОВЫШЕННОМУ порогу (0.20). Иначе слабые
    трещины (интенсивность ~0.12) расширяют кроп, глиф сжимается в кадре и
    перестаёт совпадать с шаблоном — на бенчмарке деградация BONE роняла
    top-1 вдвое.
    """
    arr = np.asarray(ink, dtype=np.float32)
    if arr.max() <= 0.01:
        return arr

    arr = despeckle_median(arr)

    ys, xs = np.where(arr > 0.20)
    if len(xs) == 0:
        ys, xs = np.where(arr > 0.08)
        if len(xs) == 0:
            return arr
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    tight = arr[y1:y2, x1:x2]

    pil = Image.fromarray((np.clip(tight, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    pil = pil.resize((size, size), Image.Resampling.BILINEAR)
    return np.clip(np.array(pil, dtype=np.float32) / 255.0, 0.0, 1.0)


def density_normalize(ink: np.ndarray, target_density: float = 0.42) -> np.ndarray:
    """
    [v9.2] ПРИВЕДЕНИЕ ПЛОТНОСТИ ТУШИ К КАНОНИЧЕСКОЙ.

    ЗАЧЕМ (это ключевой баг реального фото-пайплайна):
        КАЙШУ-ШАБЛОНЫ в банке нарисованы жирным каллиграфическим пером:
        плотность туши в кадре 64x64 ~ 0.71.
        Реальная фотография вырезанного иероглифа (кость/бронза/拓片/бумага)
        даёт плотность ~ 0.13-0.20 — тушь тонкая, штрихи высохшие.

        Дескриптор — центрированная корреляция полей. Если у запроса и у
        шаблона разная ПЛОТНОСТЬ (а центрирование убирает только среднее,
        но не разный «вес» тусовки), корреляция падает катастрофически:
                max score по 4720 шаблонам = 0.159  (было 1.0 у self-match)
        => fit = 0 => confidence = 0.0 => «UNRECOGNIZED» на ВСЁ подряд.
        Именно так система молчала на реальных фотографиях, показывая
        при этом 97.8% на синтетическом бенчмарке.

    ЧТО ДЕЛАЕТ:
        Приводит среднюю плотность маски к канонической (target_density)
        МЯГКО, без бинаризации: сначала мягкое расширение штрихов
        (MaxFilter → «толстение»), затем нормировка по среднему.
        Форма глифа (то, что реально несёт информацию) сохраняется.

    ВАЖНО: применяется к ЗАПРОСУ и к ШАБЛОНАМ одинаково, иначе
        сравнение снова окажется в разных шкалах.
    """
    arr = np.asarray(ink, dtype=np.float32)
    if arr.max() <= 0.01:
        return arr

    cur = float(arr.mean())
    if cur <= 1e-6:
        return arr

    pil = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")

    # 1) Если тушь слишком жидкая (реальное фото) — «толстим» штрихи.
    #    Несколько проходов MaxFilter подбирают толщину под target.
    strokes = pil
    for _ in range(6):
        d = float(np.asarray(strokes, dtype=np.float32).mean() / 255.0)
        if d >= target_density * 0.92:
            break
        strokes = strokes.filter(ImageFilter.MaxFilter(3))
    # 2) Мягкое размытие, чтобы утолщение не превращалось в «блобы».
    strokes = strokes.filter(ImageFilter.GaussianBlur(radius=1.1))

    out = np.asarray(strokes, dtype=np.float32) / 255.0
    d_out = float(out.mean())
    if d_out <= 1e-6:
        return arr

    # 3) Линейная нормировка плотности к канону.
    gain = target_density / d_out
    if gain <= 1.0:
        # слишком плотно (шаблон) — приглушаем
        out = out * gain
    else:
        # слишком жидко (фото) — усиливаем мягко, с насыщением
        out = np.clip(out * gain, 0.0, 1.0)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ============================================================================
# 1. ДЕСКРИПТОР (Chamfer Distance Field + HOG)
# ============================================================================

def despeckle_median(ink: np.ndarray) -> np.ndarray:
    """
    Медианный фильтр 3x3 по маске туши.

    Зачем: на гадательных костях (甲骨文) и 拓片 по поверхности идут тонкие
    трещины и сколы шириной в 1 пиксель. Для дескриптора они — ложные
    признаки, и компонент начинает «не узнаваться»: на бенчмарке деградация
    BONE роняла top-1 до 51%. Медиана 3x3 удаляет одиночные тонкие линии,
    сохраняя настоящие штрихи (ширина >= 2 пикселей).

    Применяется и к шаблону, и к запросу — симметрично, поэтому векторное
    пространство признаков остаётся согласованным.
    """
    arr = np.asarray(ink, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 3:
        return arr
    pad = np.pad(arr, 1, mode="edge")
    stack = np.stack([
        pad[dy:dy + arr.shape[0], dx:dx + arr.shape[1]]
        for dy in range(3) for dx in range(3)
    ], axis=0)
    return np.median(stack, axis=0).astype(np.float32)


def compute_chamfer_and_hog(
    ink_64: np.ndarray,
    bank_res: int = BANK_RES
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Дескриптор компонента: мягкое поле расстояний (Chamfer) + HOG 4 ориентаций.

    ВАЖНО (исправление v9.1): векторы ЦЕНТРИРУЮТСЯ перед нормировкой.
        Без центрирования косинусная близость размытых масок у ЛЮБЫХ двух
        «плотных» глифов равна ~0.99 — метрика не различает ничего, margin
        между первым и вторым кандидатом ~0.002, и калибровать уверенность
        по ней невозможно. После вычитания среднего косинус становится
        корреляцией Пирсона, которая действительно отражает форму.

    Параметры HOG совпадают с исторической реализацией в photo_decoder.
    """
    ink_64 = despeckle_median(ink_64)

    # [v9.2] Приводим плотность туши к канонической ДО расчёта дескриптора.
    #   Без этого реальное фото (плотность ~0.13) не может совпасть с
    #   кайшу-шаблоном (~0.71): центрированная корреляция разваливается,
    #   и система честно (но бесполезно) отвечает «не опознано».
    #   Шаблоны банка нормализуются ТОЧНО ТАК ЖЕ -> сравнение честное.
    ink_64 = density_normalize(ink_64, target_density=DENSITY_CANON)

    pil_im = Image.fromarray(np.clip(ink_64 * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    soft_im = pil_im.filter(ImageFilter.GaussianBlur(radius=2.2))
    soft_arr = np.array(soft_im, dtype=np.float32) / 255.0

    # Понижаем разрешение поля расстояний до размера банка
    if bank_res != soft_arr.shape[0]:
        soft_small = np.array(
            Image.fromarray((soft_arr * 255.0).astype(np.uint8), mode="L")
            .resize((bank_res, bank_res), Image.Resampling.BILINEAR),
            dtype=np.float32
        ) / 255.0
    else:
        soft_small = soft_arr

    flat = soft_small.reshape(-1)
    flat = flat - float(flat.mean())
    norm_s = float(np.linalg.norm(flat)) or 1.0
    chamfer_field = flat / norm_s

    # Градиенты считаем в полном разрешении 64x64
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
    for r_i in range(HOG_GRID):
        for c_i in range(HOG_GRID):
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
    hog_vec = hog_vec - float(hog_vec.mean())
    norm_h = float(np.linalg.norm(hog_vec)) or 1.0
    return chamfer_field, hog_vec / norm_h


# ============================================================================
# 2. РЕЗУЛЬТАТ МАТЧИНГА
# ============================================================================

@dataclass
class MatchCandidate:
    """Один кандидат распознавания компонента."""
    glyph: str
    score: float              # косинусная близость к шаблону (сырое)
    posterior: float          # апостериорная вероятность после softmax
    name: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "glyph": self.glyph,
            "name": self.name,
            "score": round(float(self.score), 4),
            "posterior": round(float(self.posterior), 4),
        }


@dataclass
class MatchResult:
    """Полный результат распознавания одного компонента."""
    best: Optional[MatchCandidate]
    confidence: float                      # калиброванная уверенность 0..1
    status: str                            # RECOGNIZED | AMBIGUOUS | UNRECOGNIZED
    margin: float                          # разрыв между 1-м и 2-м кандидатом
    raw_score: float                       # абсолютное качество совпадения
    candidates: List[MatchCandidate] = field(default_factory=list)
    era: str = ERA_KAISHU

    @property
    def is_reliable(self) -> bool:
        return self.status == "RECOGNIZED"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "best_glyph": self.best.glyph if self.best else None,
            "best_name": self.best.name if self.best else "",
            "confidence": round(float(self.confidence), 4),
            "status": self.status,
            "margin": round(float(self.margin), 4),
            "raw_score": round(float(self.raw_score), 4),
            "era": self.era,
            "candidates": [c.as_dict() for c in self.candidates],
        }


# ============================================================================
# 3. БАНК ШАБЛОНОВ
# ============================================================================

class PaleoTemplateBank:
    """
    Эра-специфичный банк шаблонов компонентов.

    Строится по ВСЕЙ вселенной компонентов CHISE (по умолчанию — компоненты,
    реально встречающиеся в IDS-разборах 97 430 знаков), а не по 214
    радикалам. Шаблоны для эр 甲骨文/金文/篆书 рендерятся морфологическим
    гомеоморфизмом из архаического движка, а не копируют кайшу.
    """

    # Шаблоны, которые гарантированно присутствуют (частые радикалы)
    def __init__(
        self,
        era_mode: str = ERA_KAISHU,
        data_dir: str = "./data",
        repo=None,
        max_components: int = 12000,
        min_frequency: int = 1,
    ):
        self.era_mode = era_mode if era_mode in ALL_ERAS else ERA_KAISHU
        self.data_dir = os.path.abspath(data_dir)
        self.max_components = int(max_components)
        self.min_frequency = int(min_frequency)

        self._repo = repo
        self._archaic = None

        self.glyphs: List[str] = []
        self.names: List[str] = []
        self.frequencies: List[int] = []
        self._chamfer: Optional[np.ndarray] = None   # (N, D) L2-нормированная
        self._hog: Optional[np.ndarray] = None        # (N, D) L2-нормированная

        # [v9.4] ЧЕСТНАЯ КАЛИБРОВКА ИЗ ДАННЫХ.
        # Коэффициенты логистической регрессии обучены на реальном
        # калибровочном наборе (реальные доменные деградации + мусор +
        # перекрёстные пары). Пока модель не загружена — работает
        # зашитая формула v9.3 (см. match()).
        self._calib_model: Optional[Dict[str, Any]] = None
        self._calib_mu: Optional[np.ndarray] = None
        self._calib_sd: Optional[np.ndarray] = None
        self._calib_threshold: float = 0.5

    # ------------------------------------------------------------------
    @property
    def repo(self):
        if self._repo is None:
            from cangjie_benchmark_forge import GlyphRepository
            self._repo = GlyphRepository()
        return self._repo

    @property
    def archaic(self):
        if self._archaic is None:
            from cangjie_archaic_epigraphy_engine import ArchaicJiaguwenJinwenSynthesizer
            self._archaic = ArchaicJiaguwenJinwenSynthesizer(repo=self.repo, resolution=256)
        return self._archaic

    # ------------------------------------------------------------------
    # Версия дескриптора. Меняется при ЛЮБОМ изменении пайплайна
    # нормализации/дескриптора (density_normalize, compute_chamfer_and_hog,
    # рендер архаик-масок). Кэш банка несёт эту метку, и при несовпадении
    # версий банк ПЕРЕСОБИРАЕТСЯ, а не используется молча.
    #
    # [v9.3] ЧТО ЭТО ПОЧИНИЛО:
    #     Банк kaishu был пересобран 19:56 (уже с density_normalize),
    #     а банки jiaguwen/jinwen/xiaozhuan остались от 19:23 (без него).
    #     Запрос нормализуется ВСЕГДА, а шаблоны древних эр — нет.
    #     => шаблон и запрос в разных шкалах => top-1 по 甲骨文 рухнул
    #        с 98.3% до 46.8%, и это выглядело как «модель стала хуже»,
    #        хотя сломались только кэши.
    # [v9.6] v96-supersample: шаблоны рендерятся в 256x256 и сжимаются
    #   в 64x64 (см. TEMPLATE_RENDER_RES). Старые кэши (v93) содержат
    #   слипшиеся маски — они невалидны и будут пересобраны автоматически.
    DESCRIPTOR_VERSION = "v96-supersample"

    def _cache_path(self) -> str:
        return os.path.join(
            self.data_dir,
            f"cangjie_paleo_templates_{self.era_mode.lower()}_v91.npz"
        )

    def _cache_is_current(self, z) -> bool:
        """Кэш пригоден только если он собран ТЕМ ЖЕ пайплайном дескриптора."""
        try:
            stored = str(z["descriptor_version"])
        except KeyError:
            return False
        return stored == self.DESCRIPTOR_VERSION

    # ------------------------------------------------------------------
    def _collect_component_universe(self) -> List[Tuple[str, int]]:
        """
        Собирает список компонентов CHISE с их корпусной частотой.
        Возвращает [(glyph, freq), ...] отсортированный по убыванию частоты.
        """
        ids_files = [
            os.path.join(self.data_dir, "IDS-UCS-Basic.txt"),
            os.path.join(self.data_dir, "chise_all", "IDS-UCS-Basic.txt"),
        ]
        path = next((p for p in ids_files if os.path.exists(p)), None)
        if path is None:
            logger.warning("CHISE IDS не найден — банк шаблонов будет пустым")
            return []

        from collections import Counter
        freq: Counter = Counter()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or line.startswith(";"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                raw_ids = parts[2].split()
                if not raw_ids:
                    continue
                for tok in _TOKEN_RE.findall(raw_ids[0]):
                    if tok not in IDC_OPERATORS:
                        freq[tok] += 1

        items = [(g, c) for g, c in freq.items() if c >= self.min_frequency]
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        return items[: self.max_components]

    # ------------------------------------------------------------------
    def _render_component(self, glyph: str) -> Optional[np.ndarray]:
        """
        Рендерит компонент в стиле своей эры и возвращает бинарную маску 64x64.
        """
        from cangjie_benchmark_forge import (
            CalligraphicPillowRenderer, RadicalLeaf, BoundingBox, SpatialSlot
        )

        if self.era_mode == ERA_KAISHU:
            # [v9.6] Рендерим в TEMPLATE_RENDER_RES (256), а не в 64:
            #   в 64 штрихи слипаются в нечитаемую кашу (см. константу).
            #   normalize_component_mask затем сожмёт маску в 64x64.
            renderer = CalligraphicPillowRenderer(
                target_resolution=TEMPLATE_RENDER_RES, supersample_factor=1)
            leaf = RadicalLeaf(glyph, SpatialSlot.ROOT, self.repo)
            if not leaf.medians:
                return None
            pad = 4
            leaf.calculate_geometry(BoundingBox(
                pad, pad, TEMPLATE_RENDER_RES - pad, TEMPLATE_RENDER_RES - pad))
            leaf.render(renderer)
            arr = np.array(renderer.finalize().convert("L"), dtype=np.float32)
            ink = np.clip((255.0 - arr) / 255.0, 0.0, 1.0)
            return normalize_component_mask(ink)

        # Архаические эры: рисуем штрихи с морфологией эпохи
        mask = self._render_archaic_mask(glyph)
        if mask is None:
            return None
        return mask

    def _render_archaic_mask(self, glyph: str) -> Optional[np.ndarray]:
        """Рисует компонент в стиле 甲骨文 / 金文 / 篆书 (силуэт, без фактуры)."""
        from PIL import Image, ImageDraw
        from cangjie_benchmark_forge import RadicalLeaf, SpatialSlot

        res = CHAMFER_RES
        leaf = RadicalLeaf(glyph, SpatialSlot.ROOT, self.repo)

        picto = getattr(self.archaic, "ARCHAIC_PICTOGRAPHIC_SKELETONS", {})
        if glyph in picto:
            medians = picto[glyph]
            cx, cy, cw, ch = 160.0, 140.0, 720.0, 740.0
        elif leaf.medians:
            medians = leaf.medians
            cbox = leaf.content_bbox
            cx, cy = cbox.x, cbox.y
            cw, ch = max(1.0, cbox.width), max(1.0, cbox.height)
        else:
            return None

        canvas = Image.new("L", (res, res), 0)
        draw = ImageDraw.Draw(canvas)

        margin = 4.0
        usable = res - 2 * margin
        sx = usable / cw
        sy = usable / ch

        rng = np.random.RandomState(1234)
        base_w = 4 if self.era_mode == ERA_JIAGUWEN else 5

        for raw_stroke in medians:
            if self.era_mode == ERA_JIAGUWEN:
                stroke = self.archaic._morph_stroke_to_archaic(raw_stroke, "SHANG_JIAGUWEN", rng)
            elif self.era_mode == ERA_JINWEN:
                stroke = self.archaic._morph_stroke_to_archaic(raw_stroke, "ZHOU_JINWEN", rng)
            else:  # 小篆 — сглаженные дуги, как в печатях
                stroke = self._morph_to_seal(raw_stroke)

            pts = [
                (margin + (pt[0] - cx) * sx, margin + (pt[1] - cy) * sy)
                for pt in stroke
            ]
            for i in range(len(pts) - 1):
                draw.line([pts[i], pts[i + 1]], fill=255, width=base_w)

            if self.era_mode == ERA_JINWEN and pts:
                # 肥笔 — литые утолщения на концах штрихов
                for end_pt in (pts[0], pts[-1]):
                    r_blob = max(2, int(base_w * 0.7))
                    draw.ellipse(
                        [end_pt[0] - r_blob, end_pt[1] - r_blob,
                         end_pt[0] + r_blob, end_pt[1] + r_blob],
                        fill=255,
                    )
            elif self.era_mode == ERA_SEAL and pts:
                for end_pt in (pts[0], pts[-1]):
                    r_blob = max(2, int(base_w * 0.6))
                    draw.ellipse(
                        [end_pt[0] - r_blob, end_pt[1] - r_blob,
                         end_pt[0] + r_blob, end_pt[1] + r_blob],
                        fill=255,
                    )

        arr = np.array(canvas, dtype=np.float32) / 255.0
        if arr.max() <= 0.01:
            return None
        return normalize_component_mask(np.clip(arr, 0.0, 1.0))

    @staticmethod
    def _morph_to_seal(stroke: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Морфология сглаживания в стиль 小篆 (печать/стела)."""
        if len(stroke) < 2:
            return stroke
        out: List[Tuple[float, float]] = []
        n = len(stroke)
        for i in range(n - 1):
            x0, y0 = stroke[i]
            x1, y1 = stroke[i + 1]
            mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            dx, dy = x1 - x0, y1 - y0
            seg = math.hypot(dx, dy)
            if seg < 1e-3:
                continue
            nx, ny = -dy / seg, dx / seg
            bow = seg * 0.10
            out.append((x0, y0))
            out.append((mx + nx * bow, my + ny * bow))
        out.append(stroke[-1])
        return out

    # ------------------------------------------------------------------
    def build(self, force: bool = False) -> "PaleoTemplateBank":
        """Строит (или загружает из кэша) банк шаблонов."""
        cache = self._cache_path()
        if not force and os.path.exists(cache):
            try:
                with np.load(cache, allow_pickle=False) as z:
                    if not self._cache_is_current(z):
                        raise ValueError(
                            "кэш собран другой версией дескриптора "
                            f"({z['descriptor_version'] if 'descriptor_version' in z else 'без метки'} "
                            f"!= {self.DESCRIPTOR_VERSION}) — пересобираю"
                        )
                    self.glyphs = json.loads(str(z["glyphs_json"]))
                    self.names = json.loads(str(z["names_json"]))
                    self.frequencies = json.loads(str(z["freqs_json"]))
                    self._chamfer = z["chamfer"]
                    self._hog = z["hog"]
                self.load_calibration()
                logger.info(
                    "⚡ [PALEO-BANK HIT] %s: %d шаблонов загружено из кэша.",
                    self.era_mode, len(self.glyphs)
                )
                return self
            except Exception as e:
                logger.warning("Кэш банка %s повреждён (%s) — пересобираю", self.era_mode, e)

        universe = self._collect_component_universe()
        if not universe:
            return self

        # Оставляем только те компоненты, для которых есть скелет
        repo_keys = set(self.repo.medians.keys())
        available = [(g, c) for g, c in universe if g in repo_keys]
        if not available:
            # Крайний случай: если MakeMeAHanzi недоступен — берём всё
            available = universe

        logger.info(
            "Сборка банка %s: %d компонентов (рендер ~%.0f с)...",
            self.era_mode, len(available), len(available) * 0.0013
        )

        chamfers: List[np.ndarray] = []
        hogs: List[np.ndarray] = []
        kept: List[str] = []
        kept_names: List[str] = []
        kept_freqs: List[int] = []

        name_lookup = self._component_names()

        for glyph, freq in available:
            try:
                mask = self._render_component(glyph)
            except Exception:
                mask = None
            if mask is None:
                continue
            c_vec, h_vec = compute_chamfer_and_hog(mask)
            chamfers.append(c_vec)
            hogs.append(h_vec)
            kept.append(glyph)
            kept_names.append(name_lookup.get(glyph, ""))
            kept_freqs.append(int(freq))

        if not kept:
            return self

        self.glyphs = kept
        self.names = kept_names
        self.frequencies = kept_freqs
        self._chamfer = np.stack(chamfers, axis=0).astype(np.float32)
        self._hog = np.stack(hogs, axis=0).astype(np.float32)
        self.load_calibration()

        try:
            np.savez_compressed(
                cache,
                glyphs_json=np.array(json.dumps(self.glyphs, ensure_ascii=False)),
                names_json=np.array(json.dumps(self.names, ensure_ascii=False)),
                freqs_json=np.array(json.dumps(self.frequencies)),
                chamfer=self._chamfer,
                hog=self._hog,
                descriptor_version=np.array(self.DESCRIPTOR_VERSION),
            )
            logger.info("Банк %s сохранён: %s (%.1f МБ)", self.era_mode, cache,
                        os.path.getsize(cache) / 1e6)
        except Exception as e:
            logger.warning("Не удалось сохранить кэш банка %s: %s", self.era_mode, e)

        return self

    def _component_names(self) -> Dict[str, str]:
        """Английские названия компонентов (из каталога Канси + фонетиков)."""
        out: Dict[str, str] = {}
        try:
            from cangjie_benchmark_forge import (
                KANGXI_214_COMPREHENSIVE_CATALOG, PHONETICS_COMPREHENSIVE_DB
            )
            for _num, base_g, allo_g, _py, _st, eng in KANGXI_214_COMPREHENSIVE_CATALOG:
                out.setdefault(base_g, eng)
                out.setdefault(allo_g, eng)
            for pg, _py, _st in PHONETICS_COMPREHENSIVE_DB:
                out.setdefault(pg, pg)
        except Exception:
            pass
        return out

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.glyphs)

    @property
    def is_ready(self) -> bool:
        return self._chamfer is not None and len(self.glyphs) > 0

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # [v9.4] ЧЕСТНАЯ КАЛИБРОВКА ИЗ ДАННЫХ
    # ------------------------------------------------------------------
    def load_calibration(self, path: Optional[str] = None) -> bool:
        """
        Загружает обученную модель калибровки.

        Модель получена файлом `cangjie_calibration.py` на РЕАЛЬНОМ
        калибровочном наборе: глифы в стиле эпохи, подвергнутые реальным
        доменным деградациям (PDE-диффузия туши, TPS-деформация, каменный
        эстампаж 拓片, киноварная печать), ПЛЮС контроли (шум, пусто,
        перекрёстные пары «глиф A спутан с глифом B»).

        Зачем это нужно: пороги, подобранные на СИНТЕТИЧЕСКОМ бенчмарке
        (где рендерер теста совпадает с рендерером банка), давали 99% на
        мёртвых данных и 0.0 confidence на живых фотографиях.
        """
        if path is None:
            # [v9.6] Авто-выбор САМОЙ СВЕЖЕЙ калибровки.
            #   БЫЛО: жёстко 'calibration_v94.json'. После смены дескриптора
            #   (v96-supersample) модель v9.4 обучена на устаревших
            #   признаках и даёт неверную уверенность.
            #   Теперь: перебираем calibration_v*.json по имени и берём
            #   последний (сортировка по имени = по версии).
            import glob
            import re as _re
            cands = sorted(glob.glob(os.path.join(self.data_dir, "calibration_v*.json")))
            if not cands:
                self._calib_model = None
                return False
            def _ver(p):
                m = _re.search(r"calibration_v(\d+)", os.path.basename(p))
                return int(m.group(1)) if m else -1
            path = max(cands, key=_ver)
        if not os.path.exists(path):
            self._calib_model = None
            return False
        try:
            with open(path, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
            self._calib_model = blob["model"]
            self._calib_mu = np.asarray(blob["feature_mean"], dtype=np.float64)
            self._calib_sd = np.asarray(blob["feature_std"], dtype=np.float64)
            self._calib_threshold = float(blob.get("metrics", {}).get("chosen_threshold", 0.5))
            logger.info("Калибровка загружена: %s (порог %.4f)",
                        os.path.basename(path), self._calib_threshold)
            return True
        except Exception as e:
            logger.warning("Калибровка v9.4 не загрузилась (%s) — работаю на формуле v9.3", e)
            self._calib_model = None
            return False

    def _calibrated_confidence(self, feats: np.ndarray) -> Optional[float]:
        """Применяет обученную логистическую регрессию к вектору признаков."""
        if self._calib_model is None or self._calib_mu is None or self._calib_sd is None:
            return None
        try:
            x = (feats - self._calib_mu) / np.where(self._calib_sd < 1e-9, 1.0, self._calib_sd)
            coef = np.asarray(self._calib_model["coef"], dtype=np.float64)
            z = float(self._calib_model["intercept"]) + float(np.dot(coef, x))
            return float(1.0 / (1.0 + np.exp(-max(-40.0, min(40.0, z)))))
        except Exception as e:
            logger.debug("Калибровка не применилась: %s", e)
            return None

    def _features_for(self, scores: np.ndarray, top_idx: int, second_score: float,
                      posterior_top: float, query_density: float) -> np.ndarray:
        """Собирает тот же вектор признаков, что и cangjie_calibration."""
        bank_mean = float(np.mean(scores))
        bank_std = float(np.std(scores))
        if bank_std > 1e-9:
            z_top = (float(scores[top_idx]) - bank_mean) / bank_std
            z_gap = (float(scores[top_idx]) - second_score) / bank_std
        else:
            z_top = z_gap = 0.0
        abs_fit = float(np.clip((float(scores[top_idx]) - 0.55) / 0.45, 0.0, 1.0))
        # Плотность намеренно НЕ используется (см. комментарий в
        # cangjie_calibration.py): после density_normalize шум получает
        # ту же плотность, что и жирный живой знак, и признак учил
        # модель понижать уверенность у настоящих кайшу-глифов.
        return np.array([z_top, z_gap, abs_fit, posterior_top], dtype=np.float64)

    def match(
        self,
        ink_64: np.ndarray,
        top_k: int = 5,
        temperature: float = 0.045,
        recognized_threshold: float = 0.30,
        ambiguous_margin: float = 0.035,
    ) -> MatchResult:
        """
        Сопоставляет бинарную маску компонента с банком шаблонов.

        КАЛИБРОВКА УВЕРЕННОСТИ [v9.3]:
            fit        = нормированное качество абсолютного совпадения top1
                         со своим шаблоном (0 = не похоже, 1 = точное совпадение)
            separation = насколько top1 оторвался от второго кандидата (в σ банка)
            confidence = fit × (0.35 + 0.65 × separation)

        ПОЧЕМУ posterior_top НЕ входит в confidence [v9.3]:
            posterior = softmax по top-K СЫРЫХ скоров с фиксированной T=0.045.
            У банка из 4720 шаблонов скоры слипаются: у одинокого знака (馬)
            разрыв top1−top2 ≈ 0.06, у распространённого (日, рядом 曰/目/田)
            ≈ 0.006. При фиксированной T posterior_top выходил ≈0.16 для
            ЛЮБОГО входа — и для идеального совпадения, и для белого шума.
            Умножение на него убивало истинную уверенность: ИДЕАЛЬНОЕ
            совпадение получало confidence 0.10 и статус UNRECOGNIZED.
            В бенчмарке это давало 286/880 = 32.5% ВЕРНЫХ ответов,
            помеченных как «не опознано». Теперь posterior остаётся в выдаче
            как относительное распределение масс (для показа альтернатив),
            но не влияет на саму уверенность.

        Нижней границы у confidence НЕТ: мусор получает ~0.05-0.15, а не 0.55.
        """
        if not self.is_ready:
            return MatchResult(
                best=None, confidence=0.0, status="UNRECOGNIZED",
                margin=0.0, raw_score=0.0, candidates=[], era=self.era_mode
            )

        q_chamfer, q_hog = compute_chamfer_and_hog(ink_64)

        # Косинусная близость: оба вектора L2-нормированы -> это dot product
        sim_c = self._chamfer @ q_chamfer.astype(np.float32)
        sim_h = self._hog @ q_hog.astype(np.float32)
        scores = 0.55 * sim_c + 0.45 * sim_h

        # Апостериор считаем ТОЛЬКО по верхним кандидатам.
        # Если softmax вести по всем 4720 шаблонам, «хвост» почти одинаковых
        # по форме глифов (讠/訁/言 и десятки подобных) съедает массу и
        # уверенность правильного ответа падает до ~0.005. По top-K хвост
        # не мешает, а реальная неоднозначность (два похожих КАНДИДАТА) —
        # честно отражается падением апостериора.
        k = int(min(max(top_k, 8), len(scores)))
        order = np.argpartition(-scores, k - 1)[:k]
        order = order[np.argsort(-scores[order])]
        top_scores = scores[order]

        z = (top_scores - float(top_scores[0])) / max(1e-6, temperature)
        np.exp(z, out=z)
        posterior = z / z.sum()

        candidates: List[MatchCandidate] = []
        for rank, idx in enumerate(order):
            candidates.append(MatchCandidate(
                glyph=self.glyphs[int(idx)],
                score=float(scores[int(idx)]),
                posterior=float(posterior[rank]),
                name=self.names[int(idx)] if int(idx) < len(self.names) else "",
            ))

        best = candidates[0]
        raw_score = best.score
        margin = raw_score - candidates[1].score if len(candidates) > 1 else raw_score

        # fit: МАСШТАБ-ИНВАРИАНТНОЕ качество совпадения.
        #
        # [v9.2] БЫЛО: fit = clip((raw_score - 0.30) / 0.50, 0, 1)
        #   Калибровка была подогнана под СИНТЕТИЧЕСКИЙ бенчмарк, где
        #   рендерер шаблона и рендерер теста — один и тот же код, поэтому
        #   raw_score почти всегда 0.62..1.0. На ЖИВОМ фото максимум
        #   raw_score по всем 4720 шаблонам был 0.159, т.е. fit = 0 и
        #   confidence = 0.0 для ВСЕГО подряд. Бенчмарк показывал 97.8%,
        #   а на реальной фотографии система молчала.
        #
        # ПОСЛЕ починки плотности (density_normalize) шкала стала
        #   осмысленной: точное совпадение с шаблоном даёт raw1 = 1.000,
        #   реальная деградированная фотография — ~0.6..0.9, шум — ~0.63.
        #   Поэтому fit — это АБСОЛЮТНОЕ качество совпадения:
        #     0.55 -> 0 (не похоже), 1.00 -> 1 (точное совпадение)
        abs_fit = float(np.clip((raw_score - 0.55) / 0.45, 0.0, 1.0))

        # separation: насколько top1 ОТОРВАЛСЯ от остального банка
        # (z-скор разрыва). Для визуально уникального знака (馬) большой,
        # для распространённого (日, вокруг которого 囗/囚/田) — маленький.
        # Это честная неопределённость, а не ошибка.
        bank_mean = float(np.mean(scores))
        bank_std = float(np.std(scores))
        if bank_std > 1e-9:
            z_gap = (raw_score - float(candidates[1].score)) / bank_std if len(candidates) > 1 else 0.0
        else:
            z_gap = 0.0
        separation = float(np.clip(z_gap / 0.35, 0.0, 1.0))

        # [v9.3] ЧЕСТНАЯ КАЛИБРОВКА: posterior_top УБРАН из формулы.
        #
        # ПОЧЕМУ (это и есть главный баг v9.2):
        #     posterior_top = softmax по top-K СЫРЫХ скоров с фиксированной
        #     температурой 0.045. Но у банка из 4720 шаблонов скоры СЛИПЛИСЬ:
        #     у зрительно «одинокого» знака (馬) разрыв top1-top2 ≈ 0.06,
        #     а у распространённого (日, рядом 曰/目/田/由/甲) ≈ 0.006.
        #     С фиксированной температурой posterior_top получался ≈0.16
        #     ДЛЯ ЛЮБОГО входа — и для ИДЕАЛЬНОГО совпадения, и для белого
        #     шума. Умножение на него убивало истинную уверенность:
        #     ИДЕАЛЬНОЕ совпадение 日 получало confidence 0.10 и статус
        #     UNRECOGNIZED. В бенчмарке это давало 286/880 = 32.5%
        #     ВЕРНЫХ ответов, помеченных как «не опознано» (кайшу — 282/300).
        #
        # КАК ТЕПЕРЬ:
        #     posterior остаётся в выдаче как ОТНОСИТЕЛЬНОЕ распределение масс
        #     среди кандидатов (для показа альтернатив палеографу), но больше
        #     НЕ решает, уверены мы или нет.
        #     confidence = fit × (0.35 + 0.65 × separation)
        #       fit        — АБСОЛЮТНОЕ качество совпадения top1 со своим
        #                    шаблоном (0 = не похоже, 1 = точное совпадение).
        #       separation — насколько top1 оторвался от второго кандидата
        #                    (в σ банка): «а не подменили ли мы знак похожим
        #                    соседом?»
        #     Ведущий коэффициент 0.35 означает: даже когда рядом есть почти
        #     идентичный сосед (曰 против 日), ИДЕАЛЬНОЕ совпадение формы всё
        #     равно даёт 0.35+ — палеограф получает сигнал, а не «не знаю».
        #     Но посредственный матч всё равно не пройдёт: fit у него мал.
        fit = abs_fit
        confidence = float(np.clip(fit * (0.35 + 0.65 * separation), 0.0, 1.0))

        # [v9.3] Порог RECOGNIZED больше не требует raw_score >= 0.90.
        # Старое условие было калибровано под синтетику (там raw_score
        # доходил до 1.0). На ЖИВОМ деградированном фото raw_score редко
        # превышает 0.85 — при старом условии палеограф получал «AMBIGUOUS»
        # на ВСЁ подряд. Теперь решение принимает confidence (он уже
        # включает fit), а fit >= 0.55 означает raw_score >= 0.80, что для
        # центрированной корреляции полей — реальное совпадение формы.
        # [v9.4] ЧЕСТНАЯ КАЛИБРОВКА: если модель обучена и загружена —
        # решение принимает ОНА, а не зашитые пороги.
        calib_conf = None
        if self._calib_model is not None:
            try:
                feats = self._features_for(
                    scores, int(order[0]),
                    float(candidates[1].score) if len(candidates) > 1 else 0.0,
                    float(best.posterior),
                    float(np.mean(ink_64)),
                )
                calib_conf = self._calibrated_confidence(feats)
            except Exception as e:
                logger.debug("Применение калибровки не удалось: %s", e)

        if calib_conf is not None:
            confidence = float(np.clip(calib_conf, 0.0, 1.0))
            thr = self._calib_threshold
            if confidence >= thr:
                status = "RECOGNIZED"
            elif confidence >= thr * 0.5:
                status = "AMBIGUOUS"
            else:
                status = "UNRECOGNIZED"
        elif confidence >= recognized_threshold and fit >= 0.55:
            status = "RECOGNIZED"
        elif fit < 0.20 or confidence < 0.12:
            status = "UNRECOGNIZED"
        else:
            status = "AMBIGUOUS"

        return MatchResult(
            best=best,
            confidence=confidence,
            status=status,
            margin=float(margin),
            raw_score=float(raw_score),
            candidates=candidates,
            era=self.era_mode,
        )


# ============================================================================
# 4. МНОГОЭРА-АНАЛИЗ (какая письменность ближе к снимку)
# ============================================================================

def match_across_eras(
    ink_64: np.ndarray,
    data_dir: str = "./data",
    repo=None,
    top_k: int = 3,
    eras: Sequence[str] = (ERA_KAISHU, ERA_JIAGUWEN, ERA_JINWEN),
) -> Dict[str, Any]:
    """
    Сравнивает один компонент со всеми эрами и возвращает ранг по «похожести».

    Для палеографа это самостоятельный сигнал: по стилистике резьбы можно
    предположить эпоху и письменность даже до идентификации знака.
    """
    out: Dict[str, Any] = {"eras": [], "best_era": None, "era_scores": {}}
    best_overall: Optional[Tuple[str, float]] = None

    for era in eras:
        bank = get_bank(era, data_dir=data_dir, repo=repo)
        if not bank.is_ready:
            continue
        res = bank.match(ink_64, top_k=top_k)
        out["eras"].append({
            "era": era,
            "confidence": round(res.confidence, 4),
            "raw_score": round(res.raw_score, 4),
            "best_glyph": res.best.glyph if res.best else None,
            "status": res.status,
        })
        out["era_scores"][era] = round(res.raw_score, 4)
        if best_overall is None or res.raw_score > best_overall[1]:
            best_overall = (era, res.raw_score)

    if best_overall:
        out["best_era"] = best_overall[0]
    return out


# ============================================================================
# 5. ГЛОБАЛЬНЫЙ КЭШ БАНКОВ
# ============================================================================

_BANK_CACHE: Dict[Tuple[str, str], PaleoTemplateBank] = {}


def get_bank(
    era_mode: str = ERA_KAISHU,
    data_dir: str = "./data",
    repo=None,
    build: bool = True,
) -> PaleoTemplateBank:
    """Возвращает (и при необходимости строит) банк шаблонов для эры."""
    key = (era_mode, os.path.abspath(data_dir))
    bank = _BANK_CACHE.get(key)
    if bank is None:
        bank = PaleoTemplateBank(era_mode=era_mode, data_dir=data_dir, repo=repo)
        _BANK_CACHE[key] = bank
    if build and not bank.is_ready:
        bank.build()
    return bank


def clear_bank_cache() -> None:
    _BANK_CACHE.clear()
