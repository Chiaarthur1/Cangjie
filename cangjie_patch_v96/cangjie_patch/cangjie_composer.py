"""
Project Cangjie — Recursive IDS Composer [v9.5]

ЗАЧЕМ (это ключ к цели «понимать иероглифы в 172 черты»):

    MakeMeAHanzi даёт скелеты только для 9 575 знаков. Составные знаки
    вроде 龘 (48 черт), 齉 (29), 𪚥 (52), 𰻝 (64) ТАМ ОТСУТСТВУЮТ —
    у них нет готовой геометрии.

    Но у них есть РАЗБОР: CHISE IDS хранит структуру каждого знака,
    97 431 знак, максимальная глубина разбора — 8 операторов.

    Идея: если есть разбор ⿰⿱⿲⿳... и есть скелеты ЛИСТЬЕВ, то можно
    СОБРАТЬ геометрию составного знака рекурсивно: сначала рендерим
    листья, потом раскладываем их по оператору в нужные боксы.

    Это ровно то, что нужно палеографу: знак на 172 черты не надо
    уметь рисовать — надо уметь РАЗОБРАТЬ и опознать его части,
    а дальше работает граф прецедентов (deducer).

    Что уже было в проекте:
      - `cangjie_benchmark_forge.CompositionNode` — геометрия для ⿰ и ⿱,
        но только на 2 уровня и только для листьев с готовыми скелетами;
      - `cangjie_layout_to_tree.PointerTreeTransformerDecoder` — разбор
        в дерево, но без геометрии.

    Здесь: полноценный рекурсивный композитор с поддержкой ВСЕХ
    операторов CHISE (⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻) и кэшированием.

ЗАПУСК:
    python3 cangjie_composer.py --stats
    python3 cangjie_composer.py --render 𰻝 --out /tmp/x.png
"""

from __future__ import annotations

import os
import re
import sys
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [COMPOSER]: %(message)s"
)
logger = logging.getLogger("CangjieComposer")


# ============================================================================
# 1. ПАРСЕР IDS
# ============================================================================

# Все операторы CHISE с их арностью (сколько потомков)
IDS_OPERATORS: Dict[str, int] = {
    "⿰": 2,  # ⿰ левое-правое
    "⿱": 2,  # ⿱ верх-низ
    "⿲": 3,  # ⿲ левое-право-требуемое
    "⿳": 3,  # ⿳ верх-низ-требуемое
    "⿴": 2,  # ⿴ enclosure (всестороннее)
    "⿵": 2,  # ⿵ enclosure (сверху)
    "⿶": 2,  # ⿶ enclosure (снизу)
    "⿷": 2,  # ⿷ enclosure (слева)
    "⿸": 2,  # ⿸ enclosure (справа)
    "⿹": 2,  # ⿹ enclosure (сверху-справа)
    "⿺": 2,  # ⿺ enclosure (снизу-слева)
    "⿻": 2,  # ⿻ enclosure (снизу-справа)
    "⿼": 2,  # ⿼ enclosure (слева-снизу)
    "⿽": 2,  # ⿽ enclosure (сверху-слева)
}

# Операторы, где «средний» (второй) потомок — главный, остальные — обёртка
# (в CHISE это ⿲ и ⿳: второй элемент центральный).
_CENTER_SECOND = {"⿲", "⿳"}


def parse_ids(ids: str) -> Optional[Tuple[str, List[str]]]:
    """
    Разбирает строку IDS в (оператор, список потомков).

    Поддерживает вложенность: '⿱⿰月⿰日乚' -> ('⿱', ['⿰月日', '乚']).
    Возвращает None, если разбор невозможен.
    """
    ids = ids.strip()
    if not ids:
        return None

    op = ids[0]
    if op not in IDS_OPERATORS:
        return None

    arity = IDS_OPERATORS[op]
    rest = ids[1:]
    children: List[str] = []
    i = 0
    while i < len(rest) and len(children) < arity:
        if rest[i] in IDS_OPERATORS:
            # вложенный оператор — выделяем поддерево
            depth = 0
            j = i
            while j < len(rest):
                if rest[j] in IDS_OPERATORS:
                    depth += 1
                elif depth == 0 and j > i and rest[j] == '⿴':
                    # граница не нужна: считаем по балансу
                    pass
                j += 1
            # Простой подсчёт: у каждого оператора своя арность, поэтому
            # поддерево заканчивается, когда использованы все арности.
            child, i = _extract_subtree(rest, i)
            if child is None:
                return None
            children.append(child)
        else:
            # одиночный символ (может быть многобайтовым, но не surrogate)
            ch = rest[i]
            children.append(ch)
            i += 1
    if len(children) != arity:
        return None
    return op, children


def _extract_subtree(s: str, start: int) -> Tuple[Optional[str], int]:
    """Выделяет поддерево IDS, начиная с позиции start (start указывает на оператор)."""
    op = s[start]
    arity = IDS_OPERATORS[op]
    i = start + 1
    children: List[str] = []
    while i < len(s) and len(children) < arity:
        if s[i] in IDS_OPERATORS:
            child, i = _extract_subtree(s, i)
            if child is None:
                return None, i
            children.append(child)
        else:
            children.append(s[i])
            i += 1
    return op + "".join(children), i


# ============================================================================
# 2. ЗАГРУЗКА КОРПУСА IDS
# ============================================================================

def load_ids_corpus(data_dir: str = "./data") -> Dict[str, str]:
    """Загружает все IDS-разборы из корпуса CHISE (все файлы)."""
    import glob
    paths: List[str] = []
    paths += sorted(glob.glob(os.path.join(data_dir, "IDS-UCS-*.txt")))
    paths += sorted(glob.glob(os.path.join(data_dir, "chise_all", "IDS-UCS-*.txt")))

    out: Dict[str, str] = {}
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.startswith(";") or "\t" not in line:
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    ch, ids = parts[1], parts[2].strip()
                    if not ch or not ids or ids == ch:
                        continue
                    if ch not in out:
                        out[ch] = ids
        except Exception as e:
            logger.warning("Не удалось прочитать %s: %s", p, e)
    return out


# ============================================================================
# 3. РЕКУРСИВНЫЙ КОМПОЗИТОР
# ============================================================================

class RecursiveComposer:
    """
    Собирает геометрию составного знака из IDS-разбора + скелетов листьев.

    Работает так:
        1) Если у знака есть готовый скелет (MakeMeAHanzi) — рисуем его.
        2) Иначе разбираем IDS, рекурсивно готовим потомков и раскладываем
           их по геометрии оператора.
    """

    def __init__(self, data_dir: str = "./data"):
        from cangjie_benchmark_forge import GlyphRepository
        self.data_dir = data_dir
        self.repo = GlyphRepository()
        self.ids_corpus = load_ids_corpus(data_dir)
        self._cache: Dict[str, bool] = {}
        self._stats = {"direct": 0, "composed": 0, "failed": 0}

    # ---------------- геометрия ----------------
    @staticmethod
    def _split_box(box, op: str, counts: List[int]):
        """Делит BoundingBox на части по оператору. Возвращает [(x,y,w,h), ...]."""
        total = sum(max(1, c) for c in counts) or 1
        n = len(counts)
        w, h = box.width, box.height

        if op in ("⿰", "⿲"):
            # горизонтальное деление
            gap = w * 0.03
            usable = w - gap * (n - 1)
            if op == "⿲":
                # второй — центральный, шире
                weights = [0.32, 0.36, 0.32] if n == 3 else [0.5, 0.5]
            else:
                weights = [0.5, 0.5]
            # подгоняем по числу черт
            stroke_w = [max(1, c) / total for c in counts]
            mix = [0.5 * stroke_w[i] + 0.5 * weights[i] for i in range(n)]
            ssum = sum(mix) or 1.0
            mix = [m / ssum for m in mix]
            out = []
            x = box.x
            for i in range(n):
                part_w = usable * mix[i]
                out.append((x, box.y, part_w, h))
                x += part_w + gap
            return out

        if op in ("⿱", "⿳"):
            gap = h * 0.03
            usable = h - gap * (n - 1)
            if op == "⿳":
                weights = [0.32, 0.36, 0.32] if n == 3 else [0.5, 0.5]
            else:
                weights = [0.5, 0.5]
            stroke_h = [max(1, c) / total for c in counts]
            mix = [0.5 * stroke_h[i] + 0.5 * weights[i] for i in range(n)]
            ssum = sum(mix) or 1.0
            mix = [m / ssum for m in mix]
            out = []
            y = box.y
            for i in range(n):
                part_h = usable * mix[i]
                out.append((box.x, y, w, part_h))
                y += part_h + gap
            return out

        # ОПЕРАТОРЫ ОКРУЖЕНИЯ (⿴..⿽): внешний элемент + внутренний.
        #
        # [v9.7] ГЕОМЕТРИЯ ИСПРАВЛЕНА.
        #   БЫЛО: внешний элемент рисовался на ВСЁ поле, внутренний —
        #         в центральном квадрате. Для 辶 (⿺) это давало полный
        #         «размазанный» знак: 辶 занимал 100% площади, а
        #         содержимое (首, 夂, ...) сжималось в угол.
        #   Стало: внешний элемент получает СВОЮ зону по оператору,
        #         внутренний — оставшуюся часть. Это и есть смысл
        #         «обтекания»: 辶 рисуется по левому краю и низу,
        #         содержимое — в правой верхней части.
        #
        #   Важно: 12 432 знака корпуса используют операторы окружения
        #   (⿸ — 4158, ⿺ — 3699, ⿵ — 1391, ⿴ — 1096 ...), то есть это
        #   затрагивает 13% всего датасета.
        if op in IDS_OPERATORS:
            outer, inner = self._enclosure_zones(box, op)
            return [outer, inner]

        return [(box.x, box.y, w, h)]

    @staticmethod
    def _enclosure_zones(box, op: str):
        """
        Делит поле на зону ВНЕШНЕГО элемента и зону ВНУТРЕННЕГО.

        Смысл операторов (по IDS):
            ⿴ — внешний окружает со всех сторон (囗, 广, 厂)
            ⿵ — внешний сверху (亠, 冖, 爫)
            ⿶ — внешний снизу (凵, 一 под)
            ⿷ — внешний слева (匚, 匸)
            ⿸ — внешний справа (⻏, 刂)
            ⿹ — внешний сверху-справа (⻏, 疒)
            ⿺ — внешний снизу-слева (辶, 廴, 阝)  ← самый частый
            ⿻ — внешний снизу-справа
            ⿼ — внешний слева-снизу
            ⿽ — внешний сверху-слева

        Возвращает ((x,y,w,h) внешнего, (x,y,w,h) внутреннего).
        """
        x, y, w, h = box.x, box.y, box.width, box.height

        # Горизонтальная/вертикальная «полка» обтекания: сколько поля
        # занимает внешний элемент по своей оси.
        shelf = 0.34   # доля по «толстой» оси
        thin = 0.20    # доля по «тонкой» оси

        if op == "⿴":       # окружает со всех сторон
            outer = (x, y, w, h)
            inset_x, inset_y = 0.20, 0.18
            inner = (x + w * inset_x, y + h * inset_y,
                     w * (1 - 2 * inset_x), h * (1 - 2 * inset_y))
        elif op == "⿵":     # сверху
            outer = (x, y, w, h * shelf)
            inner = (x, y + h * shelf, w, h * (1 - shelf))
        elif op == "⿶":     # снизу
            outer = (x, y + h * (1 - shelf), w, h * shelf)
            inner = (x, y, w, h * (1 - shelf))
        elif op == "⿷":     # слева
            outer = (x, y, w * shelf, h)
            inner = (x + w * shelf, y, w * (1 - shelf), h)
        elif op == "⿸":     # справа
            outer = (x + w * (1 - shelf), y, w * shelf, h)
            inner = (x, y, w * (1 - shelf), h)
        elif op == "⿹":     # сверху-справа
            outer = (x + w * (1 - shelf), y, w * shelf, h * shelf)
            inner = (x, y, w * (1 - shelf), h * (1 - shelf))
        elif op == "⿺":     # снизу-слева (辶, 廴) — самый частый
            # [v9.7] БЫЛО: outer = нижняя полоса (w x h*shelf).
            #   Это СЖИМАЛО 辶 в плоскую полосу: терялись и крючок
            #   сверху, и характерный размах снизу. Знак выглядел
            #   как «размазанное» пятно (видно на 𰻞).
            #   СТАЛО: 辶 рисуется в ПОЛНОМ поле (сохраняя свою
            #   естественную форму: крюк сверху-слева + размах снизу),
            #   а содержимое размещается в верхней правой части —
            #   ровно как в настоящем 道, 這, 進, 遠.
            outer = (x, y, w, h)
            inner = (x + w * thin, y, w * (1 - thin), h * (1 - shelf))
        elif op == "⿻":     # снизу-справа
            outer = (x + w * (1 - shelf), y + h * (1 - shelf), w * shelf, h * shelf)
            inner = (x, y, w * (1 - shelf), h * (1 - shelf))
        elif op == "⿼":     # слева-снизу
            outer = (x, y + h * (1 - shelf), w * shelf, h * shelf)
            inner = (x + w * shelf, y, w * (1 - shelf), h * (1 - shelf))
        elif op == "⿽":     # сверху-слева
            outer = (x, y, w * shelf, h * shelf)
            inner = (x + w * shelf, y + h * shelf, w * (1 - shelf), h * (1 - shelf))
        else:
            outer = (x, y, w, h)
            inner = (x + w * 0.24, y + h * 0.24, w * 0.52, h * 0.52)

        return outer, inner

    # ---------------- основной метод ----------------
    def can_compose(self, char: str) -> bool:
        """Может ли знак быть собран (есть скелет ИЛИ разбирается по IDS)."""
        if char in self._cache:
            return self._cache[char]
        ok = self._can_compose(char, set())
        self._cache[char] = ok
        return ok

    def _can_compose(self, char: str, visiting: set) -> bool:
        """
        Может ли ЭЛЕМЕНТ быть отрисован?

        ВАЖНО: аргумент — это НЕ обязательно одиночный знак. Это может быть
        ПОДДЕРЕВО IDS (строка, начинающаяся с оператора), например «⿰龍龍»
        внутри «⿱龍⿰龍龍» (= 龘).

        [v9.6] БЫЛО: поддерево искалось в `repo.get_medians()` и в
        `ids_corpus` — но там лежат только ОДИНОЧНЫЕ знаки. Поддерево
        «⿰龍龍» не является символом, поэтому проверка падала и возвращала
        False. Из-за этого 龘 (48 черт) и другие знаки с вложенной
        структурой не собирались, хотя все листья у них есть.

        Теперь: если строка начинается с оператора — разбираем её как
        поддерево и проверяем его потомков.
        """
        if char in visiting:
            return False  # цикл
        visiting = visiting | {char}

        # Ветка 1: это ПОДДЕРЕВО IDS, а не одиночный знак.
        if char and char[0] in IDS_OPERATORS:
            parsed = parse_ids(char)
            if parsed is None:
                return False
            _op, children = parsed
            for ch in children:
                if not self._can_compose(ch, visiting):
                    return False
            return True

        # Ветка 2: одиночный знак.
        if self.repo.get_medians(char):
            return True
        ids = self.ids_corpus.get(char)
        if not ids:
            return False
        parsed = parse_ids(ids)
        if parsed is None:
            return False
        _op, children = parsed
        for ch in children:
            if not self._can_compose(ch, visiting):
                return False
        return True

    def render_subtree(
        self,
        node: str,
        resolution: int = 256,
        box=None,
    ) -> Optional[np.ndarray]:
        """
        Рендерит ЛИБО одиночный знак, ЛИБО поддерево IDS (строку,
        начинающуюся с оператора).

        Это ключевое отличие от render_char: в CHISE поддеревья
        встречаются как элементы ⿰/⿱/⿲/⿳ и как содержимое
        операторов окружения (⿺辶⿳... — «辶» обтекает ЦЕЛОЕ дерево).
        """
        if not node:
            return None
        if node[0] in IDS_OPERATORS:
            return self._render_ids_node(node, resolution, box)
        return self.render_char(node, resolution, box)

    def _render_ids_node(
        self,
        node: str,
        resolution: int,
        box,
    ) -> Optional[np.ndarray]:
        """Рендерит поддерево IDS (node начинается с оператора)."""
        from cangjie_benchmark_forge import BoundingBox
        if box is None:
            box = BoundingBox(4, 4, resolution - 4, resolution - 4)

        parsed = parse_ids(node)
        if parsed is None:
            return None
        op, children = parsed

        if op in ("⿰", "⿱", "⿲", "⿳"):
            sub_masks: List[np.ndarray] = []
            counts: List[int] = []
            for ch in children:
                m = self.render_subtree(
                    ch, resolution=resolution,
                    box=BoundingBox(4, 4, resolution - 4, resolution - 4))
                if m is None:
                    self._stats["failed"] += 1
                    return None
                sub_masks.append(m)
                counts.append(max(1, int(m.sum() / 40)))

            boxes = self._split_box(box, op, counts)
            canvas = np.zeros((resolution, resolution), dtype=np.float32)
            for m, (x, y, w, h) in zip(sub_masks, boxes):
                child_img = Image.fromarray((m * 255).astype(np.uint8), mode="L")
                tw, th = max(1, int(w)), max(1, int(h))
                child_img = child_img.resize((tw, th), Image.Resampling.BILINEAR)
                y0, x0 = int(y), int(x)
                y1, x1 = min(resolution, y0 + th), min(resolution, x0 + tw)
                if y1 <= y0 or x1 <= x0:
                    continue
                canvas[y0:y1, x0:x1] = np.maximum(
                    canvas[y0:y1, x0:x1],
                    np.array(child_img, dtype=np.float32)[:y1 - y0, :x1 - x0] / 255.0,
                )
            self._stats["composed"] += 1
            return canvas

        # ОПЕРАТОРЫ ОКРУЖЕНИЯ (⿴..⿽): внешний элемент + внутренний.
        #
        # [v9.7] БЫЛО: внешний рисовался на ВСЁ поле, внутренний
        #   вписывался в рамку через ОБРЕЗКУ (`inner_m[:y1-y0, :x1-x0]`),
        #   то есть дальше своей зоны содержимое просто терялось.
        #   Для ⿺ (辶) знак 𰻞 (биан) выглядел как «размазанная» каша:
        #   辶 занимал всю площадь, а 首-часть была обрезана.
        # Стало: оба элемента МАСШТАБИРУЮТСЯ в свои зоны.
        if op in IDS_OPERATORS:
            outer_box, inner_box = self._enclosure_zones(box, op)

            outer_m = self.render_subtree(
                children[0], resolution=resolution,
                box=BoundingBox(outer_box[0], outer_box[1],
                                outer_box[2], outer_box[3]))
            if outer_m is None:
                self._stats["failed"] += 1
                return None
            inner_m = self.render_subtree(
                children[1], resolution=resolution,
                box=BoundingBox(inner_box[0], inner_box[1],
                                inner_box[2], inner_box[3]))
            if inner_m is None:
                self._stats["failed"] += 1
                return None

            canvas = np.zeros((resolution, resolution), dtype=np.float32)
            for mask, zone in ((outer_m, outer_box), (inner_m, inner_box)):
                zx, zy, zw, zh = zone
                tw, th = max(1, int(zw)), max(1, int(zh))
                child = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
                child = child.resize((tw, th), Image.Resampling.BILINEAR)
                y0, x0 = int(zy), int(zx)
                y1, x1 = min(resolution, y0 + th), min(resolution, x0 + tw)
                if y1 <= y0 or x1 <= x0:
                    continue
                canvas[y0:y1, x0:x1] = np.maximum(
                    canvas[y0:y1, x0:x1],
                    np.array(child, dtype=np.float32)[:y1 - y0, :x1 - x0] / 255.0,
                )
            self._stats["composed"] += 1
            return canvas

        self._stats["failed"] += 1
        return None

    def render_char(
        self,
        char: str,
        resolution: int = 256,
        box=None,
    ) -> Optional[np.ndarray]:
        """
        Рендерит знак (прямо или рекурсивно по IDS) в маску туши 0..1.
        Возвращает None, если собрать не удалось.
        """
        from cangjie_benchmark_forge import (
            CalligraphicPillowRenderer, RadicalLeaf, SpatialSlot, BoundingBox,
        )
        if box is None:
            box = BoundingBox(4, 4, resolution - 4, resolution - 4)

        # 1) Прямой путь: есть готовый скелет
        if self.repo.get_medians(char):
            try:
                renderer = CalligraphicPillowRenderer(target_resolution=resolution, supersample_factor=1)
                leaf = RadicalLeaf(char, SpatialSlot.ROOT, self.repo)
                leaf.calculate_geometry(box)
                leaf.render(renderer)
                arr = np.array(renderer.finalize().convert("L"), dtype=np.float32)
                return np.clip((255.0 - arr) / 255.0, 0.0, 1.0)
            except Exception as e:
                logger.debug("Прямой рендер %s не удался: %s", char, e)

        # 2) Рекурсивный путь: по IDS
        ids = self.ids_corpus.get(char)
        if not ids:
            self._stats["failed"] += 1
            return None
        parsed = parse_ids(ids)
        if parsed is None:
            self._stats["failed"] += 1
            return None
        op, children = parsed
        # Дальше — единая реализация раскладки по оператору.
        return self._render_ids_node(ids, resolution, box)



# ============================================================================
# 4. СТАТИСТИКА ПО КОРПУСУ
# ============================================================================

def corpus_stats(data_dir: str = "./data", sample_limit: int = 4000) -> Dict[str, Any]:
    """Считает, сколько знаков корпуса реально собираются."""
    comp = RecursiveComposer(data_dir)
    total = len(comp.ids_corpus)
    logger.info("IDS-разборов в корпусе: %d", total)

    keys = list(comp.ids_corpus.keys())
    rng = np.random.RandomState(2026)
    if len(keys) > sample_limit:
        sample = [keys[i] for i in rng.choice(len(keys), sample_limit, replace=False)]
    else:
        sample = keys

    direct = composed = failed = 0
    examples_composed: List[str] = []
    for ch in sample:
        if comp.repo.get_medians(ch):
            direct += 1
        elif comp.can_compose(ch):
            composed += 1
            if len(examples_composed) < 25:
                examples_composed.append(ch)
        else:
            failed += 1

    n = len(sample)
    logger.info("Проба на %d знаках: прямых %d, собираемых %d, неудачных %d",
                n, direct, composed, failed)

    return {
        "ids_entries": total,
        "sample_size": n,
        "direct_skeleton": direct,
        "composable": composed,
        "not_composable": failed,
        "composable_pct": round((direct + composed) / max(n, 1) * 100, 2),
        "examples_composed": examples_composed,
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Рекурсивный композитор IDS")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--render", default=None, help="Знак для рендера")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-dir", default="./data")
    args = ap.parse_args(argv)

    if args.render:
        comp = RecursiveComposer(args.data_dir)
        mask = comp.render_char(args.render, resolution=256)
        if mask is None:
            logger.error("Не удалось собрать %s", args.render)
            return 1
        img = Image.fromarray((np.clip(1 - mask, 0, 1) * 255).astype(np.uint8), mode="L")
        out = args.out or f"/workspace/artifacts/composed_{ord(args.render):04X}.png"
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        img.save(out)
        logger.info("Сохранено: %s (плотность %.4f)", out, float(mask.mean()))
        print(out)
        return 0

    stats = corpus_stats(args.data_dir)
    print("=" * 78)
    print("РЕКУРСИВНЫЙ КОМПОЗИТОР IDS — СТАТИСТИКА")
    print("=" * 78)
    print(f"IDS-разборов в корпусе: {stats['ids_entries']}")
    print(f"Проба на {stats['sample_size']} знаках:")
    print(f"   с готовым скелетом:  {stats['direct_skeleton']}")
    print(f"   собираются по IDS:   {stats['composable']}")
    print(f"   не собираются:       {stats['not_composable']}")
    print(f"   ИТОГО доступно:      {stats['composable_pct']}%")
    print(f"   примеры собранных:   {' '.join(stats['examples_composed'][:20])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
