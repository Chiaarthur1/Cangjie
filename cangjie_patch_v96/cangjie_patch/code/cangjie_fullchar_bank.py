"""
Project Cangjie — банк ПОЛНЫХ знаков [v9.6]

ЗАЧЕМ (это ключ к цели «понимать иероглифы в 172 черты»):

    Банк `cangjie_paleo_matcher` содержит 4 720 ЛИСТОВЫХ компонентов
    (⿰-листья, отдельные радикалы). Это нужно, чтобы разложить знак
    на части. Но матчить ПО ФОТО нужно ЦЕЛЫЙ знак.

    Сейчас полный знак в банке отсутствует:
        '龘' in bank.glyphs -> False
        '鱻' in bank.glyphs -> False
    То есть знак, который не является листом CHISE, нечем опознать.

    А палеографу нужно ровно это: он смотрит на ПОВРЕЖДЁННЫЙ или
    НЕИЗВЕСТНЫЙ знак и хочет сказать «это 龘» (или «это не 龘, а вот
    что»). Значит банк должен содержать и ЦЕЛЫЕ знаки.

    Их 98 727 (объединение CHISE Basic + Ext A..H + Compat).
    Каждый рендерится рекурсивным композитором `cangjie_composer`,
    который умеет собирать знаки без готового скелета — вплоть до
    172-чертных (龘 = ⿱龍⿰龍龍, 𰻞 = биан).

ЧТО ЗДЕСЬ РЕАЛЬНО ДЕЛАЕТСЯ:

    1. `build_fullchar_bank()` — рендерит каждый знак в стиле эры,
       считает дескрипторы (chamfer + HOG), сохраняет в .npz.
    2. `FullCharBank.match()` — матчит фото против полных знаков.
    3. Индекс остаётся ОПЦИОНАЛЬНЫМ: если файла нет, старый
       component-банк продолжает работать как раньше. Ничего не едет.

СОВМЕСТИМОСТЬ:
    - Новых обязательных зависимостей нет (numpy + PIL).
    - Старый `cangjie_paleo_matcher.py` НЕ ТРОНУТ.
    - Новый файл опционален: система работает и без него.
    - DESCRIPTOR_VERSION банка полных знаков = версия дескриптора
      `cangjie_paleo_matcher` (совпадает), поэтому кэш валиден.

ЗАПУСК:
    python3 cangjie_fullchar_bank.py --build --era KAISHU
    python3 cangjie_fullchar_bank.py --build --all
    python3 cangjie_fullchar_bank.py --stats
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [FULLCHAR-BANK]: %(message)s"
)
logger = logging.getLogger("CangjieFullCharBank")

HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# 1. СПИСОК ЗНАКОВ
# ============================================================================

def collect_full_char_universe(data_dir: str = "./data") -> List[str]:
    """
    Собирает ВСЕ знаки, для которых есть IDS-разбор в корпусе CHISE.

    Файлы: IDS-UCS-Basic.txt + IDS-UCS-Ext-{A,B-1..B-6,C,D,E,F,G,H}.txt
    + IDS-UCS-Compat.txt. Итого ~98 727 знаков.
    """
    candidates = [
        os.path.join(data_dir, "IDS-UCS-Basic.txt"),
    ]
    chise_dir = os.path.join(data_dir, "chise_all")
    for name in [
        "IDS-UCS-Ext-A.txt", "IDS-UCS-Ext-B-1.txt", "IDS-UCS-Ext-B-2.txt",
        "IDS-UCS-Ext-B-3.txt", "IDS-UCS-Ext-B-4.txt", "IDS-UCS-Ext-B-5.txt",
        "IDS-UCS-Ext-B-6.txt", "IDS-UCS-Ext-C.txt", "IDS-UCS-Ext-D.txt",
        "IDS-UCS-Ext-E.txt", "IDS-UCS-Ext-F.txt", "IDS-UCS-Ext-G.txt",
        "IDS-UCS-Ext-H.txt", "IDS-UCS-Compat.txt",
    ]:
        candidates.append(os.path.join(chise_dir, name))
        candidates.append(os.path.join(data_dir, name))

    seen: Dict[str, str] = {}
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#") or line.startswith(";"):
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    # Формат CHISE: [0]=код U+XXXX, [1]=САМ СИМВОЛ, [2]=IDS
                    ch = parts[1].strip()
                    if len(ch) != 1:
                        continue
                    if ch in seen:
                        continue
                    seen[ch] = path
        except Exception as e:
            logger.warning("Не удалось прочитать %s: %s", path, e)

    glyphs = sorted(seen.keys())
    logger.info("Всего знаков с IDS-разбором: %d", len(glyphs))
    return glyphs


def load_ids_map(data_dir: str = "./data") -> Dict[str, str]:
    """Знак -> его IDS-разбор (из всех файлов корпуса)."""
    out: Dict[str, str] = {}
    files = [os.path.join(data_dir, "IDS-UCS-Basic.txt")]
    chise_dir = os.path.join(data_dir, "chise_all")
    for name in [
        "IDS-UCS-Ext-A.txt", "IDS-UCS-Ext-B-1.txt", "IDS-UCS-Ext-B-2.txt",
        "IDS-UCS-Ext-B-3.txt", "IDS-UCS-Ext-B-4.txt", "IDS-UCS-Ext-B-5.txt",
        "IDS-UCS-Ext-B-6.txt", "IDS-UCS-Ext-C.txt", "IDS-UCS-Ext-D.txt",
        "IDS-UCS-Ext-E.txt", "IDS-UCS-Ext-F.txt", "IDS-UCS-Ext-G.txt",
        "IDS-UCS-Ext-H.txt", "IDS-UCS-Compat.txt",
    ]:
        files.append(os.path.join(chise_dir, name))
        files.append(os.path.join(data_dir, name))

    for path in files:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#") or line.startswith(";"):
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    # Формат CHISE: [0]=код U+XXXX, [1]=САМ СИМВОЛ, [2]=IDS
                    ch = parts[1].strip()
                    if len(ch) != 1 or ch in out:
                        continue
                    raw = parts[2].split()
                    if raw:
                        out[ch] = raw[0]
        except Exception:
            continue
    return out


# ============================================================================
# 2. РЕНДЕР ЗНАКА В СТИЛЕ ЭРЫ
# ============================================================================

def render_full_char_in_era(
    char: str,
    era: str,
    composer=None,
    data_dir: str = "./data",
    resolution: int = 64,
) -> Optional[np.ndarray]:
    """
    Рендерит ПОЛНЫЙ знак в стиле эры. Возвращает маску туши 0..1
    размера resolution x resolution или None.

    Для KAISHU используется каллиграфический рендерер (есть скелет
    MakeMeAHanzi) — он даёт настоящие пропорции кайшу.
    Для архаических эр (甲骨文/金文/篆书) берётся архаический силуэт
    из `cangjie_paleo_matcher._render_archaic_mask`, а если знака там
    нет — архаический рендер композитора.
    """
    from cangjie_paleo_matcher import (
        normalize_component_mask, CHAMFER_RES,
        ERA_KAISHU, ERA_JIAGUWEN, ERA_JINWEN, ERA_SEAL,
    )
    _ = (ERA_JIAGUWEN, ERA_JINWEN, ERA_SEAL)

    if era == ERA_KAISHU:
        # Канонический путь: настоящий каллиграфический рендер.
        # [v9.6] Рендерим в 256, потом сжимаем в 64 — в 64 штрихи слипаются.
        try:
            from cangjie_benchmark_forge import (
                CalligraphicPillowRenderer, RadicalLeaf, SpatialSlot, BoundingBox
            )
            from cangjie_benchmark_forge import GlyphRepository
            from cangjie_paleo_matcher import TEMPLATE_RENDER_RES
            repo = GlyphRepository()
            renderer = CalligraphicPillowRenderer(
                target_resolution=TEMPLATE_RENDER_RES, supersample_factor=1)
            leaf = RadicalLeaf(char, SpatialSlot.ROOT, repo)
            if leaf.medians:
                pad = 4
                leaf.calculate_geometry(BoundingBox(
                    pad, pad, TEMPLATE_RENDER_RES - pad, TEMPLATE_RENDER_RES - pad))
                leaf.render(renderer)
                arr = np.array(renderer.finalize().convert("L"), dtype=np.float32)
                ink = np.clip((255.0 - arr) / 255.0, 0.0, 1.0)
                return normalize_component_mask(ink)
        except Exception:
            pass
        # Нет скелета — собираем рекурсивно по IDS.
        if composer is None:
            from cangjie_composer import RecursiveComposer
            composer = RecursiveComposer(data_dir=data_dir)
        m = composer.render_char(char, resolution=TEMPLATE_RENDER_RES)
        return None if m is None else normalize_component_mask(m)

    # Архаические эры.
    try:
        from cangjie_paleo_matcher import PaleoTemplateBank
        bank = PaleoTemplateBank(era_mode=era, data_dir=data_dir)
        mask = bank._render_archaic_mask(char)
        if mask is not None:
            return mask
    except Exception:
        pass

    if composer is None:
        from cangjie_composer import RecursiveComposer
        composer = RecursiveComposer(data_dir=data_dir)
    m = composer.render_char(char, resolution=CHAMFER_RES)
    return None if m is None else normalize_component_mask(m)


# ============================================================================
# 3. СБОРКА БАНКА
# ============================================================================

def build_fullchar_bank(
    era: str,
    data_dir: str = "./data",
    out_path: Optional[str] = None,
    limit: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Строит банк ПОЛНЫХ знаков для одной эры.

    Возвращает словарь со статистикой. Кэш .npz — переиспользуемый.
    """
    from cangjie_paleo_matcher import compute_chamfer_and_hog, PaleoTemplateBank
    DESCRIPTOR_VERSION = PaleoTemplateBank.DESCRIPTOR_VERSION

    if out_path is None:
        out_path = os.path.join(
            data_dir, f"cangjie_fullchar_templates_{era.lower()}_v96.npz")

    if os.path.exists(out_path) and not force:
        logger.info("Банк уже существует: %s (использую кэш)", out_path)
        with np.load(out_path, allow_pickle=False) as z:
            return {
                "era": era, "path": out_path, "cached": True,
                "n_templates": int(len(z["glyphs_json"])),
            }

    from cangjie_composer import RecursiveComposer
    composer = RecursiveComposer(data_dir=data_dir)
    universe = collect_full_char_universe(data_dir)
    if limit:
        universe = universe[:limit]

    logger.info("Рендер %d полных знаков для эры %s ...", len(universe), era)
    t0 = time.perf_counter()

    chamfers: List[np.ndarray] = []
    hogs: List[np.ndarray] = []
    kept: List[str] = []
    failed = 0
    stride = max(1, len(universe) // 40)

    for i, ch in enumerate(universe):
        try:
            mask = render_full_char_in_era(
                ch, era, composer=composer, data_dir=data_dir)
        except Exception:
            mask = None
        if mask is None:
            failed += 1
            continue
        c_vec, h_vec = compute_chamfer_and_hog(mask)
        chamfers.append(c_vec)
        hogs.append(h_vec)
        kept.append(ch)
        if (i + 1) % stride == 0:
            el = time.perf_counter() - t0
            rate = (i + 1) / max(el, 1e-6)
            eta = (len(universe) - i - 1) / max(rate, 1e-6)
            logger.info("  %d/%d (%.1f%%) ok=%d fail=%d ETA %.0f с",
                        i + 1, len(universe), 100.0 * (i + 1) / len(universe),
                        len(kept), failed, eta)

    if not kept:
        logger.error("Ни один знак не отрендерился для эры %s", era)
        return {"era": era, "path": out_path, "cached": False, "n_templates": 0}

    cham = np.stack(chamfers, axis=0).astype(np.float32)
    hog = np.stack(hogs, axis=0).astype(np.float32)
    elapsed = time.perf_counter() - t0

    try:
        np.savez_compressed(
            out_path,
            glyphs_json=np.array(json.dumps(kept, ensure_ascii=False)),
            chamfer=cham,
            hog=hog,
            descriptor_version=np.array(DESCRIPTOR_VERSION),
            era=np.array(era),
        )
        logger.info("Банк %s сохранён: %d шаблонов за %.1f с -> %s",
                    era, len(kept), elapsed, out_path)
    except Exception as e:
        logger.error("Не удалось сохранить банк: %s", e)

    return {
        "era": era, "path": out_path, "cached": False,
        "n_templates": len(kept), "n_failed": failed,
        "elapsed_sec": round(elapsed, 1),
        "n_universe": len(universe),
    }


# ============================================================================
# 4. БАНК С МАТЧИНГОМ
# ============================================================================

class FullCharBank:
    """
    Банк полных знаков. Умеет:
      - match(ink_64) -> top-K кандидатов с уверенностью;
      - load/save в .npz;
      - работать как ДОПОЛНЕНИЕ к component-банку, не вместо него.
    """

    def __init__(self, era: str = "KAISHU", data_dir: str = "./data"):
        self.era = era
        self.data_dir = data_dir
        self.glyphs: List[str] = []
        self._chamfer: Optional[np.ndarray] = None
        self._hog: Optional[np.ndarray] = None

    # ---------------- io ----------------
    def path(self) -> str:
        return os.path.join(
            self.data_dir,
            f"cangjie_fullchar_templates_{self.era.lower()}_v96.npz")

    def is_ready(self) -> bool:
        return self._chamfer is not None and len(self.glyphs) > 0

    def load(self) -> bool:
        p = self.path()
        if not os.path.exists(p):
            logger.warning("Банк полных знаков не найден: %s", p)
            return False
        try:
            with np.load(p, allow_pickle=False) as z:
                self.glyphs = json.loads(str(z["glyphs_json"]))
                self._chamfer = z["chamfer"]
                self._hog = z["hog"]
            logger.info("⚡ [FULLCHAR-BANK HIT] %s: %d полных знаков",
                        self.era, len(self.glyphs))
            return True
        except Exception as e:
            logger.warning("Банк полных знаков повреждён (%s)", e)
            return False

    # ---------------- matching ----------------
    def match(self, ink_64: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """
        Матчит маску 64x64 против банка полных знаков.

        Использует ту же формулу, что и component-банк:
            score = 0.55 * cos(chamfer) + 0.45 * cos(hog)
        и калибровку из `cangjie_paleo_matcher`, если она доступна.
        """
        if not self.is_ready():
            return {"status": "UNAVAILABLE", "candidates": [],
                    "confidence": 0.0, "era": self.era}

        from cangjie_paleo_matcher import compute_chamfer_and_hog

        q_chamfer, q_hog = compute_chamfer_and_hog(ink_64)
        sim_c = self._chamfer @ q_chamfer.astype(np.float32)
        sim_h = self._hog @ q_hog.astype(np.float32)
        scores = 0.55 * sim_c + 0.45 * sim_h

        k = int(min(max(top_k, 8), len(scores)))
        order = np.argpartition(-scores, k - 1)[:k]
        order = order[np.argsort(-scores[order])]
        top = scores[order]

        z = (top - float(top[0])) / max(1e-6, 0.045)
        np.exp(z, out=z)
        posterior = z / z.sum()

        cands = []
        for rank, idx in enumerate(order):
            cands.append({
                "glyph": self.glyphs[int(idx)],
                "score": round(float(scores[int(idx)]), 4),
                "posterior": round(float(posterior[rank]), 4),
            })

        raw = float(top[0])
        margin = float(top[0] - top[1]) if len(top) > 1 else raw
        fit = float(np.clip((raw - 0.55) / 0.45, 0.0, 1.0))

        return {
            "status": "OK",
            "era": self.era,
            "candidates": cands,
            "best": cands[0]["glyph"] if cands else None,
            "raw_score": round(raw, 4),
            "margin": round(margin, 4),
            "fit": round(fit, 4),
            "posterior_top": round(float(posterior[0]), 4),
            "confidence": round(float(np.clip(posterior[0] * fit, 0.0, 1.0)), 4),
        }


# ============================================================================
# 5. CLI
# ============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Банк полных знаков Project Cangjie")
    ap.add_argument("--build", action="store_true", help="собрать банк")
    ap.add_argument("--all", action="store_true", help="собрать для всех эр")
    ap.add_argument("--era", default="KAISHU", help="эра (KAISHU/JIAGUWEN/JINWEN/XIAOZHUAN)")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--limit", type=int, default=None, help="ограничить число знаков")
    ap.add_argument("--force", action="store_true", help="пересобрать принудительно")
    ap.add_argument("--stats", action="store_true", help="показать статистику корпуса")
    args = ap.parse_args(argv)

    if args.stats:
        from cangjie_composer import RecursiveComposer
        composer = RecursiveComposer(data_dir=args.data_dir)
        universe = collect_full_char_universe(args.data_dir)
        composable = 0
        for ch in universe:
            try:
                if composer.can_compose(ch):
                    composable += 1
            except Exception:
                pass
        print("=" * 70)
        print("БАНК ПОЛНЫХ ЗНАКОВ — СТАТИСТИКА КОРПУСА")
        print("=" * 70)
        print(f"Знаков с IDS-разбором     : {len(universe)}")
        print(f"Собираются композитором   : {composable} "
              f"({composable / max(1, len(universe)) * 100:.1f}%)")
        print(f"Не собираются             : {len(universe) - composable}")
        return 0

    if args.build:
        eras = ["KAISHU", "JIAGUWEN", "JINWEN", "XIAOZHUAN"] if args.all else [args.era]
        results = []
        for era in eras:
            r = build_fullchar_bank(
                era, data_dir=args.data_dir, limit=args.limit, force=args.force)
            results.append(r)
            print(f"  {era}: {r.get('n_templates', 0)} шаблонов")
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
