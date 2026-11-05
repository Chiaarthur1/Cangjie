"""
Project Cangjie (Цанцзе) — Paleo Matcher Benchmark & Calibration Harness [v9.1]

ЗАЧЕМ:
    Модуль `cangjie_paleo_matcher.py` вводит ЧЕСТНУЮ вероятность распознавания
    вместо прежней фиктивной (`max(0.55, ...)`). Но «честная» должна значить
    «откалиброванная», а не «на глаз».

    Этот harness строит РЕАЛИСТИЧНЫЕ тест-кейсы:
        1) глиф рендерится в стиле эпохи (кайшу / 甲骨文 / 金文 / 篆);
        2) поверх накладывается деградация носителя (кость, бронза, 拓片, бумага);
        3) получившийся снимок прогоняется через матчер.

    По итогам печатаются:
        - top-1 accuracy по каждой эре;
        - распределение confidence для УСПЕШНЫХ и для ПРОВАЛЕННЫХ матчей;
        - рекомендуемые пороги (RECOGNIZED / AMBIGUOUS / UNRECOGNIZED),
          посчитанные из данных, а не выдуманные.

    Это же — готовый регрессионный тест: после любой правки матчера запусти
        python3 cangjie_matcher_benchmark.py --json report.json
    и сравнивай метрики.

ЗАПУСК:
    python3 cangjie_matcher_benchmark.py                 # сводка
    python3 cangjie_matcher_benchmark.py --full          # + вывод примеров ошибок
    python3 cangjie_matcher_benchmark.py --json out.json  # + машинный отчёт
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [BENCH]: %(message)s")
logger = logging.getLogger("CangjieMatcherBenchmark")

from cangjie_paleo_matcher import (  # noqa: E402
    get_bank, compute_chamfer_and_hog, ERA_KAISHU, ERA_JIAGUWEN, ERA_JINWEN, ERA_SEAL
)


# Тестовый набор: иероглифы, которые чаще всего встречаются в надписях
# ( Шан-Инь, Чжоу, ранние Хань) — именно их палеограф чаще всего видит.
DEFAULT_GLYPHS: List[str] = [
    "日", "月", "水", "火", "木", "金", "土", "人", "口", "手",
    "心", "目", "耳", "虫", "鳥", "石", "言", "女", "子", "雨",
    "鬼", "申", "巫", "玉", "戈", "弓", "刀", "足", "車", "馬",
    "龍", "門", "風", "食", "首", "臼", "爭", "卩", "疒", "瓜",
    "甘", "白", "皿", "冊", "禾", "糸", "貝", "辛", "酉", "廾",
    "艸", "巛", "廴", "爿", "彡", "匕", "乃", "其", "皿", "永",
]


def _synthesize_artifact(
    glyph: str,
    era: str,
    degradation: str,
    repo=None,
    resolution: int = 192,
    seed: int = 7,
) -> Optional[np.ndarray]:
    """
    Строит реалистичный «снимок» глифа: стилизованный под эпоху и испорченный
    деградацией носителя. Возвращает бинарную маску туши (float 0..1).
    """
    from cangjie_benchmark_forge import (
        CalligraphicPillowRenderer, RadicalLeaf, BoundingBox, SpatialSlot
    )
    from cangjie_archaic_epigraphy_engine import ArchaicJiaguwenJinwenSynthesizer
    from cangjie_generative_degradation import HistoricalArtifactDegradationPipeline

    repo = repo or GlyphRepository()
    res = resolution
    rng = np.random.RandomState(seed)

    # 1) Стилизация под эпоху
    if era == ERA_KAISHU:
        # ВАЖНО: настройки рендера ДОЛЖНЫ совпадать с банком шаблонов
        # (cangjie_paleo_matcher: target 64, supersample 1, box 4..56).
        # Иначе тест измеряет не устойчивость к артефактам, а расхождение
        # двух разных рендереров: сначала получалось 0% по кайшу при 94% по
        # 甲骨文. Расхождение вносится ДЕГРАДАЦИЕЙ ниже — это и есть
        # моделируемая реальность.
        #
        # [v9.6] Обновлено: банк шаблонов теперь рендерит в
        #   TEMPLATE_RENDER_RES (256) и сжимает в 64 (при рендере сразу
        #   в 64 штрихи слипались). Кейс ДОЛЖЕН готовиться так же,
        #   иначе тест сравнивает непохожие картинки и даёт 0% там,
        #   где система работает.
        from cangjie_paleo_matcher import TEMPLATE_RENDER_RES
        renderer = CalligraphicPillowRenderer(
            target_resolution=TEMPLATE_RENDER_RES, supersample_factor=1)
        leaf = RadicalLeaf(glyph, SpatialSlot.ROOT, repo)
        if not leaf.medians:
            return None
        pad = 4
        leaf.calculate_geometry(BoundingBox(
            pad, pad, TEMPLATE_RENDER_RES - pad, TEMPLATE_RENDER_RES - pad))
        leaf.render(renderer)
        arr = np.array(renderer.finalize().convert("L"), dtype=np.float32)
        ink = np.clip((255.0 - arr) / 255.0, 0.0, 1.0)
    else:
        synth = ArchaicJiaguwenJinwenSynthesizer(repo=repo, resolution=res)
        box = BoundingBox(12, 12, res - 12, res - 12)
        out = synth.render_archaic_inscription([(glyph, box)], era_mode=era, seed=seed)
        ink = out["ink_density_matrix"]

    if ink is None or float(ink.max()) <= 0.01:
        return None

    # 2) Деградация носителя (размеры берём из самой маски)
    h, w = ink.shape[:2]
    if degradation == "BONE":          # 甲骨文: трещины, сколы, выветривание
        cracks = (rng.rand(h, w) > 0.985).astype(np.float32)
        ink = np.clip(ink * (1.0 - 0.55 * cracks) + 0.12 * cracks, 0.0, 1.0)
        ink = np.clip(ink + rng.randn(h, w).astype(np.float32) * 0.05, 0.0, 1.0)
    elif degradation == "BRONZE":      # 金文: патина, зеленоватый налёт, коррозия
        patina = (rng.rand(h, w) > 0.90).astype(np.float32) * 0.25
        ink = np.clip(ink * (1.0 - 0.35 * patina) + 0.10 * patina, 0.0, 1.0)
        ink = np.clip(ink + rng.randn(h, w).astype(np.float32) * 0.04, 0.0, 1.0)
    elif degradation == "RUBBING":     # 拓片: белые штрихи по углю, сколы камня
        ink = np.clip(ink * (0.75 + 0.35 * rng.rand(h, w)), 0.0, 1.0)
        chip = (rng.rand(h, w) > 0.99).astype(np.float32)
        ink = np.clip(ink * (1.0 - 0.6 * chip), 0.0, 1.0)
    elif degradation == "PAPER":       # 宣纸: размытие, разрывы волокна, бледность
        from PIL import ImageFilter
        blurred = np.array(
            Image.fromarray((ink * 255).astype(np.uint8), mode="L")
            .filter(ImageFilter.GaussianBlur(radius=1.4)),
            dtype=np.float32) / 255.0
        ink = np.clip(0.55 * ink + 0.45 * blurred, 0.0, 1.0)
        ink = np.clip(ink * (0.85 + 0.25 * rng.rand(h, w)), 0.0, 1.0)
    elif degradation == "CLEAN":
        pass
    else:
        raise ValueError(f"Неизвестная деградация: {degradation}")

    # 3) Приведение к каноническому виду 64x64 (как в фото-декодере).
    #    ВАЖНО: ink — это уже ПЛОТНОСТЬ ТУШИ (1 = штрих, 0 = фон), поэтому
    #    инвертировать его второй раз нельзя. Раньше здесь стоял лишний
    #    переворот, из-за которого фон становился «тушью» и ВСЕ 900 кейсов
    #    бенчмарка падали.
    from cangjie_paleo_matcher import normalize_component_mask
    return normalize_component_mask(ink)


def run_benchmark(
    data_dir: str = "./data",
    glyphs: Optional[List[str]] = None,
    eras: Optional[List[str]] = None,
    degradations: Optional[List[str]] = None,
    top_k: int = 5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Прогоняет бенчмарк и возвращает метрики + рекомендуемые пороги."""
    glyphs = glyphs or DEFAULT_GLYPHS
    eras = eras or [ERA_KAISHU, ERA_JIAGUWEN, ERA_JINWEN]
    degradations = degradations or ["CLEAN", "BONE", "BRONZE", "RUBBING", "PAPER"]

    from cangjie_benchmark_forge import GlyphRepository
    repo = GlyphRepository()

    banks = {era: get_bank(era, data_dir=data_dir, repo=repo) for era in eras}
    for era, b in banks.items():
        if not b.is_ready:
            logger.warning("Банк %s пуст — пропускаю", era)

    per_case: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    for era in eras:
        bank = banks[era]
        if not bank.is_ready:
            continue
        for deg in degradations:
            for glyph in glyphs:
                mask = _synthesize_artifact(glyph, era, deg, repo=repo)
                if mask is None:
                    continue
                res = bank.match(mask, top_k=top_k)
                got = res.best.glyph if res.best else None
                per_case.append({
                    "era": era,
                    "degradation": deg,
                    "glyph": glyph,
                    "predicted": got,
                    "correct": bool(got == glyph),
                    "confidence": round(float(res.confidence), 4),
                    "raw_score": round(float(res.raw_score), 4),
                    "margin": round(float(res.margin), 4),
                    "status": res.status,
                    "top3": [c.glyph for c in res.candidates[:3]],
                })

    elapsed = time.perf_counter() - t0
    total = len(per_case)
    correct = sum(1 for c in per_case if c["correct"])

    conf_ok = np.array([c["confidence"] for c in per_case if c["correct"]], dtype=np.float64)
    conf_bad = np.array([c["confidence"] for c in per_case if not c["correct"]], dtype=np.float64)

    # Рекомендуемые пороги из данных:
    #   RECOGNIZED  — confidence выше 10-го перцентиля успешных и заметно выше
    #                 максимума провальных (т.е. ошибки «поймать» можно);
    #   UNRECOGNIZED — confidence ниже 90-го перцентиля провальных.
    def _p(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q)) if arr.size else 0.0

    rec_thr = 0.0
    if conf_ok.size and conf_bad.size:
        rec_thr = float(max(0.30, min(_p(conf_ok, 25), _p(conf_bad, 95) + 0.10)))
    elif conf_ok.size:
        rec_thr = float(max(0.30, _p(conf_ok, 25)))

    unrec_thr = 0.0
    if conf_bad.size:
        unrec_thr = float(min(0.45, _p(conf_bad, 90)))
    elif conf_ok.size:
        unrec_thr = float(min(0.45, _p(conf_ok, 10)))

    by_era: Dict[str, Any] = {}
    for era in eras:
        sub = [c for c in per_case if c["era"] == era]
        if not sub:
            continue
        by_era[era] = {
            "n": len(sub),
            "top1_accuracy": round(sum(1 for c in sub if c["correct"]) / len(sub), 4),
            "mean_confidence_correct": round(
                float(np.mean([c["confidence"] for c in sub if c["correct"]])) if any(c["correct"] for c in sub) else 0.0, 4),
        }

    by_deg: Dict[str, Any] = {}
    for deg in degradations:
        sub = [c for c in per_case if c["degradation"] == deg]
        if not sub:
            continue
        by_deg[deg] = {
            "n": len(sub),
            "top1_accuracy": round(sum(1 for c in sub if c["correct"]) / len(sub), 4),
        }

    report: Dict[str, Any] = {
        "total_cases": total,
        "top1_accuracy": round(correct / total, 4) if total else 0.0,
        "elapsed_sec": round(elapsed, 2),
        "ms_per_case": round(elapsed / total * 1000, 2) if total else 0.0,
        "by_era": by_era,
        "by_degradation": by_deg,
        "confidence_correct": {
            "n": int(conf_ok.size),
            "mean": round(float(conf_ok.mean()), 4) if conf_ok.size else 0.0,
            "p10": round(_p(conf_ok, 10), 4),
            "p50": round(_p(conf_ok, 50), 4),
            "p90": round(_p(conf_ok, 90), 4),
        },
        "confidence_wrong": {
            "n": int(conf_bad.size),
            "mean": round(float(conf_bad.mean()), 4) if conf_bad.size else 0.0,
            "p90": round(_p(conf_bad, 90), 4),
            "max": round(float(conf_bad.max()), 4) if conf_bad.size else 0.0,
        },
        "recommended_thresholds": {
            "recognized_confidence": round(rec_thr, 4),
            "unrecognized_confidence": round(unrec_thr, 4),
        },
        "cases": per_case,
    }

    if verbose:
        misses = [c for c in per_case if not c["correct"]][:40]
        report["misses"] = misses

    return report


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Бенчмарк палеографического матчера")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--full", action="store_true", help="показать примеры ошибок")
    ap.add_argument("--json", default=None, help="выгрузить отчёт в JSON")
    args = ap.parse_args()

    print("=" * 78)
    print("БЕНЧМАРК ПАЛЕОГРАФИЧЕСКОГО МАТЧЕРА — Project Cangjie v9.1")
    print("=" * 78)

    rep = run_benchmark(data_dir=args.data_dir, verbose=args.full)

    print(f"\nВсего кейсов: {rep['total_cases']}   "
          f"Top-1 accuracy: {rep['top1_accuracy']*100:.1f}%   "
          f"({rep['ms_per_case']} мс/кейс)")

    print("\nПо эрам:")
    for era, m in rep["by_era"].items():
        print(f"  {era:<18} n={m['n']:<5} acc={m['top1_accuracy']*100:5.1f}%   "
              f"conf(верн)={m['mean_confidence_correct']:.3f}")

    print("\nПо типу деградации:")
    for deg, m in rep["by_degradation"].items():
        print(f"  {deg:<18} n={m['n']:<5} acc={m['top1_accuracy']*100:5.1f}%")

    cc, cw = rep["confidence_correct"], rep["confidence_wrong"]
    print(f"\nConfidence (верные):  n={cc['n']}  mean={cc['mean']:.3f}  "
          f"p10={cc['p10']:.3f}  p50={cc['p50']:.3f}  p90={cc['p90']:.3f}")
    print(f"Confidence (ошибки):  n={cw['n']}  mean={cw['mean']:.3f}  "
          f"p90={cw['p90']:.3f}  max={cw['max']:.3f}")
    rt = rep["recommended_thresholds"]
    print(f"\nРекомендуемые пороги:  RECOGNIZED >= {rt['recognized_confidence']}   "
          f"UNRECOGNIZED < {rt['unrecognized_confidence']}")

    if args.full and rep.get("misses"):
        print("\nПримеры ошибок (первые 20):")
        for m in rep["misses"][:20]:
            print(f"  {m['era']:<16} {m['degradation']:<9} {m['glyph']} -> {m['predicted']}  "
                  f"conf={m['confidence']:.3f} top3={m['top3']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"\nОтчёт сохранён: {args.json}")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
