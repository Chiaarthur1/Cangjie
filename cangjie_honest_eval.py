"""
Project Cangjie — Honest Evaluation on REAL Degradation [v9.4]

ЗАЧЕМ:
    `cangjie_matcher_benchmark.py` синтезирует кейсы ТЕМ ЖЕ рендерером,
    которым построен банк шаблонов. Это self-match по определению: сырой
    скор упирается в 1.0, и пороги калибровки подгоняются под эту шкалу.
    Результат: 99% на синтетике и 0.0 confidence на живых фотографиях.

    Этот harness собирает кейсы ИНАЧЕ:
        1) глиф рендерится в стиле эпохи;
        2) на него НАКЛАДЫВАЕТСЯ РЕАЛЬНАЯ доменная деградация из
           `cangjie_generative_degradation`:
             - XuanPaperInkDiffusionSolver  — PDE-диффузия туши по бумаге;
             - ThinPlateSplineDeformer       — нелинейная деформация носителя;
             - StoneRubbingTopianSimulator  — каменный эстампаж 拓片;
             - SubtractiveCinnabarSealOverlay — киноварная печать;
             - а также трещины/патина/бледность для кости и бронзы;
        3) получившийся СНИМОК проходит через матчер.

    Отличие от бенчмарка v9.3 принципиальное: рендер кейса и рендер
    шаблона банка — РАЗНЫЕ пайплайны (для кайшу — «каллиграфический
    пером» из банка vs «PDE-диффузия + TPS» здесь), поэтому совпадение
    приходится заслуживать, а не получать автоматически.

    Дополнительно меряется ОТДЕЛЬНО, сколько «шумных» входов система
    принимает за знаки (false accept rate) — это то, что палеограф
    заметит первым.

ЗАПУСК:
    python3 cangjie_honest_eval.py --out ../artifacts/honest_eval_v94.json
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [HONEST-EVAL]: %(message)s"
)
logger = logging.getLogger("CangjieHonestEval")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cangjie_calibration import (  # noqa: E402
    CALIB_GLYPHS, _render_era_glyph, _degrade_realistic,
)

EVAL_GLYPHS: List[str] = [
    "日", "月", "水", "火", "木", "金", "土", "人", "口", "手",
    "心", "目", "耳", "虫", "鳥", "石", "言", "女", "子", "雨",
    "鬼", "申", "巫", "玉", "戈", "弓", "刀", "足", "車", "馬",
    "龍", "門", "風", "食", "首", "臼", "爭", "卩", "瓜", "甘",
    "白", "皿", "禾", "糸", "貝", "辛", "酉", "永", "王", "主",
]

REAL_DEGRADATIONS = [
    "XUAN_INK_SEAL",        # 宣纸 + тушь + киноварь + TPS  — самый честный
    "TOPIAN_STONE_RUBBING", # 拓片 — каменный эстампаж
    "DAOIST_HUANGBIAOZHI",  # даосский жёлтый бланк
    "BONE",                 # кость: трещины, сколы, выветривание
    "BRONZE",               # бронза: патина, коррозия
    "PAPER",                # бумага: размытие, бледность
]


def _match_top1(bank, ink: np.ndarray) -> Tuple[Optional[str], float, str, float]:
    from cangjie_paleo_matcher import normalize_component_mask
    try:
        i64 = normalize_component_mask(ink)
        res = bank.match(i64, top_k=5)
        if res.best is None:
            return None, 0.0, "UNRECOGNIZED", 0.0
        return res.best.glyph, float(res.confidence), res.status, float(res.raw_score)
    except Exception as e:
        logger.debug("Матч не удался: %s", e)
        return None, 0.0, "ERROR", 0.0


def run_eval(
    data_dir: str = "./data",
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    from cangjie_paleo_matcher import get_bank, ERA_KAISHU, ERA_JIAGUWEN, ERA_JINWEN

    bank_specs = [
        (ERA_KAISHU, "KAISHU"),
        (ERA_JIAGUWEN, "SHANG_JIAGUWEN"),
        (ERA_JINWEN, "ZHOU_JINWEN"),
    ]

    t0 = time.perf_counter()
    cases: List[Dict[str, Any]] = []
    neg_cases: List[Dict[str, Any]] = []

    for era_const, era_name in bank_specs:
        bank = get_bank(era_const, data_dir=data_dir)
        if not bank.is_ready:
            logger.warning("Банк %s не готов", era_name)
            continue
        logger.info("=== Эра %s: банк %d шаблонов, калибровка=%s ===",
                    era_name, len(bank.glyphs),
                    "да" if bank._calib_model is not None else "НЕТ (v9.3 fallback)")

        for gi, glyph in enumerate(EVAL_GLYPHS):
            base = _render_era_glyph(glyph, era_name)
            if base is None:
                continue
            for deg in REAL_DEGRADATIONS:
                ink = _degrade_realistic(base.copy(), deg, seed=101 + gi)
                top1, conf, status, raw = _match_top1(bank, ink)
                cases.append({
                    "era": era_name,
                    "degradation": deg,
                    "truth": glyph,
                    "top1": top1,
                    "correct": bool(top1 == glyph),
                    "confidence": round(conf, 4),
                    "status": status,
                    "raw_score": round(raw, 4),
                })

        # --- НЕГАТИВЫ: сколько мусора система принимает за знаки ---
        rng = np.random.RandomState(99)
        for i in range(40):
            noise = (rng.rand(192, 192) > 0.5).astype(np.float32)
            top1, conf, status, raw = _match_top1(bank, noise)
            neg_cases.append({
                "era": era_name, "kind": "white_noise",
                "top1": top1, "confidence": round(conf, 4),
                "status": status, "raw_score": round(raw, 4),
                "false_accept": status == "RECOGNIZED",
            })

        # --- НЕГАТИВЫ: перекрёстные пары ---
        #
        # ВАЖНО (исправление ошибки, аналогичной в калибровке):
        #     Отрендерен глиф A. Матчер, назвавший A, ПРАВ — он узнал то,
        #     что реально лежит на картинке. Прежний код считал это
        #     «ложным принятием», потому что «ожидаемым» был назначен
        #     другой глиф B. В итоге 20 глифов x 3 эры = 60 cases
        #     засчитались как false_accept, хотя система работала верно.
        #
        #     Ложным принятием здесь считается только ситуация, когда
        #     система уверенно назвала НЕ ТОТ знак, что на картинке.
        for gi, glyph_a in enumerate(EVAL_GLYPHS[:20]):
            base = _render_era_glyph(glyph_a, era_name)
            if base is None:
                continue
            ink = _degrade_realistic(base.copy(), "XUAN_INK_SEAL", seed=77 + gi)
            top1, conf, status, raw = _match_top1(bank, ink)
            neg_cases.append({
                "era": era_name, "kind": "cross_pair",
                "rendered": glyph_a,
                "top1": top1, "confidence": round(conf, 4),
                "status": status, "raw_score": round(raw, 4),
                # ложь = уверенно назвали чужой знак
                "false_accept": status == "RECOGNIZED" and top1 != glyph_a,
                "correctly_identified": top1 == glyph_a,
            })

    # ---------------- СВОДКА ----------------
    n = len(cases)
    correct = sum(1 for c in cases if c["correct"])
    by_era: Dict[str, Any] = {}
    for era_name in {c["era"] for c in cases}:
        sub = [c for c in cases if c["era"] == era_name]
        by_era[era_name] = {
            "n": len(sub),
            "top1_accuracy": round(sum(1 for c in sub if c["correct"]) / max(len(sub), 1), 4),
            "recognized_rate": round(sum(1 for c in sub if c["status"] == "RECOGNIZED") / max(len(sub), 1), 4),
            "mean_confidence": round(float(np.mean([c["confidence"] for c in sub])), 4) if sub else 0.0,
        }

    by_deg: Dict[str, Any] = {}
    for deg in REAL_DEGRADATIONS:
        sub = [c for c in cases if c["degradation"] == deg]
        if not sub:
            continue
        by_deg[deg] = {
            "n": len(sub),
            "top1_accuracy": round(sum(1 for c in sub if c["correct"]) / len(sub), 4),
            "mean_confidence": round(float(np.mean([c["confidence"] for c in sub])), 4),
        }

    # КЛЮЧЕВАЯ МЕТРИКА: среди ВЕРНЫХ top-1 сколько помечено как «не опознано»
    correct_cases = [c for c in cases if c["correct"]]
    correct_but_unrecognized = sum(
        1 for c in correct_cases if c["status"] == "UNRECOGNIZED"
    )

    false_accepts = sum(1 for c in neg_cases if c["false_accept"])
    # Раздельно: мусор и перекрёстные пары — разная природа ошибки
    noise_cases = [c for c in neg_cases if c["kind"] == "white_noise"]
    cross_cases = [c for c in neg_cases if c["kind"] == "cross_pair"]
    noise_fa = sum(1 for c in noise_cases if c["false_accept"])
    cross_fa = sum(1 for c in cross_cases if c["false_accept"])
    cross_ok = sum(1 for c in cross_cases if c.get("correctly_identified"))
    summary = {
        "n_cases": n,
        "top1_accuracy": round(correct / max(n, 1), 4),
        "by_era": by_era,
        "by_degradation": by_deg,
        "correct_but_unrecognized": correct_but_unrecognized,
        "correct_but_unrecognized_pct": round(
            correct_but_unrecognized / max(len(correct_cases), 1) * 100, 2),
        "n_negative_cases": len(neg_cases),
        "false_accepts": false_accepts,
        "false_accept_rate": round(false_accepts / max(len(neg_cases), 1), 4),
        "false_accepts_noise": noise_fa,
        "false_accept_rate_noise": round(noise_fa / max(len(noise_cases), 1), 4),
        "false_accepts_cross": cross_fa,
        "false_accept_rate_cross": round(cross_fa / max(len(cross_cases), 1), 4),
        "cross_pair_correctly_identified": cross_ok,
        "cross_pair_identification_rate": round(cross_ok / max(len(cross_cases), 1), 4),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }

    result = {
        "summary": summary,
        "cases": cases,
        "negatives": neg_cases,
    }

    print("=" * 78)
    print("ЧЕСТНАЯ ОЦЕНКА НА РЕАЛЬНЫХ ДЕГРАДАЦИЯХ (v9.4)")
    print("=" * 78)
    print(f"Кейсов: {n}   top-1 accuracy: {summary['top1_accuracy']}")
    print(f"Верных, но помеченных UNRECOGNIZED: {correct_but_unrecognized} "
          f"({summary['correct_but_unrecognized_pct']}%)")
    print(f"Ложных принятий ВСЕГО: {false_accepts}/{len(neg_cases)} "
          f"= {summary['false_accept_rate']}")
    print(f"   из них мусор:            {noise_fa}/{len(noise_cases)} "
          f"= {summary['false_accept_rate_noise']}")
    print(f"   из них перекрёстные:     {cross_fa}/{len(cross_cases)} "
          f"= {summary['false_accept_rate_cross']}")
    print(f"   перекрёстные опознаны верно: {cross_ok}/{len(cross_cases)} "
          f"= {summary['cross_pair_identification_rate']}")
    print("\nПо эрам:")
    for k, v in by_era.items():
        print(f"   {k:16s} top1={v['top1_accuracy']:.4f}  recognized={v['recognized_rate']:.4f}  "
              f"mean_conf={v['mean_confidence']:.4f}")
    print("\nПо деградациям:")
    for k, v in by_deg.items():
        print(f"   {k:22s} top1={v['top1_accuracy']:.4f}  mean_conf={v['mean_confidence']:.4f}")

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"\nОтчёт сохранён: {out_path}")

    return result


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Честная оценка на реальных деградациях")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    run_eval(args.data_dir, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
