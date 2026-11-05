"""
Project Cangjie — Honest Confidence Calibration Harness [v9.4]

ЗАЧЕМ ЭТОТ МОДУЛЬ (и почему он важнее, чем кажется):

    Бенчмарк v9.3 показывал 99.56% top-1 — и это была ЛОЖЬ, удобная
    самому себе. Причина: тестовые кейсы синтезировались ТЕМ ЖЕ
    рендерером, что и банк шаблонов. Self-match по определению почти
    идеален (raw_score -> 1.0), поэтому пороги калибровки
    подгонялись под синтетику.

    На ЖИВОЙ фотографии (кость, бронза, 拓片, бумага) тот же матчер
    давал max raw_score = 0.16 -> fit = 0 -> confidence = 0.0 на ВСЁ.
    Система молчала на живых данных, показывая 99% на мёртвых.

    Этот модуль собирает РЕАЛЬНЫЙ калибровочный набор:
        - ПОЗИТИВЫ: глифы, отрендеренные в эпоху и ПОДВЕРГНУТЫЕ реальной
          доменной деградации (PDE-диффузия туши, TPS-деформация,
          каменный эстампаж 拓片, киноварная печать).
        - НЕГАТИВЫ: белый шум, пустой вход, «шум с инверсией»,
          и — главное — ПЕРЕКРЕСТНЫЕ пары (глиф А спутан с глифом Б),
          чтобы научить систему отличать «похож на что-то» от
          «похож на ПРАВИЛЬНЫЙ знак».

    Калибровка = логистическая регрессия (без sklearn, вручную на numpy)
    по признакам, которые матчер честно умеет вычислять. Все коэффициенты
    ПОЛУЧЕНЫ ИЗ ДАННЫХ, а не подобраны на глаз.

ЗАПУСК:
    python3 cangjie_calibration.py --out data/calibration_v94.json
    python3 cangjie_calibration.py --report-only data/calibration_v94.json
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image, ImageFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [CALIB]: %(message)s"
)
logger = logging.getLogger("CangjieCalibration")


# ============================================================================
# 1. ПРИЗНАКИ (features) — всё, что матчер умеет честно посчитать
# ============================================================================

FEATURE_NAMES: List[str] = [
    "z_top",        # насколько top1 выделяется НАД СРЕДНИМ по банку (в σ)
    "z_gap",        # насколько top1 оторвался от top2 (в σ)
    "abs_fit",      # абсолютное качество совпадения (для self-match = 1.0)
    "posterior_top",# доля массы у top1 среди top-K
]


def extract_features(
    scores: np.ndarray,
    top_idx: int,
    second_score: float,
    posterior_top: float,
    query_density: float,
    bank_median_density: float,
) -> np.ndarray:
    """Собирает вектор признаков из внутренностей матчера."""
    bank_mean = float(np.mean(scores))
    bank_std = float(np.std(scores))
    if bank_std > 1e-9:
        z_top = (float(scores[top_idx]) - bank_mean) / bank_std
        z_gap = (float(scores[top_idx]) - second_score) / bank_std
    else:
        z_top = 0.0
        z_gap = 0.0

    abs_fit = float(np.clip((float(scores[top_idx]) - 0.55) / 0.45, 0.0, 1.0))

    # ПРИМЕЧАНИЕ: признак «плотность» УДАЛЁН сознательно.
    # После density_normalize белый шум получает плотность ~1.19 — такую же,
    # как у жирного кайшу-глифа. Модель выучивала «плотность около 1.19 =
    # мусор» и понижала уверенность у НАСТОЯЩИХ жирных знаков
    # (кайшу top1=100%, но conf=0.665 < порога 0.8066 => AMBIGUOUS).
    # Плотность не различает шум и жирный знак — значит, не должна
    # участвовать в решении.

    return np.array([
        z_top,
        z_gap,
        abs_fit,
        posterior_top,
    ], dtype=np.float64)


# ============================================================================
# 2. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ СВОИМИ РУКАМИ (numpy, без sklearn)
# ============================================================================

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    epochs: int = 4000,
    lr: float = 0.35,
    class_weight: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Обучает логистическую регрессию (двоичная) градиентным спуском.
    class_weight = (w_pos, w_neg) — балансировка классов.
    """
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(d + 1, dtype=np.float64)
    b0 = 1.0
    y = y.astype(np.float64)

    if class_weight is None:
        w_pos = w_neg = 1.0
    else:
        w_pos, w_neg = class_weight

    for _ in range(epochs):
        z = Xb @ w
        p = _sigmoid(z)
        g = (p - y) * (w_pos * y + w_neg * (1.0 - y))
        grad = Xb.T @ g / n
        grad[1:] += (l2 / n) * w[1:]
        w -= lr * grad

    z = Xb @ w
    p = _sigmoid(z)
    return {
        "intercept": float(w[0]),
        "coef": [float(x) for x in w[1:]],
        "feature_names": FEATURE_NAMES,
        "bias_note": b0,
    }


def predict_proba(model: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    coef = np.asarray(model["coef"], dtype=np.float64)
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    return _sigmoid(Xb @ np.concatenate([[model["intercept"]], coef]))


# ============================================================================
# 3. СБОР ДАННЫХ
# ============================================================================

CALIB_GLYPHS: List[str] = [
    "日", "月", "水", "火", "木", "金", "土", "人", "口", "手",
    "心", "目", "耳", "虫", "鳥", "石", "言", "女", "子", "雨",
    "鬼", "申", "巫", "玉", "戈", "弓", "刀", "足", "車", "馬",
    "龍", "門", "風", "食", "首", "臼", "爭", "卩", "瓜", "甘",
    "白", "皿", "冊", "禾", "糸", "貝", "辛", "酉", "廾", "艸",
    "永", "王", "主", "生", "田", "甲", "由", "申", "冊", "止",
]


def _render_era_glyph(glyph: str, era: str, resolution: int = 192) -> Optional[np.ndarray]:
    """Рендерит глиф в стиле эпохи (как в банке шаблонов)."""
    try:
        from cangjie_benchmark_forge import (
            CalligraphicPillowRenderer, RadicalLeaf, BoundingBox, SpatialSlot,
            GlyphRepository,
        )
        if era == "KAISHU":
            # [v9.6] Рендер должен СОВПАДАТЬ с банком шаблонов.
            #   Банк теперь рендерит в TEMPLATE_RENDER_RES (256) и сжимает
            #   в 64 — при рендере сразу в 64 штрихи слипались, и кейс
            #   переставал быть похож на свой же шаблон.
            #   Следствие было: KAISHU top-1 падал до 0.0.
            from cangjie_paleo_matcher import TEMPLATE_RENDER_RES
            repo = GlyphRepository()
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
            return np.clip((255.0 - arr) / 255.0, 0.0, 1.0)
        else:
            from cangjie_archaic_epigraphy_engine import ArchaicJiaguwenJinwenSynthesizer
            synth = ArchaicJiaguwenJinwenSynthesizer(repo=GlyphRepository(), resolution=resolution)
            box = BoundingBox(12, 12, resolution - 12, resolution - 12)
            out = synth.render_archaic_inscription([(glyph, box)], era_mode=era, seed=7)
            return out["ink_density_matrix"]
    except Exception as e:
        logger.debug("Рендер %s/%s не удался: %s", glyph, era, e)
        return None


def _degrade_realistic(ink: np.ndarray, mode: str, seed: int = 11) -> np.ndarray:
    """Реальные доменные деградации (PDE, TPS, эстампаж, киноварь)."""
    rng = np.random.RandomState(seed)
    try:
        from cangjie_generative_degradation import (
            HistoricalArtifactDegradationPipeline, StoneRubbingTopianSimulator,
            XuanPaperInkDiffusionSolver, ThinPlateSplineDeformer,
            SubtractiveCinnabarSealOverlay,
        )
        h, w = ink.shape

        if mode == "TOPIAN_STONE_RUBBING":
            rgb = StoneRubbingTopianSimulator.synthesize_topian_rubbing(ink)
            gray = np.array(rgb.convert("L"), dtype=np.float32) / 255.0
            return np.clip(1.0 - gray, 0.0, 1.0)

        if mode in ("XUAN_INK_SEAL", "DAOIST_HUANGBIAOZHI"):
            solver = XuanPaperInkDiffusionSolver()
            diffused = solver.solve_diffusion(ink)
            paper = (252.0, 232.0, 182.0) if mode == "DAOIST_HUANGBIAOZHI" else (246.0, 241.0, 229.0)
            rgb_f = np.array(paper, dtype=np.float32)[None, None, :] * (1.0 - 0.90 * diffused[:, :, None])
            rgb = np.clip(rgb_f, 0.0, 255.0).astype(np.uint8)
            rgb = SubtractiveCinnabarSealOverlay.apply_cinnabar_seal(rgb)
            tps = ThinPlateSplineDeformer(grid_size=4, max_Clean_shift=0.032)
            bboxes = np.array([[0.12, 0.12, 0.88, 0.88]], dtype=np.float64)
            warped, _ = tps.warp_image_and_bboxes(rgb, bboxes)
            gray = np.array(Image.fromarray(warped).convert("L"), dtype=np.float32) / 255.0
            return np.clip(1.0 - gray, 0.0, 1.0)

        # «Костяные» и «бронзовые» артефакты
        h, w = ink.shape
        if mode == "BONE":
            cracks = (rng.rand(h, w) > 0.985).astype(np.float32)
            out = np.clip(ink * (1.0 - 0.55 * cracks) + 0.12 * cracks, 0.0, 1.0)
            out = np.clip(out + rng.randn(h, w).astype(np.float32) * 0.05, 0.0, 1.0)
            return out
        if mode == "BRONZE":
            patina = (rng.rand(h, w) > 0.90).astype(np.float32) * 0.25
            out = np.clip(ink * (1.0 - 0.35 * patina) + 0.10 * patina, 0.0, 1.0)
            out = np.clip(out + rng.randn(h, w).astype(np.float32) * 0.04, 0.0, 1.0)
            return out
        if mode == "PAPER":
            blurred = np.array(
                Image.fromarray((ink * 255).astype(np.uint8), mode="L")
                .filter(ImageFilter.GaussianBlur(radius=1.4)),
                dtype=np.float32) / 255.0
            out = np.clip(0.55 * ink + 0.45 * blurred, 0.0, 1.0)
            return np.clip(out * (0.85 + 0.25 * rng.rand(h, w)), 0.0, 1.0)
    except Exception as e:
        logger.debug("Деградация %s не удалась: %s", mode, e)

    return ink


def _collect_dataset(data_dir: str = "./data") -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Собирает калибровочный набор:
      POSITIVE (y=1): реальная деградация глифа X, правильный ответ = X
      NEGATIVE (y=0): шум, пусто, и перекрёстные пары (глиф A, ответ B)
    """
    from cangjie_paleo_matcher import (
        get_bank, ERA_KAISHU, ERA_JIAGUWEN, ERA_JINWEN,
        compute_chamfer_and_hog, normalize_component_mask,
    )

    feats: List[np.ndarray] = []
    labels: List[float] = []
    meta: List[Dict[str, Any]] = []

    bank_specs = [
        (ERA_KAISHU, "KAISHU"),
        (ERA_JIAGUWEN, "SHANG_JIAGUWEN"),
        (ERA_JINWEN, "ZHOU_JINWEN"),
    ]
    degradations = ["BONE", "BRONZE", "PAPER", "XUAN_INK_SEAL", "TOPIAN_STONE_RUBBING", "DAOIST_HUANGBIAOZHI"]

    # ---- Позитивы ----
    for era_const, era_name in bank_specs:
        bank = get_bank(era_const, data_dir=data_dir)
        if not bank.is_ready:
            logger.warning("Банк %s не готов — пропускаю", era_name)
            continue

        for gi, glyph in enumerate(CALIB_GLYPHS):
            base = _render_era_glyph(glyph, era_name)
            if base is None:
                continue
            for deg in degradations:
                ink = _degrade_realistic(base.copy(), deg, seed=13 + gi)
                f, ok = _measure(bank, ink, correct_glyph=glyph)
                if f is None:
                    continue
                feats.append(f)
                labels.append(1.0)
                meta.append({
                    "era": era_name, "degradation": deg, "glyph": glyph,
                    "kind": "positive", "top1_correct": bool(ok),
                })

    logger.info("Позитивов собрано: %d", len(labels))

    # ---- Негативы: мусор ----
    for era_const, era_name in bank_specs:
        bank = get_bank(era_const, data_dir=data_dir)
        if not bank.is_ready:
            continue
        rng = np.random.RandomState(5)
        # Мусор: разные типы, чтобы модель не подстроилась под один.
        for i in range(30):
            noise = (rng.rand(192, 192) > 0.5).astype(np.float32)
            f, _ = _measure(bank, noise, correct_glyph=None)
            if f is not None:
                feats.append(f); labels.append(0.0)
                meta.append({"era": era_name, "degradation": "NOISE",
                             "glyph": None, "kind": "negative_noise"})

        # Размытый мусор (похож на «грязное» фото)
        for i in range(15):
            from PIL import Image as _I
            base = (rng.rand(192, 192) > 0.5).astype(np.float32)
            blurred = np.array(
                _I.fromarray((base * 255).astype(np.uint8), mode="L")
                .filter(ImageFilter.GaussianBlur(radius=3.0)),
                dtype=np.float32) / 255.0
            f, _ = _measure(bank, blurred, correct_glyph=None)
            if f is not None:
                feats.append(f); labels.append(0.0)
                meta.append({"era": era_name, "degradation": "BLUR_NOISE",
                             "glyph": None, "kind": "negative_noise"})

        # Полосы/поля — частый мусор на фото стел и оттисков
        for i in range(15):
            stripes = np.zeros((192, 192), dtype=np.float32)
            stripes[::7 + (i % 3), :] = 1.0
            f, _ = _measure(bank, stripes, correct_glyph=None)
            if f is not None:
                feats.append(f); labels.append(0.0)
                meta.append({"era": era_name, "degradation": "STRIPES",
                             "glyph": None, "kind": "negative_noise"})

        # Случайные «кляксы» — имитация дефекта носителя
        for i in range(15):
            blob = np.zeros((192, 192), dtype=np.float32)
            for _ in range(6):
                cy = rng.randint(0, 192); cx = rng.randint(0, 192)
                r = rng.randint(6, 22)
                yy, xx = np.ogrid[:192, :192]
                blob[((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r] = 1.0
            f, _ = _measure(bank, blob, correct_glyph=None)
            if f is not None:
                feats.append(f); labels.append(0.0)
                meta.append({"era": era_name, "degradation": "BLOBS",
                             "glyph": None, "kind": "negative_noise"})

        # ---- ПЕРЕКРЁСТНЫЕ ПАРЫ: глиф A отрендерен, «ожидаем» B ----
        #
        # ВАЖНО (исправление логической ошибки v9.4):
        #     Матчер, назвавший глиф A, ПРАВ — он узнал то, что реально
        #     лежит на картинке. Раньше такие случаи помечались как
        #     НЕГАТИВЫ (y=0), и модель училась понижать уверенность за
        #     ПРАВИЛЬНЫЙ ответ. Итог: recognized_rate падал до 16-22%,
        #     а верные top-1 (4.31%) уходили в UNRECOGNIZED.
        #
        #     Правильная разметка: правда ли, что top1 == ОТРЕНДЕРЕННЫЙ
        #     глиф A? Если да — это позитив (уверенность высокая, система
        #     права). Если нет — негатив (система спутала).
        #     «Ожидаемый» глиф B в задачу не входит: модель не может
        #     знать, какой знак имел в виду палеограф, и учить её этому
        #     бессмысленно.
        for gi, glyph_a in enumerate(CALIB_GLYPHS[:20]):
            base = _render_era_glyph(glyph_a, era_name)
            if base is None:
                continue
            ink = _degrade_realistic(base.copy(), "BONE", seed=31 + gi)
            f, matched = _measure(bank, ink, correct_glyph=glyph_a)
            if f is None:
                continue
            is_pos = 1.0 if matched else 0.0
            feats.append(f); labels.append(is_pos)
            meta.append({"era": era_name, "degradation": "BONE",
                         "glyph": glyph_a, "kind": "cross_pair",
                         "top1_is_rendered": bool(matched)})

    X = np.vstack(feats) if feats else np.zeros((0, len(FEATURE_NAMES)))
    y = np.asarray(labels, dtype=np.float64)
    logger.info("Негативов добавлено, итого примеров: %d (pos=%d, neg=%d)",
                len(y), int(y.sum()), int((1 - y).sum()))
    return X, y, meta


def _measure(bank, ink: np.ndarray, correct_glyph: Optional[str]) -> Tuple[Optional[np.ndarray], bool]:
    """Прогоняет вход через банк и достаёт признаки + попал ли top1."""
    from cangjie_paleo_matcher import compute_chamfer_and_hog, normalize_component_mask
    try:
        i64 = normalize_component_mask(ink)
        q_c, q_h = compute_chamfer_and_hog(i64)
        sim_c = bank._chamfer @ q_c.astype(np.float32)
        sim_h = bank._hog @ q_h.astype(np.float32)
        scores = 0.55 * sim_c + 0.45 * sim_h
        k = int(min(8, len(scores)))
        order = np.argpartition(-scores, k - 1)[:k]
        order = order[np.argsort(-scores[order])]
        top_idx = int(order[0])
        second = float(scores[int(order[1])]) if len(order) > 1 else 0.0

        t = 0.045
        z = (scores[order] - float(scores[top_idx])) / max(1e-6, t)
        np.exp(z, out=z)
        posterior = z / z.sum()
        posterior_top = float(posterior[0])

        q_density = float(np.mean(i64))
        bank_density = 0.42  # каноническая плотность после density_normalize

        f = extract_features(scores, top_idx, second, posterior_top,
                             q_density, bank_density)
        top1 = bank.glyphs[top_idx]
        ok = (correct_glyph is not None and top1 == correct_glyph)
        return f, ok
    except Exception as e:
        logger.debug("Измерение не удалось: %s", e)
        return None, False


# ============================================================================
# 4. ОЦЕНКА МОДЕЛИ
# ============================================================================

def evaluate(model: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    p = predict_proba(model, X)
    yhat = (p >= 0.5).astype(np.float64)
    tp = float(np.sum((yhat == 1) & (y == 1)))
    fp = float(np.sum((yhat == 1) & (y == 0)))
    fn = float(np.sum((yhat == 0) & (y == 1)))
    tn = float(np.sum((yhat == 0) & (y == 0)))
    acc = float(np.mean(yhat == y)) if len(y) else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "n": int(len(y)),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "mean_prob_positive": round(float(p[y == 1].mean()) if (y == 1).any() else 0.0, 4),
        "mean_prob_negative": round(float(p[y == 0].mean()) if (y == 0).any() else 0.0, 4),
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Честная калибровка уверенности матчера")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="./data/calibration_v94.json")
    ap.add_argument("--report-only", default=None, help="Пересчитать метрики по готовой модели")
    args = ap.parse_args(argv)

    if args.report_only:
        with open(args.report_only, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        model = blob["model"]
        X = np.asarray(blob["_features"], dtype=np.float64)
        y = np.asarray(blob["_labels"], dtype=np.float64)
        print(json.dumps(evaluate(model, X, y), ensure_ascii=False, indent=2))
        return 0

    t0 = time.perf_counter()
    X, y, meta = _collect_dataset(args.data_dir)
    if len(y) < 50:
        logger.error("Недостаточно данных для калибровки: %d", len(y))
        return 1

    # Стандартизация признаков
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd

    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    model = fit_logistic(
        Xs, y, l2=1.0, epochs=4000, lr=0.35,
        class_weight=(len(y) / (2.0 * max(n_pos, 1)), len(y) / (2.0 * max(n_neg, 1))),
    )
    metrics = evaluate(model, Xs, y)

    # Порог выбираем НЕ по максимуму F1, а по целевой ТОЧНОСТИ на
    # негативах: для палеографа ложное принятие мусора за знак
    # опаснее, чем «не уверен, покажи альтернативы».
    # Правило: берём порог, при котором FPR <= 5% на негативах,
    # и среди всех таких порогов — максимальный recall.
    p = predict_proba(model, Xs)
    p_pos = p[y == 1]
    p_neg = p[y == 0]
    max_fpr = 0.05

    if len(p_neg):
        # порог = квантиль негативов (т.е. 95% негативов ниже порога)
        thr_fpr = float(np.quantile(p_neg, 1.0 - max_fpr))
    else:
        thr_fpr = 0.5

    # Порог = квантиль негативов: ровно FPR доля негативов окажется выше.
    # Дальше берём минимальный порог, сохраняющий recall на позитивах:
    # ниже thr_fpr опускаться нельзя (FPR вырастет), а thr_recall по
    # построению >= thr_fpr, поэтому итоговый порог = thr_fpr.
    best_thr = float(thr_fpr)
    # Дополнительно фиксируем, сколько позитивов мы при этом теряем.
    recall_at_thr = float(np.mean(p_pos >= best_thr)) if len(p_pos) else 0.0
    fpr_at_thr = float(np.mean(p_neg >= best_thr)) if len(p_neg) else 0.0
    metrics["recall_at_threshold"] = round(recall_at_thr, 4)
    metrics["fpr_at_threshold"] = round(fpr_at_thr, 4)

    # F1-порог для справки
    order = np.argsort(-p)
    ys = y[order]
    cum_tp = np.cumsum(ys)
    total_pos = max(n_pos, 1)
    f1_thr, best_f1 = 0.5, -1.0
    for i, thr in enumerate(p[order]):
        prec = cum_tp[i] / (i + 1)
        rec = cum_tp[i] / total_pos
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, f1_thr = f1, float(thr)

    metrics["chosen_threshold"] = round(best_thr, 4)
    metrics["f1_optimal_threshold"] = round(f1_thr, 4)
    metrics["f1_optimal"] = round(best_f1, 4)
    metrics["target_fpr"] = max_fpr
    metrics["threshold_policy"] = "FPR<=5% на негативах, затем максимальный recall"

    blob = {
        "model": model,
        "feature_mean": [float(x) for x in mu],
        "feature_std": [float(x) for x in sd],
        "metrics": metrics,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "n_examples": int(len(y)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "_features": X.tolist(),
        "_labels": y.tolist(),
        "_meta": meta,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, ensure_ascii=False, indent=2)

    print("=" * 78)
    print("ЧЕСТНАЯ КАЛИБРОВКА v9.4")
    print("=" * 78)
    print(f"Примеров: {len(y)}  (позит��вов {n_pos}, негативов {n_neg})")
    print(f"Метрики: {json.dumps(metrics, ensure_ascii=False)}")
    print("Коэффициенты:")
    for name, c in zip(FEATURE_NAMES, model["coef"]):
        print(f"   {name:16s} {c:+.4f}")
    print(f"   {'intercept':16s} {model['intercept']:+.4f}")
    print(f"\nМодель сохранена: {args.out}  ({time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
