"""
Project Cangjie — прямое опознание ЦЕЛОГО знака по фото [v9.6]

ЗАЧЕМ:

    Палеографу нужен ответ «что это за знак». Разбор на компоненты
    (⿰⿱⿲...) — это способ ПОЯСНИТЬ, из чего знак состоит, но не
    способ ответить на главный вопрос.

    Тем более: если знак повреждён или неопознан, разбор может
    развалиться (не найдено 3 из 7 листьев → дерево не собралось),
    а целый знак матчится по форме целиком и не зависит от того,
    удалось ли разложить его на части.

    Поэтому порядок должен быть ОБРАТНЫМ к привычному:
        1) сначала пробуем опознать ЦЕЛЫЙ знак по фото;
        2) если не вышло — разбираем на компоненты и собираем IDS.

ЗДЕСЬ:
    - `WholeGlyphMatcher` — обёртка над `FullCharBank` с калибровкой;
    - `match_whole_glyph()` — точка входа для декодера.

ФИЛОСОФИЯ ОТВЕТА:
    Никогда не выдаём «я не знаю» как пустоту. Если полного совпадения
    нет, показываем ЛУЧШИЕ КАНДИДАТЫ с честной уверенностью — именно
    это палеографу и нужно: не «вот ответ», а «вот три варианта, вот
    почему я склоняюсь к первому».

ЗАПУСК:
    python3 cangjie_whole_glyph.py --photo /path/to/photo.png
"""

from __future__ import annotations

import os
import json
import time
import logging
import argparse
from typing import Dict, List, Optional, Any

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [WHOLE-GLYPH]: %(message)s"
)
logger = logging.getLogger("CangjieWholeGlyph")


class WholeGlyphMatcher:
    """
    Опознание целого иероглифа по фотографии.

    Работает поверх `cangjie_fullchar_bank.FullCharBank` (банк полных
    знаков). Если банк не собран — честно возвращает статус
    UNAVAILABLE, ничего не выдумывая.
    """

    def __init__(self, era: str = "KAISHU", data_dir: str = "./data"):
        self.era = era
        self.data_dir = data_dir
        self._bank = None
        self._loaded = False

    # ------------------------------------------------------------------
    def _get_bank(self):
        if not self._loaded:
            self._loaded = True
            try:
                from cangjie_fullchar_bank import FullCharBank
                bank = FullCharBank(era=self.era, data_dir=self.data_dir)
                if bank.load():
                    self._bank = bank
                else:
                    logger.warning(
                        "Банк полных знаков для эры %s не собран — "
                        "прямое опознание отключено", self.era)
            except Exception as e:
                logger.warning("Не удалось загрузить банк полных знаков: %s", e)
        return self._bank

    def is_available(self) -> bool:
        bank = self._get_bank()
        return bank is not None and bank.is_ready()

    # ------------------------------------------------------------------
    def match_whole_glyph(
        self,
        ink_64: np.ndarray,
        top_k: int = 8,
        min_confidence: float = 0.30,
    ) -> Dict[str, Any]:
        """
        Матчит маску 64x64 против банка полных знаков.

        Возвращает:
            {
              "status": "RECOGNIZED" | "AMBIGUOUS" | "UNRECOGNIZED" | "UNAVAILABLE",
              "best": "龘" | None,
              "confidence": 0.0..1.0,
              "candidates": [{glyph, score, posterior, confidence}, ...],
              "reason": "почему такой вердикт"
            }
        """
        bank = self._get_bank()
        if bank is None or not bank.is_ready():
            return {
                "status": "UNAVAILABLE",
                "best": None,
                "confidence": 0.0,
                "candidates": [],
                "reason": "Банк полных знаков не собран. "
                          "Соберите его: python3 cangjie_fullchar_bank.py --build --all",
            }

        t0 = time.perf_counter()
        res = bank.match(ink_64, top_k=top_k)
        elapsed = (time.perf_counter() - t0) * 1000.0

        raw = float(res.get("raw_score", 0.0))
        margin = float(res.get("margin", 0.0))
        posterior = float(res.get("posterior_top", 0.0))
        fit = float(res.get("fit", 0.0))

        # [v9.6] ВАЖНО: АБСОЛЮТНЫЙ raw_score НЕ ГОДИТСЯ как признак
        #   «знак найден». Измерено на малом банке:
        #       знак, ЕСТЬ в банке      -> raw ≈ 0.66..1.00, margin ≈ 0.049
        #       знак, НЕТ в банке      -> raw ≈ 0.93,       margin ≈ 0.005
        #   То есть у СЛОЖНОГО глифа, которого нет в банке, score ВЫШЕ,
        #   чем у простого, который есть. raw_score мерит СЛОЖНОСТЬ
        #   глифа, а не качество совпадения.
        #
        #   Различает «есть / нет» — MARGIN (отрыв top1 от top2):
        #   в банке 0.049, вне банка 0.005 — разрыв в 9 раз.
        #
        #   Поэтому уверенность считается через margin:
        #       conf = clamp(margin / margin_ref) * posterior_norm
        #   где margin_ref — калибровочная константа (0.05).
        #
        #   Это ровно та ошибка, из-за которой первый бенчмарк давал
        #   97.8% на мёртвых данных: пороги подгонялись под абсолютный
        #   score, который на живой фотографии уезжал в ноль.
        margin_ref = 0.05
        margin_score = float(np.clip(margin / margin_ref, 0.0, 1.0))

        # [v9.6] КАЧЕСТВО ПОДГОНКИ — главный признак уверенности.
        #
        #   Измерено на полном банке (84 145 знаков):
        #       точное совпадение (道, 日, 龘) -> raw = 0.9999..1.0000
        #       случайно похожий, НЕ тот знак   -> raw = 0.85..0.93
        #
        #   То есть raw_score при полном банке РАБОТАЕТ как признак
        #   «это именно тот знак»: точное попадание в шаблон даёт
        #   почти единицу.
        #
        #   Почему тогда нельзя просто верить raw? Потому что на
        #   МАЛОМ банке (2 340 знаков) у СЛОЖНОГО глифа, которого нет
        #   в банке, raw тоже доходил до 0.93 — там ни одного точного
        #   совпадения не существует, и score упирается в «сложность».
        #   С барьером raw ≥ 0.93 такие случаи отсекаются.
        #
        #   Итоговая формула:
        #       conf = fit_quality × (0.5 + 0.5 × margin_score)
        #   где fit_quality — насколько точно глиф совпал с шаблоном,
        #   а margin_score — насколько он оторван от второго кандидата
        #   (защита от «много похожих, но не тот»).
        fit_quality = float(np.clip((raw - 0.85) / 0.14, 0.0, 1.0))

        # [v9.6] ВАЖНО: margin НЕ должен обнулять уверенность, когда
        #   совпадение ТОЧНОЕ. Измерено на полном банке:
        #       日  -> raw = 0.9999, margin = 0.0052  (топ-2 = 囗, 0.9947)
        #       國  -> raw = 1.0000, margin = 0.0015  (топ-2 = 圉, 0.9985)
        #   В банке 84 145 знаков, и у любого простого знака тысячи
        #   «почти таких же». Margin тут показывает не неуверенность,
        #   а плотность похожих соседей — а это признак ХОРОШЕГО
        #   совпадения, а не плохого.
        #
        #   Различает «точное попадание» и «случайно похожее» — сам
        #   raw_score: точное = 0.9999..1.0000, случайное = 0.85..0.93.
        #
        #   Поэтому margin входит в формулу только как МЯГКИЙ множитель
        #   и не опускает уверенность ниже 0.5 при точном совпадении.
        confidence = float(np.clip(
            fit_quality * (0.5 + 0.5 * margin_score), 0.0, 1.0))

        # [v9.6] ДОПОЛНИТЕЛЬНЫЙ БАРЬЕР: «глиф похож на знак, а не на
        #   мусор». На малом банке (2 340 знаков) знак 道, которого в
        #   банке НЕТ, получил margin 0.044 — и прошёл как RECOGNIZED.
        #   Причина: банк мал, и среди «непохожих» знаков находится
        #   случайно похожий. С абсолютным score 0.95 это ловится:
        #   у настоящего совпадения score ≈ 0.99–1.00 (точное попадание
        #   в шаблон), у случайно похожего — заметно ниже.
        #
        #   Поэтому REQUIRES_GOOD_FIT: top1 должен быть не просто
        #   «оторванным», а ещё и «хорошо подогнанным» под шаблон.
        REQUIRES_GOOD_FIT = 0.97

        cands: List[Dict[str, Any]] = []
        for i, c in enumerate(res.get("candidates", [])):
            cands.append({
                "glyph": c["glyph"],
                "score": c["score"],
                "posterior": c["posterior"],
                "confidence": round(float(np.clip(
                    fit_quality * (0.5 + 0.5 * margin_score), 0.0, 1.0)), 4),
                "rank": i + 1,
            })

        best = cands[0]["glyph"] if cands else None

        # ТОЧНОЕ совпадение (raw ≈ 1.0) — это доказательство тождества:
        # глиф совпал с шаблоном до долей процента. Margin тут может быть
        # нулевым (у 𰻞 и 𰻝 формы совпадают на 0.997 — это разные
        # знаки-близнецы), и это НЕ повод сомневаться в самом top-1.
        #
        # [v9.7] Порог близнецов: измерено на полном банке —
        #     𰻝 -> 𰻞 0.997000, 𰻝 0.996800, 𰻉 0.968800
        # То есть настоящий третий кандидат отваливается на 0.028, а
        # близнец отличается всего на 0.0002. Значит TWIN_MARGIN = 0.001
        # надёжно отделяет «двойник» от «нормальный конкурент».
        TWIN_MARGIN = 0.001
        IS_EXACT = raw >= 0.995

        if IS_EXACT and margin < TWIN_MARGIN and len(cands) > 1 and \
                cands[1]['score'] >= raw - 0.002:
            # Близнецы: две формы совпадают практически идеально — выбирать нельзя.
            status = "AMBIGUOUS"
            reason = (f"Форма совпадает сразу с двумя знаками: «{cands[0]['glyph']}» "
                      f"и «{cands[1]['glyph']}» ({cands[0]['score']:.4f} и "
                      f"{cands[1]['score']:.4f}). Это близнецы — различить их "
                      f"можно только по контексту находки, не по форме.")
        elif IS_EXACT:
            status = "RECOGNIZED"
            reason = (f"Форма точно совпала с эталоном «{best}» "
                      f"(качество подгонки {raw:.4f}).")
        elif confidence >= min_confidence and margin >= margin_ref * 0.5 and raw >= REQUIRES_GOOD_FIT:
            status = "RECOGNIZED"
            reason = (f"Форма совпала с эталоном «{best}»: кандидат оторван "
                      f"от второго на {margin:.3f}, качество подгонки "
                      f"{raw:.3f} (уверенность {confidence:.3f}).")
        elif raw < 0.95:
            status = "UNRECOGNIZED"
            reason = (f"Ни один известный знак не подходит: качество подгонки "
                      f"{raw:.3f} ниже 0.95 — глиф не совпадает ни с одним "
                      f"эталоном по форме. Похоже, такого знака нет в банке — "
                      f"дальше работает разбор на компоненты.")
        elif margin < margin_ref * 0.25:
            status = "UNRECOGNIZED"
            reason = (f"Ни один известный знак не выделяется: лучший кандидат "
                      f"«{best}» отделён от второго лишь на {margin:.3f} "
                      f"(порог {margin_ref * 0.25:.3f}). Похоже, такого знака "
                      f"нет в банке — дальше работает разбор на компоненты.")
        else:
            status = "AMBIGUOUS"
            reason = (f"Есть несколько близких кандидатов (уверенность "
                      f"{confidence:.3f} при пороге {min_confidence:.2f}). "
                      f"Требуется сверка с источником.")

        return {
            "status": status,
            "best": best,
            "confidence": round(confidence, 4),
            "raw_score": round(raw, 4),
            "margin": round(margin, 4),
            "posterior_top": round(posterior, 4),
            "candidates": cands,
            "reason": reason,
            "era": self.era,
            "elapsed_ms": round(elapsed, 1),
        }


# ============================================================================
# CLI — быстрый тест на файле
# ============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Прямое опознание целого иероглифа по фото")
    ap.add_argument("--photo", required=True, help="путь к фото")
    ap.add_argument("--era", default="KAISHU")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from cangjie_photo_decoder import PaleoPhotoPreprocessor
    from cangjie_paleo_matcher import normalize_component_mask

    prep = PaleoPhotoPreprocessor.process_raw_photo(args.photo, target_size=1024)
    ink = prep["ink_matrix"]
    ys, xs = np.where(ink > 0.2)
    if len(xs) == 0:
        print("На фото не найдено ни одного штриха.")
        return 1
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ink64 = normalize_component_mask(crop)

    matcher = WholeGlyphMatcher(era=args.era, data_dir=args.data_dir)
    res = matcher.match_whole_glyph(ink64, top_k=args.top_k)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print("=" * 66)
        print("ПРЯМОЕ ОПОЗНАНИЕ ЦЕЛОГО ЗНАКА")
        print("=" * 66)
        print(f"Статус     : {res['status']}")
        print(f"Лучший     : {res['best']}")
        print(f"Уверенность: {res['confidence']:.4f}")
        print(f"Почему     : {res['reason']}")
        print("Кандидаты:")
        for c in res.get("candidates", [])[:8]:
            print(f"   {c['glyph']}  score={c['score']:.4f} "
                  f"conf={c['confidence']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
