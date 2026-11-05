"""
Project Cangjie — проверка прямого опознания полных знаков [v9.6]

Гоняет матч по фотографии-«эталону» для набора знаков разной сложности
(от 1-чертного 日 до 58-чертного 𰻞) и печатает честную таблицу:

    знак | черт | top-1 | top-1 == ожидаемый | confidence | статус

И отдельно — негативы (шум, пусто, клякса): сколько из них ЛОЖНО
принято за знак (это и есть false accept rate).

ЗАПУСК:
    python3 cangjie_verify_fullchar.py
    python3 cangjie_verify_fullchar.py --era SHANG_JIAGUWEN
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Optional, Any

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] [VERIFY]: %(message)s"
)
logger = logging.getLogger("CangjieVerifyFullChar")


# Знаки разной сложности: (символ, ожидаемое_имя_для_лога, комментарий)
TEST_CHARS: List[str] = [
    "日", "月", "水", "火", "木", "金", "土", "人", "口", "手",
    "心", "目", "耳", "虫", "鳥", "石", "言", "女", "子", "雨",
    "鬼", "申", "巫", "玉", "戈", "弓", "刀", "足", "車", "馬",
    "龍", "門", "風", "食", "首", "臼", "爭", "卩", "疒", "瓜",
    "龘", "鱻", "麤", "齉", "𪚥", "𰻞", "𰻝", "𨭎", "𬢕", "𡧃",
]

# Знаки с операторами окружения (⿺ 辶, ⿴ 囗, ⿸ 阝) — там была поломка геометрии
ENCLOSURE_CHARS: List[str] = ["道", "這", "進", "遠", "過", "國", "廟", "建", "延"]


def synth_photo(ink: np.ndarray) -> np.ndarray:
    """Готовит маску 64x64 из маски композитора (как это делает детектор)."""
    from cangjie_paleo_matcher import normalize_component_mask
    return normalize_component_mask(ink)


def make_negatives(rng: np.random.RandomState) -> Dict[str, np.ndarray]:
    """Негативы: мусор, который НЕ должен опознаваться как знак."""
    out: Dict[str, np.ndarray] = {}
    out["white_noise"] = (rng.rand(64, 64) > 0.5).astype(np.float32)
    out["blurred_noise"] = np.array(
        Image.fromarray((rng.rand(64, 64) * 255).astype(np.uint8), "L")
        .filter(__import__("PIL.ImageFilter", fromlist=["ImageFilter"]).GaussianBlur(3.0)),
        dtype=np.float32) / 255.0
    stripes = np.zeros((64, 64), dtype=np.float32)
    stripes[::8, :] = 1.0
    out["stripes"] = stripes
    blob = np.zeros((64, 64), dtype=np.float32)
    blob[20:44, 20:44] = 1.0
    out["solid_blob"] = blob
    out["blank"] = np.zeros((64, 64), dtype=np.float32)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", default="KAISHU")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    from cangjie_composer import RecursiveComposer
    from cangjie_whole_glyph import WholeGlyphMatcher
    from cangjie_fullchar_bank import render_full_char_in_era

    composer = RecursiveComposer(data_dir=args.data_dir)
    matcher = WholeGlyphMatcher(era=args.era, data_dir=args.data_dir)
    if not matcher.is_available():
        print(f"Банк полных знаков для эры {args.era} не собран — нечего проверять.")
        print("Соберите: python3 cangjie_fullchar_bank.py --build --all")
        return 1

    all_chars = TEST_CHARS + ENCLOSURE_CHARS
    results: List[Dict[str, Any]] = []
    hits = 0
    total = 0
    failed_render: List[str] = []

    print("=" * 92)
    print(f"ПРОВЕРКА ПРЯМОГО ОПОЗНАНИЯ — эра {args.era}")
    print("=" * 92)
    print(f"{'знак':<6}{'top-1':<8}{'верно':<7}{'conf':<9}{'статус':<14}score")
    print("-" * 92)

    for ch in all_chars:
        # ВАЖНО: запрос рендерим ТЕМ ЖЕ путём, что и шаблон банка.
        # Иначе сравниваются два разных рендерера и top-1 падает
        # не из-за качества матчера, а из-за несовпадения картинок.
        ink = render_full_char_in_era(ch, args.era, composer=composer,
                                      data_dir=args.data_dir)
        if ink is None:
            failed_render.append(ch)
            continue
        photo = synth_photo(ink)
        res = matcher.match_whole_glyph(photo)
        total += 1
        correct = (res.get("best") == ch)
        if correct:
            hits += 1
        results.append({
            "char": ch, "expected": ch, "best": res.get("best"),
            "correct": correct, "confidence": res.get("confidence"),
            "status": res.get("status"), "raw_score": res.get("raw_score"),
        })
        print(f"{ch:<6}{str(res.get('best')):<8}{'ДА' if correct else 'НЕТ':<7}"
              f"{res.get('confidence', 0):<9.4f}{res.get('status', ''):<14}"
              f"{res.get('raw_score', 0):.4f}")

    # ---- негативы ----
    rng = np.random.RandomState(42)
    negs = make_negatives(rng)
    false_accepts = 0
    print()
    print("-" * 92)
    print("НЕГАТИВЫ (мусор не должен опознаваться как знак):")
    for name, m in negs.items():
        r = matcher.match_whole_glyph(m)
        fa = r.get("status") == "RECOGNIZED"
        if fa:
            false_accepts += 1
        print(f"  {name:<16} status={r.get('status'):<14} best={str(r.get('best')):<6} "
              f"conf={r.get('confidence', 0):.4f} {'ЛОЖНОЕ СРАБАТЫВАНИЕ' if fa else 'ok'}")

    acc = hits / total if total else 0.0
    far = false_accepts / len(negs) if negs else 0.0

    print()
    print("=" * 92)
    print(f"top-1 accuracy : {acc:.4f}  ({hits}/{total})")
    print(f"false accept rate: {far:.4f}  ({false_accepts}/{len(negs)})")
    if failed_render:
        print(f"не отрендерилось: {''.join(failed_render)}")
    print("=" * 92)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({
                "era": args.era,
                "top1_accuracy": acc,
                "hits": hits, "total": total,
                "false_accept_rate": far,
                "false_accepts": false_accepts, "n_negatives": len(negs),
                "cases": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"Машинный отчёт: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
