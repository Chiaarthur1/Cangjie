"""
Project Cangjie — честная оценка сборки/разбора IDS [v10.0].

Проверяет структурную верность парсера IDS на РЕАЛЬНЫХ записях CHISE
(без синтетики): парсим каждую формулу в AST и восстанавливаем строку IDS
обратно; совпадение = корректный round-trip. Также считает распределение
по операторам (⿰ ⿱ ⿲ ⿳ ⿴ …).

Никаких заглушек: метрики считаются по фактическим файлам data/IDS-UCS-*.txt.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

from cangjie_parser import IDSParser, ALL_OPERATORS


def load_chise_ids(data_dir: str) -> List[Tuple[str, str]]:
    """Возвращает список (иероглиф, IDS) из локальных файлов CHISE."""
    pairs: List[Tuple[str, str]] = []
    names = ["IDS-UCS-Basic.txt"]
    if os.path.isdir(data_dir):
        names += sorted(
            f for f in os.listdir(data_dir)
            if (f.startswith("IDS-UCS-Ext") or f.startswith("IDS-UCS-Compat")) and f.endswith(".txt")
        )
    for fname in names:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith(";;") or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                glyph = parts[1].strip()
                ids_field = parts[2].strip()
                # CHISE может давать несколько вариантов IDS через пробел — берём первый.
                ids = ids_field.split()[0] if ids_field.split() else ""
                if glyph and ids:
                    pairs.append((glyph, ids))
    return pairs


def ast_to_ids(node) -> str:
    """Восстанавливает строку IDS из AST (обратная операция к parse_sequence)."""
    if node is None:
        return ""
    if node.node_type == "component":
        return node.value
    return node.value + "".join(ast_to_ids(c) for c in node.children)


def is_composite(ids: str) -> bool:
    return any(tok in ALL_OPERATORS for tok in ids)


def run_eval(data_dir: str, max_samples: int = 3000, seed: int = 2026) -> Dict[str, object]:
    pairs = load_chise_ids(data_dir)
    composites = [(g, i) for g, i in pairs if is_composite(i)]

    import random
    rng = random.Random(seed)
    sample = composites if len(composites) <= max_samples else rng.sample(composites, max_samples)

    # Структурный round-trip не требует словаря радикалов.
    parser = IDSParser(radical_map={}, phonetic_set=set())

    parsed_ok = 0
    roundtrip_ok = 0
    op_counts: Dict[str, int] = {}
    op_roundtrip: Dict[str, int] = {}
    failures: List[Dict[str, str]] = []

    for glyph, ids in sample:
        root_op = next((t for t in ids if t in ALL_OPERATORS), "?")
        op_counts[root_op] = op_counts.get(root_op, 0) + 1
        ast = parser.parse_sequence(ids)
        if ast is None:
            failures.append({"glyph": glyph, "ids": ids, "error": "parse_failed"})
            continue
        parsed_ok += 1
        rebuilt = ast_to_ids(ast)
        if rebuilt == ids:
            roundtrip_ok += 1
            op_roundtrip[root_op] = op_roundtrip.get(root_op, 0) + 1
        else:
            failures.append({"glyph": glyph, "ids": ids, "rebuilt": rebuilt, "error": "roundtrip_mismatch"})

    n = len(sample)
    return {
        "total_composites_loaded": len(composites),
        "sampled": n,
        "parsed_ok": parsed_ok,
        "roundtrip_ok": roundtrip_ok,
        "roundtrip_accuracy": round(roundtrip_ok / n, 4) if n else 0.0,
        "operator_distribution": {k: op_counts[k] for k in sorted(op_counts)},
        "operator_roundtrip": {k: op_roundtrip.get(k, 0) for k in sorted(op_counts)},
        "failures": failures[:20],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Project Cangjie — честная оценка round-trip IDS (CHISE)")
    ap.add_argument("--data-dir", default="./data", help="Папка с IDS-UCS-*.txt")
    ap.add_argument("--samples", type=int, default=3000, help="Сколько композитных IDS оценить")
    ap.add_argument("--out", default=None, help="Сохранить JSON-отчёт")
    args = ap.parse_args()

    report = run_eval(args.data_dir, max_samples=args.samples)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Отчёт сохранён: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
