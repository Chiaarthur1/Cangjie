"""
Project Cangjie — пересборка component-банков [v9.6]

Использование:
    python3 cangjie_rebuild_banks.py --component --era KAISHU
    python3 cangjie_rebuild_banks.py --component --all
    python3 cangjie_rebuild_banks.py --fullchar --all
    python3 cangjie_rebuild_banks.py --all

Зачем отдельный скрипт: у `cangjie_paleo_matcher.py` нет CLI (это
библиотечный модуль), а пересборка нужна после смены DESCRIPTOR_VERSION —
старые кэши содержат слипшиеся маски (рендер в 64x64) и непригодны.
"""

from __future__ import annotations

import os
import sys
import time
import logging
import argparse
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [REBUILD]: %(message)s"
)
logger = logging.getLogger("CangjieRebuildBanks")

ERAS = ["KAISHU", "SHANG_JIAGUWEN", "ZHOU_JINWEN", "XIAOZHUAN"]


def rebuild_component(era: str, data_dir: str = "./data") -> int:
    from cangjie_paleo_matcher import PaleoTemplateBank
    t0 = time.perf_counter()
    bank = PaleoTemplateBank(era_mode=era, data_dir=data_dir)
    bank.build(force=True)
    ok = bank.is_ready
    logger.info("component %s: %s шаблонов за %.1f с (ready=%s)",
                era, len(bank.glyphs), time.perf_counter() - t0, ok)
    return 0 if ok else 1


def rebuild_fullchar(era: str, data_dir: str = "./data") -> int:
    from cangjie_fullchar_bank import build_fullchar_bank
    r = build_fullchar_bank(era, data_dir=data_dir, force=True)
    logger.info("fullchar %s: %s шаблонов", era, r.get("n_templates"))
    return 0 if r.get("n_templates") else 1


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Пересборка банков Project Cangjie")
    ap.add_argument("--component", action="store_true", help="component-банки")
    ap.add_argument("--fullchar", action="store_true", help="банк полных знаков")
    ap.add_argument("--all", action="store_true", help="и то и другое")
    ap.add_argument("--era", default="KAISHU")
    ap.add_argument("--data-dir", default="./data")
    args = ap.parse_args(argv)

    do_comp = args.component or args.all
    do_full = args.fullchar or args.all
    if not do_comp and not do_full:
        ap.print_help()
        return 1

    eras = ERAS if args.all else [args.era]
    rc = 0
    if do_comp:
        for era in eras:
            rc |= rebuild_component(era, args.data_dir)
    if do_full:
        for era in eras:
            rc |= rebuild_fullchar(era, args.data_dir)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
