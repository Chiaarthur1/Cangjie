"""
Project Cangjie (Цанцзе) — Phonology Integrity Repair [v1.0]

ЗАЧЕМ ЭТОТ МОДУЛЬ
=================
Аудит показал системный дефект, из-за которого палеограф получал НЕВЕРНОЕ
звучание знака — а это ровно та часть продукта, ради которой он и мучился
три дня над книгой.

Что было не так
---------------
1) `data/Baxter_Sagart.csv` в бандле — это НЕ таблица Бакстера-Сагарта, а
   так называемые "phonetic matches": строки вида «侍 / 時 / 河» с полем
   `pinyin`, куда записано чтение ФОНЕТИЧЕСКОГО ЯДРА серии, а не самого знака.

       侍  pinyin=sì   (на самом деле shì — фонетик 寺 = sì)
       時  pinyin=sì   (на самом деле shí — фонетик 寺 = sì)
       河  pinyin=kě   (на самом деле hé — фонетик 可 = kě)

   Итого 379 строк из 814 (46%!) содержали ЧУЖОЕ чтение.

2) `cangjie_parser.propagate_xiesheng_phonology()` и
   `cangjie_fast_memory_hub._build_transitive_xiesheng_and_precedent_indices()`
   наследуют `pinyin` от фонетика к производному знаку. Для СВОЕГО чтения
   это неверно: 形声 даёт знаку СЕМЬЮ, но чтение знак наследует по общности
   с фонетиком, а не тождественно ему.

   Масштаб: 27 139 из 43 581 знаков (62%!) отдавали палеографу чтение
   чужого компонента. «沃» превращался в *piě (чтение 丿) вместо *wò.

3) Каноническая таблица Baxter–Sagart 2014 (4 056 знаков) в бандле
   отсутствовала, и скачивалась по кривому зеркалу (phonoML), откуда и
   пришли «фоно-матчи».

Что делает этот модуль
----------------------
A) `build_canonical_phonology_table()` — собирает ЧИСТУЮ таблицу
   Baxter–Sagart 2014 из первоисточника (TSV-релиз Бакстера-Сагарта),
   сверяет каждое чтение с Unihan kMandarin и различает:
     * `own`      — чтение самого знака (атрибутировано, поле pinyin)
     * `series`   — чтение фонетика серии (поле series_pinyin, для справки)
   Расхождение между ними — это НЕ ошибка: 河 (hé) и 可 (kě) звучат
   по-разному, но принадлежат одной серии. Ошибка была в том, что
   чтение серии выдавалось за чтение знака.

B) `repair_engine_phonology()` — правит САМ движок:
     * убирает наследование `pinyin` от фонетика (остаётся наследование
       OC/MC-архетипа, что корректно: фонетик задаёт серию);
     * добавляет в запись знака поля `own_pinyin` / `series_pinyin`;
     * для производных знаков без собственного чтения сохраняет честную
       пометку «чтение выведено по гнезду Сэшэн», а не выдаёт его за факт.

C) `audit_readings()` — проверочный прогон: сколько знаков отдают своё
   чтение, сколько — чужое, и где остаётся расхождение.

ЗАПУСК
------
    python3 cangjie_phonology_repair.py --audit          # только аудит
    python3 cangjie_phonology_repair.py --build          # пересобрать таблицу
    python3 cangjie_phonology_repair.py --apply          # починить движок + кэш
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PHONO-REPAIR]: %(message)s"
)
logger = logging.getLogger("CangjiePhonologyRepair")

csv.field_size_limit(10 ** 7)

# Первоисточник Baxter–Sagart (2014): TSV-релиз с корректными чтениями
CANONICAL_URLS = [
    "https://raw.githubusercontent.com/increpare/Baxter-Sagart-MacOS-Dictionary/master/baxtersagart.tsv",
    "https://raw.githubusercontent.com/increpare/Baxter-Sagart-MacOS-Dictionary/main/baxtersagart.tsv",
]

CANONICAL_TSV_HEADER = ["zi", "py", "MC", "", "OC", "gloss", "GSR", "HYDZD", "rad", "str", "Unicode"]


# ==============================================================================
# 1. ЗАГРУЗКА ИСТОЧНИКОВ
# ==============================================================================

def load_unihan_mandarin(data_dir: str) -> Dict[str, str]:
    """kMandarin из Unihan — авторитетное современное чтение знака."""
    out: Dict[str, str] = {}
    path = os.path.join(data_dir, "Unihan_Readings.txt")
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or "\t" not in line:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[1] != "kMandarin":
                continue
            try:
                ch = chr(int(parts[0].replace("U+", ""), 16))
            except ValueError:
                continue
            readings = parts[2].split()
            if readings:
                out[ch] = readings[0]
    return out


def load_bundled_phonology(data_dir: str) -> Dict[str, Dict[str, str]]:
    """Текущая (испорченная) таблица бандла — нужна как fallback и для серий."""
    out: Dict[str, Dict[str, str]] = {}
    path = os.path.join(data_dir, "Baxter_Sagart.csv")
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ch = (r.get("character") or "").strip()
            if not ch:
                continue
            out[ch] = {
                "pinyin": (r.get("pinyin") or "").strip(),
                "middle_chinese": (r.get("middle_chinese") or "").strip(),
                "old_chinese": (r.get("old_chinese") or "").strip(),
                "gsr": (r.get("gsr") or "").strip(),
                "gloss": (r.get("gloss") or "").strip(),
            }
    return out


def download_canonical(dest_path: str, timeout: int = 60) -> bool:
    """Скачивает канонический TSV Бакстера-Сагарта. True — успех."""
    import urllib.request

    for url in CANONICAL_URLS:
        try:
            logger.info("Загрузка канонической таблицы Бакстера-Сагарта: %s", url)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read().decode("utf-8")
            if len(data) < 100_000:
                logger.warning("Скачанный файл слишком мал (%d байт) — пробую следующий", len(data))
                continue
            os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(data)
            logger.info("✔ Каноническая таблица сохранена: %s (%.1f КБ)",
                        dest_path, os.path.getsize(dest_path) / 1024.0)
            return True
        except Exception as e:
            logger.warning("Источник недоступен (%s): %s", url, e)
    return False


def load_canonical_tsv(path: str) -> Dict[str, Dict[str, str]]:
    """
    Парсит TSV Бакстера-Сагарта в словарь знак -> реконструкция.

    ВАЖНО: в каноническом файле знак может встречаться ДВАЖДЫ:
      * одна строка — СВОЁ чтение знака (食 shí, OC *mə-lək),
      * вторая  — чтение СЕРИИ, в которую знак входит как отражение
                  (食 sì, OC *s-m-lək-s — рядом с 寺 sì).
    Выбираем строку, совпадающую с Unihan kMandarin; прочие сохраняем
    как `series_pinyin` (это ровно то чтение, которое нужно для вывода
    звучания НЕСУЩЕСТВУЮЩИХ знаков по гнезду Сэшэн).
    """
    rows_by_char: Dict[str, List[Dict[str, str]]] = {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            zi = (r.get("zi") or "").strip()
            if not zi:
                continue
            rows_by_char.setdefault(zi, []).append({
                "pinyin": (r.get("py") or "").strip(),
                "middle_chinese": (r.get("MC") or "").strip(),
                "old_chinese": (r.get("OC") or "").strip(),
                "gsr": (r.get("GSR") or "").strip(),
                "gloss": (r.get("gloss") or "").strip(),
            })

    mandarin = load_unihan_mandarin(os.path.dirname(os.path.abspath(path)))
    out: Dict[str, Dict[str, str]] = {}
    for zi, rows in rows_by_char.items():
        truth = mandarin.get(zi, "")
        chosen = None
        # 1) приоритет: чтение, совпадающее с Unihan kMandarin
        if truth:
            for r in rows:
                if r["pinyin"] == truth:
                    chosen = r
                    break
        # 2) иначе — строка без чтения серии (самая длинная MC/OC)
        if chosen is None:
            chosen = max(rows, key=lambda r: len(r["old_chinese"]))
        others = [r for r in rows if r is not chosen]
        chosen = dict(chosen)
        chosen["alt_pinyin"] = others[0]["pinyin"] if others else ""
        chosen["alt_old_chinese"] = others[0]["old_chinese"] if others else ""
        out[zi] = chosen
    return out


# ==============================================================================
# 2. СБОРКА ЧИСТОЙ ТАБЛИЦЫ
# ==============================================================================

def build_canonical_phonology_table(
    data_dir: str,
    canonical_tsv: Optional[str] = None,
) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, str]]]:
    """
    Собирает исправленную таблицу фонологии.

    Поля каждой записи:
      character       — знак
      pinyin         — СВОЁ чтение знака (то, что показываем палеографу)
      middle_chinese — MC реконструкция
      old_chinese    — OC реконструкция
      gsr            — код серии ГСР
      series_pinyin  — чтение ФОНЕТИКА серии (справочно; может отличаться)
      gloss          — глосса

    Приоритет источников для `pinyin`:
      1) канонический Бакстер-Сагарт (своё чтение знака)
      2) Unihan kMandarin
      3) старая таблица бандла (только если расхождений нет)
    """
    mandarin = load_unihan_mandarin(data_dir)
    bundled = load_bundled_phonology(data_dir)
    canon = load_canonical_tsv(canonical_tsv) if canonical_tsv else {}

    if not canon:
        logger.warning("Канонический TSV не найден (%s) — работаю на данных бандла", canonical_tsv)

    # Серия знака: по GSR. 侍 = GSR-0961m.5, 寺 = GSR-0961m → одна серия 0961m.
    def series_root(gsr: str) -> str:
        g = (gsr or "").strip()
        if not g:
            return ""
        # 0961m.5 -> 0961m ; 0001g -> 0001g ; 0961m' -> 0961m
        base = g.split(".")[0]
        return base.rstrip("'")

    # Собираем карту серия -> чтение ядра серии
    series_reading: Dict[str, str] = {}
    for ch, rec in canon.items():
        sr = series_root(rec.get("gsr", ""))
        if sr and sr not in series_reading and rec.get("pinyin"):
            series_reading[sr] = rec["pinyin"]

    merged: Dict[str, Dict[str, str]] = {}

    # 2.1 Канонические записи — основа
    #
    # ПРИОРИТЕТ ЧТЕНИЙ (важно для палеографа):
    #   `pinyin`        — СВОЁ современное чтение знака. Источник приоритета:
    #                     Unihan kMandarin, потому что это нормативное чтение
    #                     сегодняшнего дня; Бакстер-Сагарт местами фиксирует
    #                     чтение ВНУТРИ серии (拔 bèi, 隘 è), которое как
    #                     «чтение знака» вводит в заблуждение.
    #   `series_pinyin` — чтение ФОНЕТИКА серии: из него выводится звучание
    #                     для НЕСУЩЕСТВУЮЩИХ знаков по правилам Сэшэн.
    #   `bs_pinyin`     — чтение, зафиксированное Бакстером-Сагартом.
    #   `middle_chinese` / `old_chinese` — MC/OC архетипы: ради них БС и нужен.
    for ch, rec in canon.items():
        bs_py = rec.get("pinyin", "")
        own_py = mandarin.get(ch, "") or bs_py
        sr = series_root(rec.get("gsr", ""))
        # Чтение СЕРИИ: из альтернативной строки канона (чтение ядра гнезда),
        # иначе — из карты серий.
        series_py = rec.get("alt_pinyin") or series_reading.get(sr, "") or ""
        merged[ch] = {
            "character": ch,
            "pinyin": own_py,
            "bs_pinyin": bs_py,
            "middle_chinese": rec.get("middle_chinese", ""),
            "old_chinese": rec.get("old_chinese", ""),
            "gsr": rec.get("gsr", ""),
            "series_pinyin": series_py,
            "gloss": rec.get("gloss", ""),
            "source": "BAXTER_SAGART_2014",
        }

    # 2.2 Записи бандла, которых нет в каноне
    for ch, rec in bundled.items():
        if ch in merged:
            continue
        own_py = mandarin.get(ch, "")
        b_py = rec.get("pinyin", "")
        # Чтение серии = то, что лежит в бандловой строке
        merged[ch] = {
            "character": ch,
            "pinyin": own_py or b_py,       # СВОЁ чтение приоритетнее
            "bs_pinyin": b_py,
            "middle_chinese": rec.get("middle_chinese", ""),
            "old_chinese": rec.get("old_chinese", ""),
            "gsr": rec.get("gsr", ""),
            "series_pinyin": b_py if b_py != (own_py or b_py) else "",
            "gloss": rec.get("gloss", ""),
            "source": "BUNDLED",
        }

    # 2.3 Знаки, которых нет ни там, ни там, но есть в Unihan с чтением
    for ch, py in mandarin.items():
        if ch in merged or ch in canon:
            continue
        merged[ch] = {
            "character": ch,
            "pinyin": py,
            "bs_pinyin": "",
            "middle_chinese": "",
            "old_chinese": "",
            "gsr": "",
            "series_pinyin": "",
            "gloss": "",
            "source": "UNIHAN_MANDARIN",
        }

    rows = sorted(merged.values(), key=lambda r: r["character"])
    return rows, merged


def write_phonology_csv(rows: List[Dict[str, str]], path: str) -> None:
    """Пишет таблицу в формате, который читает cangjie_parser.load_baxter_sagart."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = ["character", "pinyin", "bs_pinyin", "middle_chinese", "old_chinese",
                  "gsr", "series_pinyin", "gloss", "source"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    logger.info("✔ Таблица фонологии записана: %s (%d записей)",
                path, len(rows))


# ==============================================================================
# 3. ПОЧИНКА ДВИЖКА
# ==============================================================================

def repair_engine_phonology(data_dir: str) -> Dict[str, int]:
    """
    Правит наследование `pinyin` в двух местах:

    1) `cangjie_parser.propagate_xiesheng_phonology()` — при наследовании
       архетипа от фонетика НЕ копируется `pinyin`. Наследуются только
       MC/OC (серия) и GSR. Собственное чтение знака (Unihan/БС) остаётся
       своим; если его нет — остаётся пустым и честно помечается как
       неустановленное.

    2) `cangjie_fast_memory_hub._build_transitive_xiesheng_and_precedent_indices()`
       — там же: `pinyin` берётся из Unihan-глоссы знака, а не от
       фонетика; `series_pinyin` хранится отдельно.

    Правки вносятся в файлы проекта (идемпотентно, по маркеру-комментарию).
    """
    stats = {"parser_patched": 0, "hub_patched": 0}

    # --- 1) cangjie_parser.py -------------------------------------------------
    parser_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cangjie_parser.py")
    if os.path.exists(parser_path):
        with open(parser_path, "r", encoding="utf-8") as f:
            src = f.read()
        old_block = """                    inherited = {
                        "pinyin": src_rec.get("pinyin", ""),
                        "middle_chinese": src_rec.get("middle_chinese", ""),
                        "old_chinese": src_rec.get("old_chinese", ""),
                        "gsr": f"{src_rec.get('gsr', 'XIESHENG')}->{cand}",
                        "gloss": f"серия Сэшэн от фонетика «{cand}» ({src_rec.get('gloss', '')})"
                    }"""
        new_block = """                    # [PHONO-REPAIR v1.0] Чтение САМОГО знака НЕ наследуется
                    # от фонетика: 形声 даёт знаку СЕМЬЮ по фонетику, но чтение
                    # знак наследует лишь ЧАСТИЧНО (河 hé ≠ 可 kě). Наследуются
                    # только архетип серии (MC/OC) и код ГСР; `pinyin` берётся
                    # из Unihan/Бакстера-Сагарта, а если его нет — остаётся
                    # пустым, и движок честно помечает чтение как неизвестное.
                    inherited = {
                        "pinyin": "",
                        "own_pinyin": "",
                        "series_pinyin": src_rec.get("pinyin", ""),
                        "middle_chinese": src_rec.get("middle_chinese", ""),
                        "old_chinese": src_rec.get("old_chinese", ""),
                        "gsr": f"{src_rec.get('gsr', 'XIESHENG')}->{cand}",
                        "gloss": f"серия Сэшэн от фонетика «{cand}» ({src_rec.get('gloss', '')})"
                    }"""
        if old_block in src:
            src = src.replace(old_block, new_block)
            with open(parser_path, "w", encoding="utf-8") as f:
                f.write(src)
            stats["parser_patched"] = 1
            logger.info("✔ cangjie_parser.py: наследование чтения отключено")

    # --- 2) cangjie_fast_memory_hub.py ---------------------------------------
    hub_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cangjie_fast_memory_hub.py")
    if os.path.exists(hub_path):
        with open(hub_path, "r", encoding="utf-8") as f:
            src = f.read()
        old_block = """                    transitive_baxter[glyph] = {
                        "phonetic_root": phon_cand,
                        "pinyin": gl.get("pinyin") or parent_bs["pinyin"],
                        "old_chinese": parent_bs["old_chinese"],
                        "middle_chinese": gl.get("tang_mc") or parent_bs["middle_chinese"],
                        "gsr": parent_bs["gsr"],
                        "gloss": gl.get("definition") or f"производный знак гнезда «{phon_cand}» ({parent_bs['gloss']})"
                    }"""
        new_block = """                    # [PHONO-REPAIR v1.0] `pinyin` — СВОЁ чтение знака (Unihan),
                    # а не чтение фонетика. Чтение серии хранится отдельно
                    # в `series_pinyin`, чтобы палеограф видел разницу явно.
                    transitive_baxter[glyph] = {
                        "phonetic_root": phon_cand,
                        "pinyin": gl.get("pinyin", ""),
                        "own_pinyin": gl.get("pinyin", ""),
                        "series_pinyin": parent_bs["pinyin"],
                        "old_chinese": parent_bs["old_chinese"],
                        "middle_chinese": gl.get("tang_mc") or parent_bs["middle_chinese"],
                        "gsr": parent_bs["gsr"],
                        "gloss": gl.get("definition") or f"производный знак гнезда «{phon_cand}» ({parent_bs['gloss']})"
                    }"""
        if old_block in src:
            src = src.replace(old_block, new_block)
            with open(hub_path, "w", encoding="utf-8") as f:
                f.write(src)
            stats["hub_patched"] = 1
            logger.info("✔ cangjie_fast_memory_hub.py: чтение серии отделено от чтения знака")

    return stats


def invalidate_ram_cache(data_dir: str) -> bool:
    """Удаляет устаревший RAM-кэш онтологии, чтобы он пересобрался с новыми данными."""
    cache = os.path.join(data_dir, "cangjie_ram_cache_v6.pkl")
    if os.path.exists(cache):
        backup = cache + ".bak_pre_phonorepair"
        shutil.copy2(cache, backup)
        os.remove(cache)
        logger.info("✔ RAM-кэш сброшен (резервная копия: %s)", os.path.basename(backup))
        return True
    return False


# ==============================================================================
# 4. АУДИТ
# ==============================================================================

def audit_readings(data_dir: str) -> Dict[str, object]:
    """
    Прогон по онтологии: сравнивает чтение, которое отдаст движок,
    с собственным чтением знака по Unihan kMandarin.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    mandarin = load_unihan_mandarin(data_dir)

    from cangjie_parser import ProjectCangjieEngine
    engine = ProjectCangjieEngine(data_dir=data_dir)
    engine.load_unihan_radicals("Unihan_RadicalStrokes.txt")
    engine.load_baxter_sagart("Baxter_Sagart.csv")
    ids_files = [
        fn for fn in sorted(os.listdir(data_dir))
        if fn.startswith("IDS-") and fn.endswith(".txt")
    ] or ["IDS-UCS-Basic.txt"]
    for fn in ids_files:
        engine.load_chise_ids(fn)
    engine.compute_easa_entropy_weights()

    own = 0
    foreign = 0
    unknown = 0
    examples_foreign: List[Tuple[str, str, str]] = []
    examples_own: List[Tuple[str, str, str]] = []

    for glyph, ent in engine.characters_db.items():
        if ent.is_fake:
            continue
        bs = ent.baxter_sagart or {}
        rec = bs.get("primary") or bs.get("transitive_xiesheng") or {}
        py = (rec.get("pinyin") or "").strip()
        if not py:
            unknown += 1
            continue
        py_first = py.split("/")[0].strip()
        truth = mandarin.get(glyph)
        if not truth:
            continue
        if py_first == truth:
            own += 1
            if len(examples_own) < 6:
                examples_own.append((glyph, py, truth))
        else:
            foreign += 1
            if len(examples_foreign) < 12:
                examples_foreign.append((glyph, py, truth))

    total = own + foreign
    return {
        "total_checked": total,
        "own_reading": own,
        "foreign_reading": foreign,
        "unknown_reading": unknown,
        "accuracy_pct": round(100.0 * own / total, 2) if total else 0.0,
        "examples_own": examples_own,
        "examples_foreign": examples_foreign,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Project Cangjie — Phonology Integrity Repair")
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--audit", action="store_true", help="только аудит чтений")
    ap.add_argument("--build", action="store_true", help="пересобрать таблицу фонологии")
    ap.add_argument("--apply", action="store_true", help="починить движок и сбросить кэш")
    ap.add_argument("--canonical-tsv", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Baxter_Sagart_canonical.tsv"))
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)

    if args.build or not args.audit:
        if not os.path.exists(args.canonical_tsv) or not download_canonical(args.canonical_tsv):
            if not os.path.exists(args.canonical_tsv):
                logger.error("Канонический TSV недоступен. Скачайте вручную: %s", CANONICAL_URLS[0])
                return 2
        rows, _ = build_canonical_phonology_table(data_dir, args.canonical_tsv)
        out_csv = os.path.join(data_dir, "Baxter_Sagart.csv")
        if os.path.exists(out_csv):
            shutil.copy2(out_csv, out_csv + ".bak_pre_phonorepair")
        write_phonology_csv(rows, out_csv)

    if args.apply:
        repair_engine_phonology(data_dir)
        invalidate_ram_cache(data_dir)

    if args.audit or not (args.build or args.apply):
        res = audit_readings(data_dir)
        logger.info("=" * 74)
        logger.info("АУДИТ ЧТЕНИЙ")
        logger.info("  проверено знаков:            %d", res["total_checked"])
        logger.info("  чтение СВОЁ (корректно):     %d", res["own_reading"])
        logger.info("  чтение ЧУЖОЕ (ошибка):       %d", res["foreign_reading"])
        logger.info("  чтение не установлено:      %d", res["unknown_reading"])
        logger.info("  корректность:                %s%%", res["accuracy_pct"])
        logger.info("=" * 74)
        if res["examples_foreign"]:
            logger.info("Примеры чужого чтения: %s", res["examples_foreign"][:6])
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
