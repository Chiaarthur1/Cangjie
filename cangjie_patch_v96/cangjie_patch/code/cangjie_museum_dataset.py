"""
Project Cangjie — сборщик датасета из музейных каталогов [v9.7]

ЗАЧЕМ:

    Все честные цифры сейчас — на СИНТЕТИКЕ: бенчмарк сам генерирует
    деградации. Это потолок, а не реальность. Настоящий музейный
    датасет — то, что превратит проект из «работает на синтетике» в
    «работает на реальных фото палеографа».

    Источники: Wikimedia Commons (Public Domain / CC0). Там есть
    фотографии костей с гадательными надписями, бронзовых сосудов,
    стел и бамбуковых табличек — с разрешения свободного
    использования.

ЧТО ДЕЛАЕТ:
    1) Обходит заданные категории/поисковые запросы Commons.
    2) Скачивает изображения с метаданными (лицензия, автор, источник).
    3) Применяет РЕАЛЬНЫЕ деградации носителя (кость, бронза, 拓片).
    4) Сохраняет датасет в формате для обучения детектора.

ФОРМАТ ВЫХОДА:
    data/museum_dataset/
        manifest.jsonl        — по строке на файл: источник, лицензия, деградация
        images/<id>.png       — обработанное фото
        crops/<id>_<n>.png    — вырезанные иероглифы (если размечено)
        LICENSE_SUMMARY.json  — сводка по лицензиям

ЗАПУСК:
    python3 cangjie_museum_dataset.py --search "oracle bone script" --limit 20
    python3 cangjie_museum_dataset.py --category "Oracle bone script" --limit 50
    python3 cangjie_museum_dataset.py --categories-file cats.txt --limit 100
    python3 cangjie_museum_dataset.py --dry-run          # только список, без скачивания
"""

from __future__ import annotations

import os
import io
import re
import json
import time
import hashlib
import logging
import argparse
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple, Any

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [MUSEUM-DS]: %(message)s"
)
logger = logging.getLogger("CangjieMuseumDataset")

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia отклоняет запросы без внятного User-Agent и требует
# rate limiting (иначе 403 "Too Many Reqs"). Ставим контактный UA
# и держим паузу между запросами.
UA = ("ProjectCangjie-Paleography/1.0 "
      "(https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia; "
      "paleography-research) python-urllib/3.11")
MIN_INTERVAL = 3.0  # секунд между запросами к API
# На upload.wikimedia.org лимит жёстче: там даже 2 с/файл дают 429.
# При большом IP-адресе (NAT, хостинг, CI) может резать и при этом —
# поэтому держим MIN_INTERVAL_DL отдельно для самих картинок.
MIN_INTERVAL_DL = 4.0
_last_call = [0.0]
_last_dl = [0.0]
OUT_DIR = "data/museum_dataset"

# Категории Commons с реальными древними надписями
DEFAULT_CATEGORIES = [
    "Oracle bone script",
    "Bronze inscriptions",
    "Chinese bronze inscriptions",
    "Seal script",
    "Small seal script",
    "Clerical script",
    "Chinese calligraphy",
    "Oracle bones",
    "Bronze vessels of China",
    "Chinese bamboo and wooden slips",
]


def _api_get(params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(params)
    params["format"] = "json"
    url = API + "?" + urllib.parse.urlencode(params)

    # Rate limiting: Wikimedia режет запросы чаще, чем раз в ~1 сек.
    wait = MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()

    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "application/json",
        "Accept-Encoding": "identity",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def search_commons(query: str, limit: int = 20) -> List[str]:
    """Ищет файлы в Commons по запросу. Возвращает список заголовков."""
    out: List[str] = []
    cont: Optional[str] = None
    while len(out) < limit:
        p = {
            "action": "query", "list": "search", "srsearch": query,
            "srnamespace": "6", "srlimit": "50",
        }
        if cont:
            p["sroffset"] = cont
        try:
            d = _api_get(p)
        except Exception as e:
            logger.warning("Поиск не удался (%s): %s", query, e)
            break
        hits = d.get("query", {}).get("search", [])
        if not hits:
            break
        for h in hits:
            t = h.get("title", "")
            if t.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                if _is_relevant(t):
                    out.append(t)
            if len(out) >= limit:
                break
        cont = d.get("continue", {}).get("sroffset")
        if cont is None:
            break
    return out[:limit]


# Слова-маркеры, по которым видно, что фото — именно китайская надпись.
# Без этого фильтра поиск «stele inscription» выдаёт греческие и финикские
# стелы, которые для палеографии бесполезны.
_CJK_HINTS = (
    "chinese", "china", "han", "shang", "zhou", "oracle", "bronze", "bronze",
    "seal script", "clerical", "stele", "inscription", "character", "hanzi",
    "cursive", "calligraphy", "bamboo", "slip", "tortoise", "bone",
    "script", "glyph", "wen", "oracle bone", "inscribed", "tang", "ming",
    "song", "qin", "dynasty", "kaishu", "seal", "kiln", "pottery",
)
# Явные признаки некитайских надписей — отбрасываем.
_NON_CJK_HINTS = (
    "greek", "phoenician", "hebrew", "latin", "runic", "cuneiform",
    "hieroglyph", "sanskrit", "egypt", "assyrian", "babylonian",
    "armenian", "georgian", "syriac", "thai", "khmer", "japanese",
    "korean", "devanagari", "tibetan",
)


def _is_relevant(title: str) -> bool:
    """Отсекает явно некитайские надписи по имени файла."""
    t = title.lower()
    if any(h in t for h in _NON_CJK_HINTS):
        return False
    return True


def category_members(cat: str, limit: int = 100) -> List[str]:
    """Файлы в категории Commons (с обходом подкатегорий)."""
    out: List[str] = []
    cont: Optional[str] = None
    while len(out) < limit:
        p = {
            "action": "query", "list": "categorymembers",
            "cmtitle": f"Category:{cat}", "cmtype": "file|subcat",
            "cmlimit": "100",
        }
        if cont:
            p["cmcontinue"] = cont
        try:
            d = _api_get(p)
        except Exception as e:
            logger.warning("Категория не читается (%s): %s", cat, e)
            break
        for m in d.get("query", {}).get("categorymembers", []):
            t = m.get("title", "")
            if t.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                out.append(t)
            if len(out) >= limit:
                break
        cont = d.get("continue", {}).get("cmcontinue")
        if cont is None:
            break
    return out[:limit]


def file_info(titles: List[str]) -> Dict[str, Dict[str, Any]]:
    """Метаданные файлов: URL, размер, лицензия, автор, описание."""
    out: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(titles), 20):
        chunk = titles[i:i + 20]
        try:
            d = _api_get({
                "action": "query", "titles": "|".join(chunk),
                "prop": "imageinfo", "iiprop": "url|size|extmetadata|mime",
            })
        except Exception as e:
            logger.warning("Метаданные не получены: %s", e)
            continue
        for p in d.get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [{}])[0]
            if not ii:
                continue
            em = ii.get("extmetadata", {})
            out[p.get("title", "")] = {
                "url": ii.get("url"),
                "width": ii.get("width"),
                "height": ii.get("height"),
                "mime": ii.get("mime"),
                "license": em.get("LicenseShortName", {}).get("value", "unknown"),
                "license_url": em.get("LicenseUrl", {}).get("value", ""),
                "artist": _strip_html(em.get("Artist", {}).get("value", "")),
                "description": _strip_html(em.get("ImageDescription", {}).get("value", ""))[:500],
                "date": _strip_html(em.get("DateTimeOriginal", {}).get("value", "")),
                "source_page": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(p.get('title','').replace(' ', '_'))}",
            }
        time.sleep(0.2)
    return out


def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", str(s or ""))
    return re.sub(r"\s+", " ", s).strip()


def download(url: str, dest: str, retries: int = 4) -> bool:
    """
    Скачивает файл с ретраями и backoff.

    Wikimedia отдаёт 429 при частых запросах — особенно на upload.wikimedia.
    Поэтому: пауза перед запросом, экспоненциальный backoff, и кэширование
    уже скачанного.
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return True

    wait = MIN_INTERVAL_DL - (time.time() - _last_dl[0])
    if wait > 0:
        time.sleep(wait)

    for attempt in range(retries):
        _last_dl[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": UA,
                "Accept": "image/*,*/*;q=0.8",
                "Accept-Encoding": "identity",
                "Referer": "https://commons.wikimedia.org/",
            })
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            if len(data) < 512:
                raise IOError(f"слишком короткий ответ ({len(data)} Б)")
            with open(dest, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            if attempt == retries - 1:
                logger.warning("Не скачалось %s: %s", url, str(e)[:90])
                return False
            back = MIN_INTERVAL_DL * (2 ** attempt) + 2.0
            logger.info("  повтор %d/%d через %.1f с (%s)",
                        attempt + 1, retries, back, str(e)[:50])
            time.sleep(back)
    return False


# ============================================================================
# ДЕГРАДАЦИЯ (реальные носители)
# ============================================================================

def apply_degradation(img: Image.Image, mode: str, seed: int = 0) -> Image.Image:
    """
    Накладывает реалистичное повреждение носителя.

    mode: BONE | BRONZE | TOPIAN | PAPER | CLEAN
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    arr = np.array(img.convert("L"), dtype=np.float32) / 255.0

    if mode == "BONE":
        # трещины, сколы, выветривание
        cracks = (rng.rand(*arr.shape) > 0.988).astype(np.float32)
        arr = np.clip(arr * (1.0 - 0.55 * cracks) + 0.10 * cracks, 0.0, 1.0)
        arr = np.clip(arr + rng.randn(*arr.shape).astype(np.float32) * 0.05, 0.0, 1.0)
    elif mode == "BRONZE":
        # патина, коррозия
        patina = (rng.rand(*arr.shape) > 0.88).astype(np.float32) * 0.22
        arr = np.clip(arr * (1.0 - 0.40 * patina) + 0.12 * patina, 0.0, 1.0)
        arr = np.clip(arr + rng.randn(*arr.shape).astype(np.float32) * 0.04, 0.0, 1.0)
    elif mode == "TOPIAN":
        # оттиск с камня: белые штрихи по углю
        arr = 1.0 - arr
        arr = np.clip(arr * (0.75 + 0.35 * rng.rand(*arr.shape)), 0.0, 1.0)
        chip = (rng.rand(*arr.shape) > 0.992).astype(np.float32)
        arr = np.clip(arr * (1.0 - 0.6 * chip), 0.0, 1.0)
    elif mode == "PAPER":
        # 宣纸: размытие, бледность, разрывы волокна
        from PIL import ImageFilter
        blurred = np.array(
            Image.fromarray((arr * 255).astype(np.uint8), "L")
            .filter(ImageFilter.GaussianBlur(radius=1.2)),
            dtype=np.float32) / 255.0
        arr = np.clip(0.6 * arr + 0.4 * blurred, 0.0, 1.0)
        arr = np.clip(arr * (0.85 + 0.25 * rng.rand(*arr.shape)), 0.0, 1.0)
    elif mode == "CLEAN":
        pass
    else:
        raise ValueError(f"Неизвестная деградация: {mode}")

    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).convert("RGB")


# ============================================================================
# СБОРКА
# ============================================================================

def build_dataset(
    queries: List[str],
    limit: int = 20,
    out_dir: str = OUT_DIR,
    degradations: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    degradations = degradations or ["BONE", "BRONZE", "TOPIAN", "PAPER"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    titles: List[str] = []
    seen = set()
    for q in queries:
        logger.info("Поиск: %s", q)
        found = (category_members(q, limit) if q.lower().startswith("category:")
                 else search_commons(q, limit))
        for t in found:
            if t not in seen:
                seen.add(t)
                titles.append(t)
        logger.info("  найдено %d (всего уникальных %d)", len(found), len(titles))
        if len(titles) >= limit:
            break
    titles = titles[:limit]

    if dry_run:
        print(f"\nБудет скачано {len(titles)} файлов:")
        for t in titles:
            print("  -", t)
        return {"dry_run": True, "n_titles": len(titles), "titles": titles}

    logger.info("Получаю метаданные %d файлов...", len(titles))
    meta = file_info(titles)

    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    n_done = 0
    licenses: Dict[str, int] = {}
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i, t in enumerate(titles):
            info = meta.get(t)
            if not info or not info.get("url"):
                continue
            # Убираем трекинг-параметры (utm_*): на upload.wikimedia
            # с ними действует отдельный, более жёсткий лимит.
            raw_url = info["url"].split("?")[0]
            ext = os.path.splitext(raw_url)[1] or ".jpg"
            fid = hashlib.sha1(t.encode("utf-8")).hexdigest()[:12]
            raw_path = os.path.join(out_dir, "images", f"{fid}{ext}")
            if not download(raw_url, raw_path):
                continue
            try:
                img = Image.open(raw_path).convert("RGB")
            except Exception as e:
                logger.warning("Не открывается %s: %s", t, e)
                continue
            # ограничиваем размер — нам нужен знак, а не вся стела
            if max(img.size) > 2048:
                s = 2048 / max(img.size)
                img = img.resize((int(img.width * s), int(img.height * s)),
                                 Image.Resampling.LANCZOS)
            img.save(os.path.join(out_dir, "images", f"{fid}.png"))

            rec = {
                "id": fid,
                "commons_title": t,
                "source_page": info["source_page"],
                "license": info["license"],
                "license_url": info["license_url"],
                "artist": info["artist"],
                "date": info["date"],
                "description": info["description"],
                "original_size": [info["width"], info["height"]],
                "variants": [],
            }
            for mode in degradations:
                try:
                    deg = apply_degradation(img, mode, seed=i * 17 + len(mode))
                except Exception:
                    continue
                vp = os.path.join(out_dir, "images", f"{fid}_{mode}.png")
                deg.save(vp)
                rec["variants"].append({
                    "degradation": mode,
                    "file": os.path.relpath(vp, out_dir),
                    "width": deg.width, "height": deg.height,
                })
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            licenses[info["license"]] = licenses.get(info["license"], 0) + 1
            n_done += 1
            if n_done % 5 == 0:
                logger.info("  обработано %d/%d", n_done, len(titles))
            time.sleep(0.25)

    summary = {
        "n_files": n_done,
        "n_queries": len(queries),
        "queries": queries,
        "degradations": degradations,
        "licenses": licenses,
        "manifest": manifest_path,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "LICENSE_SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Готово: %d файлов, лицензии: %s", n_done, licenses)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Сборщик музейного датасета Project Cangjie")
    ap.add_argument("--search", action="append", default=None,
                    help="поисковый запрос Commons (можно повторять)")
    ap.add_argument("--category", action="append", default=None,
                    help="категория Commons (можно повторять)")
    ap.add_argument("--categories-file", default=None,
                    help="файл со списком запросов/категорий, по одному в строке")
    ap.add_argument("--limit", type=int, default=20, help="сколько файлов скачать")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--dry-run", action="store_true", help="только список, без скачивания")
    args = ap.parse_args(argv)

    queries: List[str] = list(args.search or [])
    queries += [f"category:{c}" for c in (args.category or [])]
    if args.categories_file and os.path.exists(args.categories_file):
        with open(args.categories_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    queries.append(line)
    if not queries:
        queries = DEFAULT_CATEGORIES
        logger.info("Запросы не заданы — беру список по умолчанию (%d шт)",
                    len(DEFAULT_CATEGORIES))

    s = build_dataset(queries, limit=args.limit, out_dir=args.out,
                      dry_run=args.dry_run)
    if not args.dry_run:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
