"""
Project Cangjie — OBIMD → YOLO конвертер [v10.0].

Преобразует датасет OBIMD (Oracle Bone Inscription Multi-modal Dataset,
HuggingFace KLOBIP/OBIMD, CC BY-4.0) в формат YOLO с разбиением по ДОКУМЕНТАМ
(без утечки данных между train/val).

Без заглушек: если OBIMD отсутствует на диске — честная ошибка с инструкцией,
а не молчаливый пропуск. Разметка читается из реальных JSON-аннотаций.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple


def detect_obimd(obimd_dir: str) -> Dict[str, Any]:
    """Определяет, что именно лежит в obimd_dir."""
    d = os.path.abspath(obimd_dir)
    if not os.path.isdir(d):
        return {"present": False, "dir": d}
    entries = os.listdir(d)
    images_dir = next((e for e in entries if e.lower() in {"images", "image", "img"}), None)
    ann_files = [e for e in entries if e.lower().endswith(".json")]
    return {
        "present": True,
        "dir": d,
        "images_dir": images_dir,
        "annotation_files": ann_files,
    }


def _load_coco_style(ann_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Читает COCO-подобный JSON: {images:[{id,file_name}], annotations:[{image_id,category_id,bbox}]}."""
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = {img["id"]: img for img in data.get("images", [])}
    cats = data.get("categories", [])
    cat_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    return {"images": images, "categories": cats, "cat_to_idx": cat_to_idx}, data.get("annotations", [])


def convert_to_yolo(obimd_dir: str, out_dir: str, val_frac: float = 0.1, seed: int = 2026) -> Dict[str, Any]:
    info = detect_obimd(obimd_dir)
    if not info["present"]:
        return {
            "ok": False,
            "error": (
                f"OBIMD не найден в {info['dir']}. Скачайте: "
                "huggingface-cli download KLOBIP/OBIMD --repo-type dataset --local-dir data/OBIMD"
            ),
        }
    ann_path = None
    for cand in info["annotation_files"]:
        if "annot" in cand.lower() or "label" in cand.lower() or "train" in cand.lower():
            ann_path = os.path.join(info["dir"], cand)
            break
    if ann_path is None and info["annotation_files"]:
        ann_path = os.path.join(info["dir"], info["annotation_files"][0])
    if ann_path is None:
        return {"ok": False, "error": f"В {info['dir']} нет JSON-аннотаций (ожидался COCO-подобный файл)."}

    try:
        meta, annotations = _load_coco_style(ann_path)
    except Exception as e:
        return {"ok": False, "error": f"Не удалось разобрать аннотации {ann_path}: {e}"}

    images = meta["images"]
    cat_to_idx = meta["cat_to_idx"]
    if not images or not annotations:
        return {"ok": False, "error": f"Аннотации {ann_path} пусты или формат не распознан (COCO)."}

    # Разбиение по ДОКУМЕНТАМ: документ = префикс имени файла до разделителя.
    docs: Dict[str, List[int]] = {}
    for img_id, img in images.items():
        name = img.get("file_name", str(img_id))
        doc = name.split("_")[0] if "_" in name else name.split(".")[0]
        docs.setdefault(doc, []).append(img_id)

    import random
    rng = random.Random(seed)
    doc_list = sorted(docs)
    rng.shuffle(doc_list)
    n_val = max(1, int(len(doc_list) * val_frac))
    val_docs = set(doc_list[:n_val])

    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in annotations:
        by_image.setdefault(ann["image_id"], []).append(ann)

    written = {"train": 0, "val": 0}
    classes = {i: c for c, i in cat_to_idx.items()}

    for img_id, img in images.items():
        name = img.get("file_name", str(img_id))
        doc = name.split("_")[0] if "_" in name else name.split(".")[0]
        split = "val" if doc in val_docs else "train"
        w = float(img.get("width", 1))
        h = float(img.get("height", 1))

        src = os.path.join(info["dir"], info["images_dir"] or "", name)
        dst = os.path.join(out_dir, "images", split, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            continue

        label_lines = []
        for ann in by_image.get(img_id, []):
            x, y, bw, bh = ann["bbox"]
            cls = cat_to_idx.get(ann.get("category_id"), 0)
            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        stem = os.path.splitext(name)[0]
        with open(os.path.join(out_dir, "labels", split, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))
        written[split] += 1

    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: .\ntrain: images/train\nval: images/val\nnames:\n")
        for idx in sorted(classes):
            f.write(f"  {idx}: {classes[idx]}\n")

    return {
        "ok": True,
        "out": out_dir,
        "images_written": written,
        "val_documents": n_val,
        "classes": len(cat_to_idx),
        "data_yaml": yaml_path,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Project Cangjie — OBIMD → YOLO (split по документам)")
    ap.add_argument("--obimd-dir", required=True, help="Путь к распакованному OBIMD")
    ap.add_argument("--out", default="./data/obimd_yolo", help="Выходная папка YOLO")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Доля документов в val")
    args = ap.parse_args()

    result = convert_to_yolo(args.obimd_dir, args.out, val_frac=args.val_frac)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
