"""Тесты OBIMD → YOLO конвертера."""

import json

from PIL import Image

from cangjie_obimd_loader import convert_to_yolo, detect_obimd


def test_detect_missing(tmp_path):
    info = detect_obimd(str(tmp_path / "nope"))
    assert info["present"] is False


def test_convert_synthetic(tmp_path):
    base = tmp_path / "obimd"
    (base / "images").mkdir(parents=True)
    for i in range(4):
        Image.new("L", (64, 64), 255).save(base / "images" / f"doc{i // 2}_{i}.png")
    coco = {
        "images": [{"id": i, "file_name": f"doc{i // 2}_{i}.png", "width": 64, "height": 64} for i in range(4)],
        "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
        "annotations": [
            {"image_id": 0, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20]},
            {"image_id": 2, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"image_id": 3, "category_id": 2, "bbox": [10, 10, 20, 20]},
        ],
    }
    with open(base / "annotations.json", "w") as f:
        json.dump(coco, f)

    out = tmp_path / "out"
    r = convert_to_yolo(str(base), str(out), val_frac=0.5)
    assert r["ok"]
    assert r["images_written"]["train"] >= 1
    assert (out / "data.yaml").exists()
