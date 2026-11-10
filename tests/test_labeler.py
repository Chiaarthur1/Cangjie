"""Тесты разметки (экспорт в YOLO)."""

from cangjie_labeler import boxes_to_yolo_lines


def test_boxes_to_yolo_shape():
    boxes = [{"raw_glyph": "日", "class_id": 0, "bbox_norm": [0.0, 0.0, 0.5, 0.5]}]
    lines = boxes_to_yolo_lines(boxes)
    assert len(lines) == 1
    parts = lines[0].split()
    assert len(parts) == 5  # class cx cy w h
    cx, cy, w, h = (float(x) for x in parts[1:])
    assert abs(cx - 0.25) < 1e-6
    assert abs(cy - 0.25) < 1e-6
    assert abs(w - 0.5) < 1e-6
    assert abs(h - 0.5) < 1e-6


def test_boxes_class_from_glyph():
    # Класс должен маппиться по глифу, даже если class_id=0 (как шлёт UI).
    boxes = [{"raw_glyph": "日", "class_id": 0, "bbox_norm": [0.1, 0.1, 0.4, 0.4]}]
    line = boxes_to_yolo_lines(boxes)[0]
    cls = int(line.split()[0])
    assert cls > 0  # не fallback 0 для известного радикала


def test_boxes_unknown_glyph_fallback_zero():
    # Неизвестный глиф не пропускается, а экспортируется с fallback-классом 0
    # (и попадает в unmapped_glyphs у export_yolo).
    lines = boxes_to_yolo_lines([{"raw_glyph": "x", "bbox_norm": [0, 0, 1, 1]}])
    assert len(lines) == 1
    assert int(lines[0].split()[0]) == 0
