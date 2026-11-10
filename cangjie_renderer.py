"""
Project Cangjie (Цанцзе) — Renderer Facade / Shim Module [v8.1]

ИСТОРИЧЕСКАЯ ПРОБЛЕМА:
    В исходной поставке этот файл был символической ссылкой на
    `/home/dyngtc/cangjie_project/cangjie_benchmark_forge.py` — абсолютный путь
    к машине конкретного разработчика. Внутри бандла (и на любой другой машине)
    ссылка была мертвой, и модуль `cangjie_renderer` не импортировался.

    Четыре модуля на него ссылались: cangjie_parser, cangjie_benchmark_factory,
    cangjie_rtdetr_infer, cangjie_yolo_factory. Все они падали в `except ImportError`
    и работали через прямой импорт из cangjie_benchmark_forge, но при любом
    изменении порядка импортов ветка падала.

РЕШЕНИЕ:
    Этот файл теперь — настоящий Python-модуль (не симлинк), который
    реэкспортирует публичный API генератора. Исторический смысл имени сохранён
    (это «точка входа» рендера), но зависимости от локального пути больше нет.

    Канонический источник истины — cangjie_benchmark_forge.
    Здесь нет дублирования данных: только переэкспорт.
"""

from __future__ import annotations

from cangjie_benchmark_forge import (  # noqa: F401
    KANGXI_214_COMPREHENSIVE_CATALOG,
    PHONETICS_COMPREHENSIVE_DB,
    EXISTING_COMBINATIONS,
    SpatialSlot,
    BoundingBox,
    SafeAllographResolver,
    GlyphRepository,
    RadicalLeaf,
    CompositionNode,
    CalligraphicPillowRenderer,
    SVGVectorRenderer,
)

__all__ = [
    "KANGXI_214_COMPREHENSIVE_CATALOG",
    "PHONETICS_COMPREHENSIVE_DB",
    "EXISTING_COMBINATIONS",
    "SpatialSlot",
    "BoundingBox",
    "SafeAllographResolver",
    "GlyphRepository",
    "RadicalLeaf",
    "CompositionNode",
    "CalligraphicPillowRenderer",
    "SVGVectorRenderer",
]
