"""
Project Cangjie — реестр моделей [v10.0].

Сканирует дисковые артефакты (детекторы .pt, индексы эмбеддингов, банки
шаблонов) и даёт горячее переключение активной модели через конфиг —
без перезапуска сервера (перезагрузка весов при следующем запросе).

Никаких заглушек: статус считается по реальным файлам и размерам.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cangjie_config import CangjieConfig, load_config, save_config, EMBEDDING_INDEX_PATH


@dataclass
class ModelEntry:
    kind: str          # detector | matcher | llm | template_bank | fullchar_bank
    id: str
    path: str
    size_mb: float
    status: str        # active | available | coco_base


# Файл-заглушка sanity-state — НЕ модель, в реестр не попадает.
_SANITY_SUFFIXES = ("_sanity_state.npz", "loss_sanity_state.npz")


def _scan_pt_files(roots: List[str]) -> List[str]:
    found: List[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        found.extend(glob.glob(os.path.join(root, "*.pt")))
        # рекурсивно на один уровень вложенности (runs/…/weights/best.pt)
        for sub in glob.glob(os.path.join(root, "*", "weights", "*.pt")):
            found.append(sub)
    # уникальные абсолютные пути
    seen = set()
    out = []
    for p in found:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append(ap)
    return sorted(out)


class CangjieModelRegistry:
    def __init__(self, data_dir: str = "./data", config: Optional[CangjieConfig] = None):
        self.data_dir = os.path.abspath(data_dir)
        self.config = config if config is not None else load_config()

    # --- сканирование --------------------------------------------------
    def scan(self) -> List[ModelEntry]:
        entries: List[ModelEntry] = []

        # Детекторы (.pt)
        for pt in _scan_pt_files([".", "cangjie_rtdetr_runs", "runs"]):
            name = os.path.basename(pt)
            size = round(os.path.getsize(pt) / (1024.0 * 1024.0), 2)
            active = os.path.abspath(self.config.detector.weights) == os.path.abspath(pt)
            status = "active" if active else ("coco_base" if name == "rtdetr-x.pt" else "available")
            entries.append(ModelEntry("detector", name, pt, size, status))

        # Эмбеддинг-индекс
        if os.path.exists(EMBEDDING_INDEX_PATH):
            size = round(os.path.getsize(EMBEDDING_INDEX_PATH) / (1024.0 * 1024.0), 4)
            entries.append(ModelEntry("matcher", "embedding_index", EMBEDDING_INDEX_PATH, size,
                                      "active" if self.config.matcher.backend == "embedding" else "available"))

        # Банки шаблонов компонентов (4 эры)
        for npz in sorted(glob.glob(os.path.join(self.data_dir, "cangjie_paleo_templates_*.npz"))):
            era = os.path.basename(npz).replace("cangjie_paleo_templates_", "").replace(".npz", "")
            size = round(os.path.getsize(npz) / (1024.0 * 1024.0), 2)
            entries.append(ModelEntry("template_bank", era, npz, size, "available"))

        # Банки полных знаков
        for npz in sorted(glob.glob(os.path.join(self.data_dir, "cangjie_fullchar_templates_*.npz"))):
            era = os.path.basename(npz).replace("cangjie_fullchar_templates_", "").replace(".npz", "")
            size = round(os.path.getsize(npz) / (1024.0 * 1024.0), 2)
            entries.append(ModelEntry("fullchar_bank", era, npz, size, "available"))

        # LLM (из конфига)
        llm = self.config.llm
        entries.append(ModelEntry(
            "llm", llm.backend, llm.url or "(не настроен)", 0.0,
            "active" if llm.backend != "none" else "available"
        ))
        return entries

    def active(self) -> Dict[str, Any]:
        return {
            "detector": {"weights": self.config.detector.weights, "backend": self.config.detector.backend},
            "matcher": {"backend": self.config.matcher.backend},
            "llm": {"backend": self.config.llm.backend, "model": self.config.llm.model, "url": self.config.llm.url},
        }

    # --- переключение --------------------------------------------------
    def switch(self, kind: str, model_id: str) -> Dict[str, Any]:
        if kind == "detector":
            path = model_id
            if not os.path.exists(path):
                return {"ok": False, "error": f"Файл весов не найден: {path}"}
            self.config.detector.weights = path
            self.config.detector.backend = "auto"
        elif kind == "matcher":
            if model_id == "embedding_index":
                if not os.path.exists(EMBEDDING_INDEX_PATH):
                    return {"ok": False, "error": "Индекс эмбеддингов не собран (runs/embeddings/index.npz отсутствует)"}
                self.config.matcher.backend = "embedding"
            elif model_id == "chamfer_hog":
                self.config.matcher.backend = "chamfer_hog"
            else:
                return {"ok": False, "error": f"Неизвестный матчер: {model_id}"}
        elif kind == "llm":
            # model_id = "none" | "ollama:<model>@<url>" | "vllm:<model>@<url>"
            if model_id == "none":
                self.config.llm.backend = "none"
                self.config.llm.model = ""
                self.config.llm.url = ""
            elif ":" in model_id:
                backend, rest = model_id.split(":", 1)
                if backend not in ("ollama", "vllm"):
                    return {"ok": False, "error": f"Неизвестный LLM-бэкенд: {backend}"}
                model, _, url = rest.partition("@")
                self.config.llm.backend = backend
                self.config.llm.model = model
                self.config.llm.url = url
            else:
                return {"ok": False, "error": f"Формат LLM: none | ollama:<model>@<url> | vllm:<model>@<url>"}
        else:
            return {"ok": False, "error": f"Неизвестный тип модели: {kind}"}

        save_config(self.config)
        return {"ok": True, "active": self.active()}

    def summary(self) -> Dict[str, Any]:
        return {
            "active": self.active(),
            "models": [e.__dict__ for e in self.scan()],
        }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Project Cangjie — реестр моделей")
    ap.add_argument("--list", action="store_true", help="Список моделей")
    ap.add_argument("--switch", nargs=2, metavar=("KIND", "ID"), default=None, help="Переключить: detector <path> | matcher <chamfer_hog|embedding_index> | llm <none|ollama:model@url>")
    args = ap.parse_args()
    reg = CangjieModelRegistry()
    if args.switch:
        print(json.dumps(reg.switch(args.switch[0], args.switch[1]), ensure_ascii=False, indent=2))
    if args.list:
        print(json.dumps(reg.summary(), ensure_ascii=False, indent=2))
