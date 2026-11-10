"""
Project Cangjie — единый конфиг [v10.0].

Назначение:
  - Профили железа (laptop / turbo_t1_rtx4060 / station_4x4090);
  - Выбор модели на каждом уровне пайплайна:
      * detector.backend: auto | rtdetr | yolo | pixel | none
      * matcher.backend : chamfer_hog | embedding
      * llm.backend     : none | ollama | vllm  (+ model + url)
  - Персистентность: обычный JSON, создаётся автоматически при первом запуске.

Никаких заглушек: все значения реально читаются и валидируются. Если модель
недоступна (нет весов / нет модуля / нет GPU), система честно переключается на
доступный путь, а не подставляет фейк.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG_PATH = "./cangjie_config.json"

# Путь к обученному kNN-индексу эмбеддингов (собирается cangjie_metric_embed.py index).
EMBEDDING_INDEX_PATH = "runs/embeddings/index.npz"

DETECTOR_BACKENDS = ("auto", "rtdetr", "yolo", "pixel", "none")
MATCHER_BACKENDS = ("chamfer_hog", "embedding")
LLM_BACKENDS = ("none", "ollama", "vllm")

# Профили железа: значения по умолчанию для обучения/инференса.
MACHINE_PROFILES: Dict[str, Dict[str, Any]] = {
    "laptop": {
        "device": "cpu", "batch": 2, "epochs": 10, "imgsz": 512,
        "amp": False, "bf16": False, "description": "Ноутбук, только CPU",
    },
    "turbo_t1_rtx4060": {
        "device": "0", "batch": 8, "epochs": 30, "imgsz": 640,
        "amp": True, "bf16": False, "description": "1×RTX 4060 (8 ГБ VRAM)",
    },
    "station_4x4090": {
        "device": "0,1,2,3", "batch": 16, "epochs": 50, "imgsz": 640,
        "amp": True, "bf16": True, "description": "4×RTX 4090 (96 ГБ VRAM)",
    },
}


@dataclass
class DetectorConfig:
    backend: str = "auto"      # auto | rtdetr | yolo | pixel | none
    weights: str = "rtdetr-x.pt"


@dataclass
class MatcherConfig:
    backend: str = "chamfer_hog"  # chamfer_hog | embedding


@dataclass
class LLMConfig:
    backend: str = "none"       # none | ollama | vllm
    model: str = ""
    url: str = ""


@dataclass
class CangjieConfig:
    machine: str = "auto"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    api_key: str = ""  # пусто = аутентификация выключена

    # --- сериализация -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine": self.machine,
            "detector": asdict(self.detector),
            "matcher": asdict(self.matcher),
            "llm": asdict(self.llm),
            "api_key": self.api_key,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CangjieConfig":
        det = d.get("detector", {}) or {}
        mat = d.get("matcher", {}) or {}
        llm = d.get("llm", {}) or {}
        return cls(
            machine=str(d.get("machine", "auto")),
            detector=DetectorConfig(
                backend=str(det.get("backend", "auto")),
                weights=str(det.get("weights", "rtdetr-x.pt")),
            ),
            matcher=MatcherConfig(
                backend=str(mat.get("backend", "chamfer_hog")),
            ),
            llm=LLMConfig(
                backend=str(llm.get("backend", "none")),
                model=str(llm.get("model", "")),
                url=str(llm.get("url", "")),
            ),
            api_key=str(d.get("api_key", "")),
        )

    # --- машина --------------------------------------------------------
    def resolve_machine(self) -> str:
        """Возвращает имя профиля железа. 'auto' определяется по CUDA."""
        if self.machine in MACHINE_PROFILES:
            return self.machine
        return detect_machine()

    def machine_profile(self) -> Dict[str, Any]:
        return MACHINE_PROFILES.get(self.resolve_machine(), MACHINE_PROFILES["laptop"])

    # --- выбор модели --------------------------------------------------
    def resolve_detector_backend(self) -> str:
        """Финальный бэкенд детектора с учётом 'auto' и наличия весов/GPU."""
        backend = self.detector.backend
        if backend != "auto":
            return backend
        # auto: нейросетевой детектор только если веса существуют и не COCO.
        if not os.path.exists(self.detector.weights):
            return "pixel"
        if _is_coco_weights(self.detector.weights):
            return "pixel"
        if backend_uses_gpu(backend) and not cuda_available():
            return "pixel"
        return "rtdetr"

    def resolve_matcher_backend(self) -> str:
        """embeddings активны только при наличии ОБУЧЕННОГО индекса.

        Модуль metric_embed может существовать, но без собранного kNN-индекса
        (runs/embeddings/index.npz) выбор 'embedding' честно падает на chamfer_hog.
        """
        if self.matcher.backend == "embedding":
            try:
                import cangjie_metric_embed  # noqa: F401
                if os.path.exists(EMBEDDING_INDEX_PATH):
                    return "embedding"
            except Exception:
                pass
        return "chamfer_hog"

    # --- валидация -----------------------------------------------------
    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.machine not in MACHINE_PROFILES and self.machine != "auto":
            errors.append(f"Неизвестный профиль железа: {self.machine} (доступны: auto, {', '.join(MACHINE_PROFILES)})")
        if self.detector.backend not in DETECTOR_BACKENDS:
            errors.append(f"detector.backend='{self.detector.backend}' — допустимо: {', '.join(DETECTOR_BACKENDS)}")
        if self.matcher.backend not in MATCHER_BACKENDS:
            errors.append(f"matcher.backend='{self.matcher.backend}' — допустимо: {', '.join(MATCHER_BACKENDS)}")
        if self.llm.backend not in LLM_BACKENDS:
            errors.append(f"llm.backend='{self.llm.backend}' — допустимо: {', '.join(LLM_BACKENDS)}")
        if self.llm.backend in ("ollama", "vllm") and not self.llm.url:
            errors.append(f"llm.backend='{self.llm.backend}' требует llm.url")
        return errors


# --- хелперы -----------------------------------------------------------

def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def backend_uses_gpu(backend: str) -> bool:
    return backend in ("rtdetr", "yolo")


def _is_coco_weights(weights_path: str) -> bool:
    """Возвращает True, если файл весов — базовая COCO-модель, а не палео-детектор."""
    try:
        from ultralytics import RTDETR
        model = RTDETR(weights_path)
        names = getattr(model, "names", {}) or {}
        first = str(names.get(0, "")).lower()
        return not (len(names) >= 200 and first not in {"person", "bicycle", "car"})
    except Exception:
        return True


def detect_machine() -> str:
    """Определяет профиль по доступному железу (без догадок)."""
    try:
        import torch
        n = int(torch.cuda.device_count())
        if n >= 4:
            return "station_4x4090"
        if n >= 1:
            return "turbo_t1_rtx4060"
    except Exception:
        pass
    return "laptop"


def load_config(path: str = DEFAULT_CONFIG_PATH) -> CangjieConfig:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return CangjieConfig.from_dict(json.load(f))
        except Exception:
            pass
    return CangjieConfig()


def save_config(config: CangjieConfig, path: str = DEFAULT_CONFIG_PATH) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
    return path


def _set_dotted(config: CangjieConfig, dotted: str, value: str) -> None:
    """Применяет --set detector.epochs=50 / detector.backend=pixel и т.п."""
    parts = dotted.split(".")
    if parts[0] == "machine":
        config.machine = value
        return
    if parts[0] == "detector":
        if parts[1] == "backend":
            config.detector.backend = value
        elif parts[1] == "weights":
            config.detector.weights = value
        return
    if parts[0] == "matcher":
        if parts[1] == "backend":
            config.matcher.backend = value
        return
    if parts[0] == "llm":
        if parts[1] == "backend":
            config.llm.backend = value
        elif parts[1] == "model":
            config.llm.model = value
        elif parts[1] == "url":
            config.llm.url = value
        return
    raise ValueError(f"Неизвестный ключ конфига: {dotted}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Project Cangjie — единый конфиг (профили железа + выбор модели)")
    ap.add_argument("--path", default=DEFAULT_CONFIG_PATH, help="Путь к JSON-конфигу")
    ap.add_argument("--show", action="store_true", help="Показать весь конфиг")
    ap.add_argument("--machine", default=None, help="Установить профиль железа")
    ap.add_argument("--detect-machine", action="store_true", help="Определить железо")
    ap.add_argument("--set", action="append", default=[], help="Установить значение: detector.backend=pixel")
    ap.add_argument("--validate", action="store_true", help="Проверить конфиг")
    args = ap.parse_args()

    config = load_config(args.path)

    if args.machine:
        config.machine = args.machine
    for dotted in args.set:
        key, _, value = dotted.partition("=")
        _set_dotted(config, key, value)

    if args.validate:
        errors = config.validate()
        if errors:
            for e in errors:
                print(f"ERROR: {e}")
            return 1
        print("OK: конфиг валиден")

    if args.detect_machine:
        print(f"machine = {detect_machine()} | cuda = {cuda_available()}")

    save_config(config, args.path)

    if args.show:
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
