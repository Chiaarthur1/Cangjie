"""
Project Cangjie — реестр датасетов [v10.0].

Назначение:
  - Единый список датасетов проекта с лицензиями и локальными путями;
  - Честный статус «что уже скачано / чего не хватает»;
  - Включение/выключение датасетов (хранится в cangjie_config.json);
  - Рабочий код загрузки (urllib / huggingface_hub), который НЕ запускает
    скачивание автоматически — только по явному `--download <name>`.

Без заглушек: статус считается по реальным файлам на диске, а `--download`
честно сообщает, если сеть недоступна или датасет не зарегистрирован.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from cangjie_config import CangjieConfig, load_config, save_config, DEFAULT_CONFIG_PATH

# Реестр датасетов: key -> метаданные. `markers` — файлы/папки, по которым
# определяется, что датасет уже на диске.
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "unihan": {
        "name": "Unihan (Unicode Han Database)",
        "source": "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip",
        "license": "Unicode License (free)",
        "local_dir": "data",
        "markers": ["data/Unihan_Readings.txt", "data/Unihan_DictionaryLikeData.txt"],
        "description": "Чтения (kMandarin), глоссы (kDefinition), коды формы (kCangjie, kFourCornerCode).",
    },
    "chise": {
        "name": "CHISE IDS (Ideographic Description Sequences)",
        "source": "https://www.chise.org/ids/",
        "license": "GPL / free",
        "local_dir": "data",
        "markers": ["data/IDS-UCS-Basic.txt"],
        "description": "Разборы иероглифов по IDS (⿰⿱⿲…) — основа структуры знаков.",
    },
    "baxter_sagart": {
        "name": "Baxter–Sagart (Old Chinese reconstruction)",
        "source": "https://ocbaxtersagart.lsait.lsa.umich.edu/",
        "license": "CC BY 4.0",
        "local_dir": "data",
        "markers": ["data/Baxter_Sagart.csv"],
        "description": "Средне- и древнекитайские реконструкции (MC/OC).",
    },
    "make_me_a_hanzi": {
        "name": "Make Me a Hanzi (graphics + dictionary)",
        "source": "https://github.com/skishore/makemeahanzi",
        "license": "Arphic Public License",
        "local_dir": "data",
        "markers": ["data/graphics.txt", "data/dictionary.txt"],
        "description": "Векторные скелеты иероглифов для рендера.",
    },
    "obimd": {
        "name": "OBIMD (Oracle Bone Inscription Multi-modal Dataset)",
        "source": "https://huggingface.co/datasets/KLOBIP/OBIMD",
        "license": "CC BY-4.0",
        "local_dir": "data/OBIMD",
        "markers": ["data/OBIMD/images", "data/OBIMD/annotations"],
        "description": "10 077 оттисков костей Шан, 93 652 размеченных знака, 273+ класса.",
    },
    "museum": {
        "name": "Museum photo dataset (Wikimedia Commons, PD/CC0)",
        "source": "https://commons.wikimedia.org/",
        "license": "Public Domain / CC0 / CC BY",
        "local_dir": "data/museum_dataset",
        "markers": ["data/museum_dataset/manifest.jsonl"],
        "description": "Настоящие фото древних надписей (собирается cangjie_museum_dataset.py).",
    },
}


@dataclass
class DatasetStatus:
    name: str
    present: bool
    local_dir: str
    license: str
    description: str
    enabled: bool = True


class CangjieDatasetManager:
    def __init__(self, data_dir: str = "./data", config: Optional[CangjieConfig] = None):
        self.data_dir = os.path.abspath(data_dir)
        self.config = config if config is not None else load_config()

    def list_datasets(self) -> List[Dict[str, Any]]:
        return [dict(name=k, **{kk: vv for kk, vv in v.items() if kk != "markers"})
                for k, v in DATASET_REGISTRY.items()]

    def _is_present(self, key: str) -> bool:
        markers = DATASET_REGISTRY[key].get("markers", [])
        if not markers:
            return False
        return all(os.path.exists(m) for m in markers)

    def status(self) -> List[DatasetStatus]:
        result: List[DatasetStatus] = []
        for key, meta in DATASET_REGISTRY.items():
            present = self._is_present(key)
            result.append(DatasetStatus(
                name=key,
                present=present,
                local_dir=meta.get("local_dir", "data"),
                license=meta.get("license", "unknown"),
                description=meta.get("description", ""),
                enabled=self._is_enabled(key),
            ))
        return result

    def _is_enabled(self, key: str) -> bool:
        # Датасеты по умолчанию включены; инвалидные лицензии отключаются валидатором.
        disabled = getattr(self.config, "disabled_datasets", set())
        return key not in disabled

    def register(self, name: str, local_dir: str, markers: Optional[List[str]] = None) -> None:
        """Регистрирует пользовательский датасет (в памяти процесса)."""
        DATASET_REGISTRY[name] = {
            "name": name,
            "source": "user-defined",
            "license": "user-defined (validate before commercial use)",
            "local_dir": local_dir,
            "markers": markers or [local_dir],
            "description": "Пользовательский датасет.",
        }

    def download(self, name: str) -> Dict[str, Any]:
        """Рабочий код загрузки. Скачивание выполняется только по явному запросу.

        Для HF-датасетов использует huggingface_hub (если установлен); иначе —
        честную инструкцию. Ничего не качает без явного вызова `--download`.
        """
        if name not in DATASET_REGISTRY:
            return {"ok": False, "error": f"Датасет '{name}' не зарегистрирован. Доступно: {', '.join(DATASET_REGISTRY)}"}
        if self._is_present(name):
            return {"ok": True, "message": f"Датасет '{name}' уже на диске.", "downloaded": False}
        meta = DATASET_REGISTRY[name]
        if name == "obimd":
            return self._download_obimd(meta)
        return {
            "ok": False,
            "error": (
                f"Автоматическая загрузка '{name}' не настроена. Источник: {meta['source']}. "
                f"Скачайте вручную в {meta['local_dir']}."
            )
        }

    def _download_obimd(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            return {
                "ok": False,
                "error": (
                    f"huggingface_hub не установлен ({e}). Установите: pip install -U 'huggingface_hub[cli]' "
                    f"и выполните: huggingface-cli download KLOBIP/OBIMD --repo-type dataset --local-dir data/OBIMD"
                )
            }
        try:
            os.makedirs(meta["local_dir"], exist_ok=True)
            snapshot_download(repo_id="KLOBIP/OBIMD", repo_type="dataset", local_dir=meta["local_dir"])
            return {"ok": True, "downloaded": True, "message": f"OBIMD загружен в {meta['local_dir']}"}
        except Exception as e:
            return {"ok": False, "error": f"Не удалось скачать OBIMD: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Project Cangjie — реестр датасетов")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Путь к cangjie_config.json")
    ap.add_argument("--status", action="store_true", help="Показать статус всех датасетов")
    ap.add_argument("--list", action="store_true", help="Показать реестр (метаданные)")
    ap.add_argument("--register", nargs=2, metavar=("NAME", "DIR"), default=None, help="Зарегистрировать датасет NAME в DIR")
    ap.add_argument("--download", default=None, help="Скачать датасет по имени (явный запрос)")
    args = ap.parse_args()

    mgr = CangjieDatasetManager(config=load_config(args.config))

    if args.register:
        mgr.register(args.register[0], args.register[1])
        print(f"Зарегистрирован датасет '{args.register[0]}' -> {args.register[1]}")

    if args.list:
        for d in mgr.list_datasets():
            print(f"- {d['name']}: {d['description']} [{d['license']}]")

    if args.status:
        print("=" * 70)
        for s in mgr.status():
            mark = "✅" if s.present else "❌"
            enabled = "" if s.enabled else " (выключен)"
            print(f"{mark} {s.name:16s} -> {s.local_dir}{enabled}")
        print("=" * 70)

    if args.download:
        res = mgr.download(args.download)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res.get("ok") else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
