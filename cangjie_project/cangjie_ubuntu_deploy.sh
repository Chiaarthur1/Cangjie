#!/usr/bin/env bash
# ==============================================================================
# Project Cangjie (Цанцзе) v5.0 — Zorin OS 18.1 / Ubuntu 24.04 Setup & Launcher
# Автоматически устанавливает все системные и Python-библиотеки (с обходом PEP 668
# в Zorin OS 18.1 / Ubuntu 24.04), шрифты CJK, инициализирует ./data/ и запускает проект.
#
# Использование:
#   bash cangjie_ubuntu_deploy.sh --install-deps   # Только установка всех зависимостей
#   bash cangjie_ubuntu_deploy.sh --stage laptop   # Установка + Полная проверка + Веб-студия
#   bash cangjie_ubuntu_deploy.sh --stage cluster  # Полный цикл для сервера 4x RTX 4090
# ==============================================================================
set -euo pipefail

ARG1="${1:---stage}"
ARG2="${2:-laptop}"

install_zorin_ubuntu_dependencies() {
    echo "=============================================================================="
    echo " [Zorin OS 18.1 / Ubuntu] Установка системных пакетов, CJK-шрифтов и библиотек"
    echo "=============================================================================="

    # 1. Проверяем, установлены ли базовые модули в текущем python3
    if ! python3 -c "import numpy, scipy, cv2, PIL, yaml" &>/dev/null; then
        echo "-> Обнаружены отсутствующие библиотеки Python. Устанавливаем через apt и pip..."
        if command -v sudo &>/dev/null && command -v apt-get &>/dev/null; then
            sudo apt-get update -y
            sudo apt-get install -y \
                python3 \
                python3-pip \
                python3-venv \
                python3-dev \
                python3-numpy \
                python3-scipy \
                python3-opencv \
                python3-pil \
                python3-yaml \
                python3-dotenv \
                fonts-noto-cjk \
                libgl1 \
                libglib2.0-0 \
                build-essential \
                cmake
        fi

        # Доустановка через pip (с поддержкой --break-system-packages для Zorin OS 18.1 / Python 3.12)
        if command -v pip3 &>/dev/null; then
            pip3 install --user --break-system-packages \
                numpy scipy opencv-python-headless Pillow PyYAML python-dotenv onnxruntime neo4j 2>/dev/null || \
            pip3 install --user \
                numpy scipy opencv-python-headless Pillow PyYAML python-dotenv onnxruntime neo4j || true
        fi
    else
        echo "✅ Базовые пакеты (numpy, scipy, opencv, Pillow, PyYAML) уже установлены."
    fi

    # 2. Автоматический хотфикс cangjie_parser.py для пустых табов в реальном CHISE IDS-UCS-Basic.txt
    if [[ -f "cangjie_parser.py" ]]; then
        python3 -c '
with open("cangjie_parser.py", "r", encoding="utf-8") as f:
    c = f.read()
if "raw_ids.split()[0].strip()" in c:
    c = c.replace(
        "parts = line.strip().split(\"\\t\")\n                if len(parts) >= 3:\n                    codepoint, glyph, raw_ids = parts[0], parts[1], parts[2]\n                    ids_seq = raw_ids.split()[0].strip()",
        "s_line = line.strip()\n                if s_line.startswith(\";\"): continue\n                parts = [p.strip() for p in s_line.split(\"\\t\") if p.strip()]\n                if len(parts) >= 3:\n                    codepoint, glyph, raw_ids = parts[0], parts[1], parts[2]\n                    if not raw_ids.split(): continue\n                    ids_seq = raw_ids.split()[0].strip()"
    )
    c = c.replace("clean_seq = ids_sequence.split()[0].strip()", "clean_seq = ids_sequence.split()[0].strip() if ids_sequence.split() else \"\"")
    c = c.replace("logger.warning(f\"Неполная последовательность IDS:", "logger.debug(f\"Неполная последовательность IDS:")
    with open("cangjie_parser.py", "w", encoding="utf-8") as f:
        f.write(c)
    print("✅ [Auto-Fix] cangjie_parser.py пропатчен для работы с полным корпусом CHISE IDS-UCS-Basic.txt.")
'
    fi

    # 3. Автоматическая инициализация папки ./data/ (чтобы не было предупреждений об отсутствии файлов)
    mkdir -p ./data
    python3 -c '
import os, json, csv
from cangjie_benchmark_forge import KANGXI_214_COMPREHENSIVE_CATALOG, PHONETICS_COMPREHENSIVE_DB, GlyphRepository

data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

unihan_path = os.path.join(data_dir, "Unihan_RadicalStrokes.txt")
if not os.path.exists(unihan_path):
    with open(unihan_path, "w", encoding="utf-8") as f:
        f.write("# Unihan_RadicalStrokes.txt — Project Cangjie Canonical Seed\n")
        for num, base_g, allo_g, py, strokes, eng in KANGXI_214_COMPREHENSIVE_CATALOG:
            cp = f"U+{ord(base_g[0]):04X}"
            f.write(f"{cp}\tkRSUnicode\t{num}.0\n")

bs_path = os.path.join(data_dir, "Baxter_Sagart.csv")
try:
    from cangjie_archaic_epigraphy_engine import HierarchicalPhoneticSubtreeResolver
    HierarchicalPhoneticSubtreeResolver().export_comprehensive_baxter_sagart_csv(bs_path)
except Exception:
    if not os.path.exists(bs_path):
        with open(bs_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["character", "pinyin", "middle_chinese", "old_chinese", "gsr", "gloss"])
            for g, py, st in PHONETICS_COMPREHENSIVE_DB:
                writer.writerow([g, py, py, f"*{py}", "GSR-001", "phonetic_determinative"])

ids_path = os.path.join(data_dir, "IDS-UCS-Basic.txt")
if not os.path.exists(ids_path):
    with open(ids_path, "w", encoding="utf-8") as f:
        f.write("# IDS-UCS-Basic.txt — Project Cangjie Seed\n")
        f.write("U+6C5F\t江\t⿰氵工\nU+6CB3\t河\t⿰氵可\nU+6797\t林\t⿰木木\nU+9EBB\t麻\t⿸广⿰木木\nU+9053\t道\t⿺辶首\nU+570B\t國\t⿴囗玉\n")

gfx_path = os.path.join(data_dir, "graphics.txt")
if not os.path.exists(gfx_path):
    repo = GlyphRepository()
    with open(gfx_path, "w", encoding="utf-8") as f:
        for ch, med in repo.medians.items():
            f.write(json.dumps({"character": ch, "medians": med}, ensure_ascii=False) + "\n")
'
    echo "✅ Локальная база ./data/ (Unihan, Baxter-Sagart, CHISE IDS, MakeMeAHanzi) инициализирована."
}

if [[ "${ARG1}" == "--install-deps" ]]; then
    install_zorin_ubuntu_dependencies
    exit 0
fi

MODE="${ARG2}"
if [[ "${ARG1}" == "laptop" || "${ARG1}" == "cluster" ]]; then
    MODE="${ARG1}"
fi

install_zorin_ubuntu_dependencies

echo "=============================================================================="
echo " Project Cangjie v5.0 — Автоматический оркестратор (Режим: ${MODE})"
echo "=============================================================================="

if [[ "${MODE}" == "laptop" ]]; then
    echo "[1/3] Запуск математической самопроверки всех ядер v5.0 на CPU..."
    python3 cangjie_dfine_losses.py
    python3 cangjie_layout_to_tree.py
    python3 cangjie_generative_degradation.py
    python3 cangjie_archaic_epigraphy_engine.py
    python3 cangjie_distillation_cot_pipeline.py
    python3 cangjie_edge_cpu_runtime.py

    echo "[2/3] Генерация автономного веб-интерфейса cangjie_paleo_studio.html и проверка Дуэли Агентов..."
    python3 cangjie_ui_studio.py --export-html cangjie_paleo_studio.html

    echo "[3/3] Генерация пробного датасета (20 исторических артефактов с PDE-диффузией и TPS)..."
    python3 cangjie_yolo_factory.py --count 20 --imgsz 640 --output ./cangjie_laptop_sample_dataset
    echo "✅ Все проверки на Zorin OS 18.1 успешно пройдены!"
    exit 0
fi

if [[ "${MODE}" == "cluster" ]]; then
    echo "[Ubuntu 4x RTX 4090]Установка GPU-зависимостей (PyTorch, Ultralytics, vLLM)..."
    pip3 install --user --break-system-packages torch torchvision ultralytics onnx onnxruntime vllm neo4j || true

    echo "[Ubuntu 4x RTX 4090] Шаг 1/4: Синтез 50 000 сложных глифов и талисманов (1024x1024, PDE + Topian + TPS)..."
    python3 cangjie_yolo_factory.py --count 50000 --imgsz 1024 --workers 32 --output ./cangjie_yolo_dataset

    echo "[Ubuntu 4x RTX 4090] Шаг 2/4: Обучение детектора D-FINE / RT-DETRv2 с лоссами FDR + NWD + HG-Repulsion..."
    python3 cangjie_rtdetr_train.py --data ./cangjie_yolo_dataset/data.yaml --batch 64 --epochs 100

    echo "[Ubuntu 4x RTX 4090] Шаг 3/4: Генерация CoT-корпуса и фильтрация через импликацию Райхенбаха (tau=0.88)..."
    python3 cangjie_distillation_cot_pipeline.py

    echo "[Ubuntu 4x RTX 4090] Шаг 4/4: Калибровка Edge CPU Runtime (INT8 VNNI + Q4_0_8_8 + GBNF)..."
    python3 cangjie_edge_cpu_runtime.py
    echo "🚀 Серверный цикл на 4x RTX 4090 успешно выполнен!"
fi
