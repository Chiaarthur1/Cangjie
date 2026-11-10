#!/usr/bin/env bash
# ==============================================================================
# Project Cangjie — обучение детектора на GPU (RTX 4060 / станция).
# Запускать на машине с NVIDIA GPU (torch + CUDA).
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "=============================================================="
echo " Project Cangjie — ОБУЧЕНИЕ ДЕТЕКТОРА (GPU)"
echo "=============================================================="

# --- Проверка GPU ------------------------------------------------------------
echo ""
echo "[проверка] GPU для torch…"
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA не видна! Установи torch с CUDA (pip install torch --index-url https://download.pytorch.org/whl/cu121)'; print('GPU:', torch.cuda.get_device_name(0))"

# --- Датасет (если ещё не конвертирован) -------------------------------------
echo ""
echo "[1/3] Проверка YOLO-датасета…"
if [ ! -f data/obimd_yolo/data.yaml ]; then
  python3 cangjie_obimd_loader.py --obimd-dir data/OBIMD --out data/obimd_yolo --val-frac 0.1
else
  echo "  data/obimd_yolo/data.yaml уже есть."
fi

# --- Обучение ----------------------------------------------------------------
echo ""
echo "[2/3] Обучение RT-DETR (профиль turbo_t1_rtx4060, 50 эпох)…"
python3 cangjie_rtdetr_train.py \
  --profile turbo_t1_rtx4060 \
  --data data/obimd_yolo/data.yaml \
  --run-ultralytics --epochs 50

# --- Честный eval ------------------------------------------------------------
echo ""
echo "[3/3] Честная оценка после обучения…"
python3 cangjie_honest_eval.py --out eval_after_train.json || true

echo ""
echo "=============================================================="
echo " Обучение завершено. Веса: cangjie_rtdetr_runs/*/weights/best.pt"
echo " Активировать: во вкладке «Модели» или:"
echo "   python3 cangjie_config.py --set detector.weights <путь_к_best.pt>"
echo "=============================================================="
