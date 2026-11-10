#!/usr/bin/env bash
# Project Cangjie — загрузка недостающих датасетов.
# Запуск: bash download_datasets.sh [obimd|museum|all]
set -euo pipefail

TARGET="${1:-all}"
DATA_DIR="${DATA_DIR:-./data}"

echo "=== Project Cangjie: загрузка датасетов ==="
echo "Целевой каталог: $DATA_DIR | режим: $TARGET"

# Уже на диске (проверка)
have() { [ -e "$1" ] && echo "  ✅ уже есть: $1" || echo "  ❌ нет: $1"; }
echo "--- Статус локальных данных ---"
have "$DATA_DIR/Unihan_Readings.txt"
have "$DATA_DIR/IDS-UCS-Basic.txt"
have "$DATA_DIR/Baxter_Sagart.csv"
have "$DATA_DIR/graphics.txt"
have "$DATA_DIR/dictionary.txt"
have "$DATA_DIR/OBIMD"
have "$DATA_DIR/museum_dataset"

download_obimd() {
  echo ""
  echo "--- OBIMD (Oracle Bone Inscription Multi-modal Dataset, CC BY-4.0) ---"
  if [ -d "$DATA_DIR/OBIMD/images" ]; then
    echo "  OBIMD уже скачан."
  else
    if command -v huggingface-cli >/dev/null 2>&1; then
      huggingface-cli download KLOBIP/OBIMD --repo-type dataset --local-dir "$DATA_DIR/OBIMD"
    else
      echo "  ⚠️ huggingface-cli не найден. Установи: pip install -U 'huggingface_hub[cli]'"
      echo "  Затем: huggingface-cli download KLOBIP/OBIMD --repo-type dataset --local-dir $DATA_DIR/OBIMD"
    fi
  fi
}

download_museum() {
  echo ""
  echo "--- Музейные фото (Wikimedia Commons, PD/CC0/CC BY) ---"
  python3 cangjie_museum_dataset.py \
    --search "oracle bone script" \
    --search "bronze inscription china" \
    --search "seal script chinese" \
    --limit 60 --out "$DATA_DIR/museum_dataset"
}

case "$TARGET" in
  obimd)  download_obimd ;;
  museum) download_museum ;;
  all)    download_obimd; download_museum ;;
  *)      echo "Неизвестный режим: $TARGET (доступно: obimd | museum | all)"; exit 1 ;;
esac

echo ""
echo "=== Готово. После загрузки OBIMD сконвертируйте в YOLO: ==="
echo "  python3 cangjie_obimd_loader.py --obimd-dir $DATA_DIR/OBIMD --out $DATA_DIR/obimd_yolo --val-frac 0.1"
