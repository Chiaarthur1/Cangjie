#!/usr/bin/env bash
# Project Cangjie — установка патча v9.6
# Запуск:  bash install.sh /путь/к/cangjie_project
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-.}"

if [ ! -d "$TARGET" ]; then
  echo "❌ Каталог не найден: $TARGET"
  echo "Использование: bash install.sh /путь/к/cangjie_project"
  exit 1
fi

echo "=== Project Cangjie — установка патча v9.6 ==="
echo "Целевой каталог: $(cd "$TARGET" && pwd)"
echo ""

# 1) Проверка Python и зависимостей
PY="$(command -v python3 || true)"
if [ -z "$PY" ]; then echo "❌ python3 не найден"; exit 1; fi
echo "[1/4] Python: $($PY -V 2>&1)"

if ! $PY -c "import numpy, PIL" 2>/dev/null; then
  echo "      ⚠️  Не хватает numpy/pillow — ставлю..."
  $PY -m pip install --quiet numpy pillow || {
    echo "      ❌ Не удалось поставить. Поставь вручную:"
    echo "         pip install numpy pillow"
    exit 1
  }
fi
echo "      ✅ numpy + pillow на месте"

# 2) Бэкап заменяемых файлов
BACKUP="$TARGET/_backup_before_v96"
mkdir -p "$BACKUP"
echo "[2/4] Бэкап старых файлов -> $BACKUP"
for f in "$HERE"/code/*.py; do
  base="$(basename "$f")"
  if [ -f "$TARGET/$base" ]; then
    cp "$TARGET/$base" "$BACKUP/$base"
  fi
done
# старые банки — отдельным бэкапом (они несовместимы)
mkdir -p "$BACKUP/data_old"
for f in "$TARGET"/data/cangjie_paleo_templates_*.npz; do
  [ -e "$f" ] && cp "$f" "$BACKUP/data_old/" || true
done
echo "      ✅ Бэкап сделан"

# 3) Копирование кода и данных
echo "[3/4] Копирование..."
# В исходном архиве cangjie_renderer.py — висячий симлинк на абсолютный
# путь /home/dyngtc/... (машина автора). Его надо ЗАМЕНИТЬ файлом, иначе
# cp падает с "not writing through dangling symlink".
for f in "$HERE"/code/*.py; do
  base="$(basename "$f")"
  if [ -L "$TARGET/$base" ]; then
    rm -f "$TARGET/$base"
  fi
  cp -f "$f" "$TARGET/$base"
done
# Битые симлинки в data/ (Unihan_RadicalStrokes.txt -> Unihan_IRGSources.txt
# указывает на путь машины автора) — заменяем на настоящий файл.
for f in "$HERE"/data/*; do
  [ -f "$f" ] || continue
  base="$(basename "$f")"
  if [ -L "$TARGET/data/$base" ]; then
    rm -f "$TARGET/data/$base"
  fi
done
cp "$HERE"/data/cangjie_paleo_templates_*.npz "$TARGET/data/" 2>/dev/null || {
  mkdir -p "$TARGET/data"; cp "$HERE"/data/cangjie_paleo_templates_*.npz "$TARGET/data/"; }
cp "$HERE"/data/calibration_v96.json "$TARGET/data/"
cp "$HERE"/docs/*.md "$TARGET/" 2>/dev/null || true

# Готовый YOLO-датасет для обучения (если его ещё нет)
if [ -d "$HERE/dataset" ] && [ ! -f "$TARGET/cangjie_yolo_dataset/data.yaml" ]; then
  cp -r "$HERE/dataset" "$TARGET/cangjie_yolo_dataset"
  echo "      ✅ YOLO-датасет для обучения установлен (24 картинки + разметка)"
fi
echo "      ✅ Код и данные установлены"

# 4) Проверка
echo "[4/4] Проверка..."
cd "$TARGET"
$PY - <<'EOF'
import importlib, sys
mods = ['cangjie_paleo_matcher','cangjie_photo_decoder','cangjie_whole_glyph',
        'cangjie_fullchar_bank','cangjie_composer','cangjie_calibration',
        'cangjie_honest_eval','cangjie_verify_fullchar','cangjie_rebuild_banks',
        'cangjie_deducer','cangjie_dfine_losses']
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append((m, str(e)[:70]))
if bad:
    print("      ❌ Не импортируются:")
    for m, e in bad:
        print(f"         {m}: {e}")
    sys.exit(1)
print("      ✅ Все 11 модулей импортируются")
try:
    from cangjie_paleo_matcher import get_bank, ERA_KAISHU
    b = get_bank(ERA_KAISHU, data_dir='./data')
    print(f"      ✅ Банк KAISHU: {len(b.glyphs)} шаблонов, "
          f"калибровка={'да' if b._calib_model is not None else 'НЕТ'}")
except Exception as e:
    print(f"      ⚠️  Банк не поднялся: {e}")
EOF

cat <<'DONE'

============================================================
  ✅ УСТАНОВКА ЗАВЕРШЕНА
============================================================

ОБУЧЕНИЕ НА RTX 4060 (профиль turbo_t1_rtx4060):

  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install ultralytics

  # 30 минут тестового обучения:
  python3 cangjie_rtdetr_train.py \
      --profile turbo_t1_rtx4060 \
      --auto-synth-dataset \
      --run-ultralytics \
      --epochs 30

  Подробности: TRAIN_T1.md

ОСНОВНОЙ ВХОД — фото иероглифа:
  python3 cangjie_photo_decoder.py --photo photo.jpg --output ./out

ПРОВЕРКА КАЧЕСТВА:
  python3 cangjie_honest_eval.py --out ./eval.json

БАНК ПОЛНЫХ ЗНАКОВ (~130-180 МБ на эру, не входит в пакет):
  python3 cangjie_rebuild_banks.py --fullchar --all --data-dir ./data

Откат, если что-то не так:
  cp _backup_before_v96/*.py .
  cp _backup_before_v96/data_old/* data/
============================================================
DONE
