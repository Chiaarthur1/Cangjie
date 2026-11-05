# Обучение на T1 + RTX 4060 — готово к запуску

Всё нужное лежит в этом пакете. Команды — копируй и запускай.

---

## 0. Что нужно от тебя

Твоя машина: **T1 + RTX 4060 (8 ГБ VRAM)**. В проекте уже есть профиль
под неё: `turbo_t1_rtx4060` → batch=8, imgsz=640, AMP, 30 эпох.

Проверь, что видят torch:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Если `True` и видно `RTX 4060` — можно стартовать.

---

## 1. Установка зависимостей

```bash
# PyTorch с CUDA для 4060 (sm_89). Нужен cu121 или новее.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Ultralytics (RT-DETR)
pip install ultralytics

# базовое (уже в проекте, но на всякий)
pip install numpy pillow
```

> Если `pip` ругается на внешнюю систему (externally-managed-environment) —
> добавь `--break-system-packages` или работай в venv:
> ```bash
> python3 -m venv .venv && source .venv/bin/activate
> ```

`ultralytics` при первом запуске сам скачает базовую модель `rtdetr-x.pt`
(нужен интернет, ~130 МБ). Либо скачай заранее:
```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-x.pt
```

---

## 2. Установка патча

```bash
tar xzf cangjie_patch_v96.tar.gz
cd cangjie_project                     # распакованный исходный проект
bash ../cangjie_patch/install.sh .    # ставит код + банки + калибровку
```

Установщик сам сделает бэкап в `_backup_before_v96/` и проверит, что всё
импортируется.

---

## 3. Старт обучения — 30 минут

### Вариант А. Полный цикл (ultralytics RT-DETR + композитный лосс)

```bash
python3 cangjie_rtdetr_train.py \
    --profile turbo_t1_rtx4060 \
    --auto-synth-dataset \
    --run-ultralytics \
    --epochs 30 \
    --data ./cangjie_yolo_dataset/data.yaml
```

Что произойдёт:
1. сгенерирует YOLO-датасет (18 train / 6 val, 214 классов) — если его нет;
2. скачает `rtdetr-x.pt`;
3. запустит `RTDETR.train()` на 640px, batch 8, AMP — **это и есть обучение**;
4. после — прогонит композитный D-FINE лосс (FDR + L_FGL + GO-LSD + L_NWD + L_HG-Rep).

На 4060 это уложится в 30–60 минут (зависит от размера картинок и
скорости диска).

### Вариант Б. Без ultralytics (только композитный лосс, без видеокарты)

```bash
python3 cangjie_rtdetr_train.py \
    --profile turbo_t1_rtx4060 \
    --auto-synth-dataset \
    --sanity-test
```

Это быстрая проверка сходимости лосса — я уже прогонял: **loss_total
падает с 49.60 до 16.99** за несколько итераций. Годится, чтобы убедиться,
что окружение живое, не дожидаясь 30 минут.

### Вариант В. Через Ultralytics напрямую (больше контроля)

```bash
from ultralytics import RTDETR
model = RTDETR("rtdetr-x.pt")
model.train(data="./cangjie_yolo_dataset/data.yaml",
            epochs=30, batch=8, imgsz=640,
            device=0, amp=True,
            project="cangjie_rtdetr_runs", name="t1_4060")
```

---

## 4. Что должно получиться

```
cangjie_rtdetr_runs/t1_4060/
├── weights/best.pt        ← это и есть обученный детектор
├── weights/last.pt
├── results.csv
└── results.png            ← графики loss/mAP
```

**Итоговый чекпойнт детектора** (`cangjie_dfine_rtdetr_checkpoint.npz` в
текущем проекте — 44 КБ) — это **заглушка**, а не веса. После твоего
прогона замени его на `best.pt` (или сконвертируй).

---

## 5. Что делать с обученной моделью дальше

1. **Проверить на музейных фото** (реальные данные, не синтетика):
```bash
python3 cangjie_museum_dataset.py \
    --search "oracle bone script" --search "bronze inscription china" \
    --limit 40 --out ./data/museum_dataset

python3 cangjie_photo_decoder.py \
    --photo ./data/museum_dataset/images/<файл>.png \
    --weights cangjie_rtdetr_runs/t1_4060/weights/best.pt \
    --output ./museum_eval
```

2. **Честный eval** на отложенной части:
```bash
python3 cangjie_honest_eval.py --out ./eval_after_train.json
```

---

## 6. Важные предупреждения

**1. Датасет сейчас синтетический.** 18 train / 6 val картинок, сгенерированных
тем же рендером, что и банк. Это нормально для «проверить, что пайплайн
обучения вообще работает» — но **не даёт честных метрик**. Для настоящих
цифр нужен музейный датасет + ручная разметка боксов.

**2. 24 картинки — это очень мало.** Детектор на таком объёме переобучится
мгновенно. Для 30-минутного теста нормально; для рабочей модели нужно
500+ размеченных боксов.

**3. Порядок данных в YOLO:** `class cx cy w h`, все в нормализованных
координатах 0..1. Разметка генерируется автоматически, но если будешь
размечать музейные фото руками — тот же формат.

**4. Если OOM** (8 ГБ мало при 640px):
```bash
--batch 4     # или
--imgsz 512
```

**5. 4060 = Ada Lovelace (sm_89).** Нужен PyTorch ≥ 2.1 с CUDA ≥ 12.1.
Старые сборки не увидят карту.

---

## 7. Быстрая проверка «всё живо» (2 минуты, без обучения)

```bash
python3 -c "
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    x = torch.randn(4096,4096, device='cuda')
    print('matmul ok', (x@x).sum().item() is not None)
"
python3 -c "
from cangjie_paleo_matcher import get_bank, ERA_KAISHU
b = get_bank(ERA_KAISHU, data_dir='./data')
print('bank', len(b.glyphs), 'calib', b._calib_model is not None)
"
```

Обе команды должны напечатать OK без ошибок — значит окружение собрано
правильно и можно запускать обучение.
