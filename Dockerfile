# Project Cangjie — воспроизводимый контейнер (CPU-режим, без GPU).
# Сборка:  docker build -t cangjie .
# Запуск:  docker run -p 8080:8080 -v "$PWD/data":/app/data cangjie
FROM python:3.12-slim

WORKDIR /app

# Системные зависимости для opencv/PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Код + зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest

COPY . .

EXPOSE 8080

# По умолчанию запускаем веб-панель
CMD ["python3", "cangjie_ui_studio.py", "--serve", "--host", "0.0.0.0", "--port", "8080"]
