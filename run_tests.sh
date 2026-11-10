#!/usr/bin/env bash
# Project Cangjie — запуск тестов.
set -euo pipefail
cd "$(dirname "$0")"
echo "=== Project Cangjie: pytest ==="
if ! ./.venv/bin/python -c "import pytest" 2>/dev/null; then
  echo "Устанавливаю pytest…"
  ./.venv/bin/pip install -q pytest
fi
./.venv/bin/python -m pytest "$@"
