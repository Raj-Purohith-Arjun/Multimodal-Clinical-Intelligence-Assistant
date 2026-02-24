#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8000}"

python - <<'PY'
from pathlib import Path
from PIL import Image

Path("data/raw").mkdir(parents=True, exist_ok=True)
Image.new("RGB", (128, 128), color=(120, 30, 30)).save("data/raw/demo_xray.png")
print("Created data/raw/demo_xray.png")
PY

python -m uvicorn api.app:app --host 0.0.0.0 --port "$PORT"
