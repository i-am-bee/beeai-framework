import json
from pathlib import Path

_dir = Path(__file__).parent

def load_items() -> list[dict]:
    with open(_dir / "dataset.json", encoding="utf-8") as f:
        return json.load(f)