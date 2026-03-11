from __future__ import annotations

import json
from pathlib import Path

from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")
templates.env.filters["from_json"] = lambda s: json.loads(s) if s else {}
templates.env.filters["basename"] = lambda p: Path(p).name if p else ""

def _set_app_version(version: str) -> None:
    templates.env.globals["app_version"] = version
