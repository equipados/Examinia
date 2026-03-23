from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from loguru import logger


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(level: str) -> None:
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )


def safe_filename(value: str, fallback: str = "sin_nombre") -> str:
    cleaned = re.sub(r"[^\w\s-]", "", value or "", flags=re.UNICODE).strip().lower()
    cleaned = re.sub(r"[-\s]+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def deduplicate_name(base_name: str, seen: dict[str, int]) -> str:
    count = seen.get(base_name, 0)
    seen[base_name] = count + 1
    if count == 0:
        return base_name
    return f"{base_name}_{count + 1}"


def normalize_identifier(value: str | None, default: str = "") -> str:
    if value is None:
        return default
    normalized = str(value).strip().lower().replace(" ", "")
    normalized = normalized.replace(")", "").replace("(", "")
    return normalized or default


def part_column_id(question_id: str, part_id: str) -> str:
    qid = normalize_identifier(question_id, "0")
    pid = normalize_identifier(part_id, "")
    if not pid or pid == "single":
        return qid
    return f"{qid}.{pid}"


def parse_point_string(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    raw = str(value).strip().lower()
    if not raw:
        return None
    raw = raw.replace("ptos", "").replace("puntos", "").replace("valor", "")
    raw = raw.replace(",", ".")
    match = re.search(r"(-?\d+(?:\.\d+)?)", raw)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def format_page_range(pages: Iterable[int]) -> str:
    unique_pages = sorted(set(int(p) for p in pages))
    if not unique_pages:
        return "-"
    if len(unique_pages) == 1:
        return str(unique_pages[0])
    ranges: list[str] = []
    start = prev = unique_pages[0]
    for page in unique_pages[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = page
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def round_points(value: float) -> float:
    return round(float(value), 2)
