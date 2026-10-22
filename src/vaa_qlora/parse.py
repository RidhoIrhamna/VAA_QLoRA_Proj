from __future__ import annotations

import json
import re

LABELS = {"VA", "BVA", "NVA"}
VALID_WASTE_TYPES = [
    "Defects",
    "Overproduction",
    "Waiting",
    "Unused talent",
    "Transportation",
    "Inventory",
    "Motion",
    "Extra-processing",
]
_WASTE_ALIAS_MAP = {
    "overprocessing": "Extra-processing",
    "extra work": "Extra-processing",
    "extra-processing": "Extra-processing",
    "extra processing": "Extra-processing",
    "waiting": "Waiting",
    "defects": "Defects",
    "transportation": "Transportation",
    "motion": "Motion",
    "overproduction": "Overproduction",
    "over production": "Overproduction",
    "unused talent": "Unused talent",
    "inventory": "Inventory",
}


def strip_code_fences(text: str) -> str:
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return cleaned.strip()


def extract_label(text: str) -> str:
    match = re.search(r"\b(VA|BVA|NVA)\b", str(text).upper())
    return match.group(1) if match else "UNKNOWN"


def canonicalize_waste_type(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    if normalized in _WASTE_ALIAS_MAP:
        return _WASTE_ALIAS_MAP[normalized]
    for wt in VALID_WASTE_TYPES:
        if normalized == wt.lower():
            return wt
    return text


def _normalize_activity(raw: dict) -> dict:
    raw_cls = str(
        raw.get("label")
        or raw.get("classification")
        or raw.get("category")
        or raw.get("class")
        or ""
    ).strip()
    label = extract_label(raw_cls) if raw_cls else "UNKNOWN"
    justification = str(
        raw.get("justification")
        or raw.get("reasoning")
        or raw.get("explanation")
        or ""
    ).strip()
    waste = canonicalize_waste_type(raw.get("waste_type") or raw.get("waste") or "")
    if label in {"VA", "BVA"}:
        waste = ""
    return {
        "activity_name": str(raw.get("activity_name") or raw.get("name") or "").strip(),
        "label": label,
        "waste_type": waste,
        "justification": justification,
    }


def parse_process_output(raw_text: str, process_name: str = "") -> dict:
    cleaned = strip_code_fences(raw_text)
    obj = None
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                obj = None

    activities: list[dict] = []
    name = process_name
    if isinstance(obj, dict):
        name = str(obj.get("process_name") or process_name).strip()
        pred_list = (
            obj.get("activities")
            or obj.get("activities_list")
            or obj.get("steps")
            or []
        )
        if isinstance(pred_list, list):
            activities = [_normalize_activity(x) for x in pred_list if isinstance(x, dict)]
    elif isinstance(obj, list):
        activities = [_normalize_activity(x) for x in obj if isinstance(x, dict)]

    return {
        "process_name": name,
        "activities": activities,
        "raw_output": raw_text,
        "parsed": bool(activities),
    }
