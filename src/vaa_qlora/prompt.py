from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = ROOT / "prompts" / "system_prompt.txt"


def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_user_prompt(process_name: str, activities: list[dict]) -> str:
    total = len(activities)
    lines = [
        f"Process Name        : {process_name or '(unnamed)'}",
        (
            f"Process Objective   : Successfully complete '{process_name}' "
            "while maximising value and minimising waste."
            if process_name
            else "Process Objective   : Complete the end-to-end business process efficiently."
        ),
        f"Total activities    : {total}",
        "",
        "Process Flow:",
    ]

    for i, row in enumerate(activities, start=1):
        act_name = str(row.get("activity_name") or row.get("name") or "").strip() or "Unknown activity"
        desc = str(row.get("description") or row.get("description_activity") or "").strip()
        prev_name = (
            str(activities[i - 2].get("activity_name") or activities[i - 2].get("name") or "").strip()
            if i > 1
            else None
        )
        next_name = (
            str(activities[i].get("activity_name") or activities[i].get("name") or "").strip()
            if i < total
            else None
        )
        parts = []
        if prev_name:
            parts.append(f"preceded by '{prev_name}'")
        if next_name:
            parts.append(f"followed by '{next_name}'")
        neighbour = ", ".join(parts) if parts else "no adjacent activities"

        lines.append(f"\n{i}. Activity Name: {act_name}")
        lines.append(f"   Sequence context: {neighbour}")
        if desc:
            lines.append(f"   Description: {desc}")

    lines.append("\nClassify ALL the activities listed above using the full process context.")
    lines.append('Return a valid JSON object matching the requested schema (a dictionary with a list under the key "activities").')
    lines.append("Do not return any other text before or after the JSON.")
    return "\n".join(lines)
