"""Gradio demo for Lean Value-Adding Analysis with the public QLoRA adapter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from vaa_qlora.engine import VAAEngine  # noqa: E402

EXAMPLE = json.loads((ROOT / "examples" / "goods_delivery.json").read_text(encoding="utf-8"))
ENGINE: VAAEngine | None = None


def get_engine() -> VAAEngine:
    global ENGINE
    if ENGINE is None:
        ENGINE = VAAEngine()
    return ENGINE


def run(process_name: str, activities_text: str):
    activities_text = (activities_text or "").strip()
    if not activities_text:
        raise gr.Error("Paste at least one activity.")

    try:
        parsed = json.loads(activities_text)
        if isinstance(parsed, dict) and "activities" in parsed:
            process_name = process_name or parsed.get("process_name") or ""
            activities = parsed["activities"]
        elif isinstance(parsed, list):
            activities = parsed
        else:
            raise ValueError("JSON must be a list of activities or an object with 'activities'.")
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Activities must be valid JSON. {exc}") from exc

    result = get_engine().classify(process_name.strip(), activities)
    rows = [
        [
            a.get("activity_name", ""),
            a.get("label", ""),
            a.get("waste_type", ""),
            a.get("justification", ""),
        ]
        for a in result["activities"]
    ]
    return rows, json.dumps(result, indent=2, ensure_ascii=False)


def build_ui() -> gr.Blocks:
    example_acts = json.dumps(EXAMPLE["activities"], indent=2, ensure_ascii=False)
    with gr.Blocks(title="VAA QLoRA — Lean Process Classifier") as demo:
        gr.Markdown(
            """
# Lean Value-Adding Analysis
**Mistral-7B-Instruct-v0.3 + QLoRA adapter**
[`ridoley/mistral7b-vaa-general-qlora`](https://huggingface.co/ridoley/mistral7b-vaa-general-qlora)

Paste a process flow. The model classifies each activity as **VA**, **BVA**, or **NVA**
and names a TIMWOODS waste type when the activity is waste.
"""
        )
        with gr.Row():
            process_name = gr.Textbox(label="Process name", value=EXAMPLE["process_name"])
        activities = gr.Code(
            label="Activities JSON",
            language="json",
            value=example_acts,
            lines=18,
        )
        run_btn = gr.Button("Classify process", variant="primary")
        table = gr.Dataframe(
            headers=["activity_name", "label", "waste_type", "justification"],
            label="Classification",
            wrap=True,
        )
        raw = gr.Code(label="Raw JSON result", language="json")
        run_btn.click(run, inputs=[process_name, activities], outputs=[table, raw])
    return demo


if __name__ == "__main__":
    build_ui().launch()
