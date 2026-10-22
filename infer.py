"""CLI: classify a business process JSON with the public VAA QLoRA adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

from vaa_qlora.engine import DEFAULT_ADAPTER, DEFAULT_BASE_MODEL, VAAEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(ROOT / "examples" / "goods_delivery.json"),
        help="Process JSON with process_name + activities[]",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "last_result.json"),
        help="Where to write the classification JSON",
    )
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter_id", default=DEFAULT_ADAPTER)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    process_name = str(payload.get("process_name") or "").strip()
    activities = payload.get("activities") or []
    if not isinstance(activities, list) or not activities:
        raise SystemExit("Input JSON must contain a non-empty 'activities' list.")

    print(f"Base model : {args.base_model}")
    print(f"Adapter    : {args.adapter_id}")
    print(f"Process    : {process_name} ({len(activities)} activities)")
    print("Loading model (first run downloads weights)...")

    engine = VAAEngine(
        base_model=args.base_model,
        adapter_id=args.adapter_id,
        max_new_tokens=args.max_new_tokens,
    )
    result = engine.classify(process_name, activities)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    if not result["parsed"]:
        print("Could not parse JSON from the model. Raw output saved in the result file.")
    for row in result["activities"]:
        waste = f" | {row['waste_type']}" if row["waste_type"] else ""
        print(f"- {row['activity_name']}: {row['label']}{waste}")
        if row["justification"]:
            print(f"  {row['justification']}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
