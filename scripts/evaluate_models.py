"""Compare models on validation data, select one, then evaluate it once on the held-out test set.

Usage (PowerShell):
    # Step 1: rank all trained models by VALIDATION macro-F1 and record the selection.
    #         --activate makes it the default model of the inference service / MCP server.
    python scripts/evaluate_models.py compare --data data/processed/amazon_electronics --activate

    # Step 2: evaluate the selected model on the TEST split (only possible after step 1).
    #         --all also reports the other models, labelled "post-selection".
    python scripts/evaluate_models.py test --data data/processed/amazon_electronics --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from review_intel.evaluation.selection import evaluate_on_test, select_model  # noqa: E402
from review_intel.models.artifacts import SELECTED_POINTER  # noqa: E402
from review_intel.utils import ARTIFACTS_DIR, REPORTS_DIR, read_json, relative_to_root, setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["compare", "test"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--activate", action="store_true",
                        help=f"compare: also write {relative_to_root(SELECTED_POINTER)} (used by the MCP server)")
    parser.add_argument("--all", action="store_true", help="test: also evaluate non-selected models")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--reports-dir", type=Path)
    args = parser.parse_args()
    setup_logging()

    name = read_json(args.data / "manifest.json")["name"]
    models_dir = args.models_dir or ARTIFACTS_DIR / "models" / name
    reports_dir = args.reports_dir or REPORTS_DIR / name

    if args.command == "compare":
        selection = select_model(reports_dir, models_dir, SELECTED_POINTER if args.activate else None)
        print(json.dumps({k: selection[k] for k in ("selected_family", "validation_macro_f1", "ranking")}, indent=2))
        print(f"Report: {relative_to_root(reports_dir / 'validation_comparison.md')}", file=sys.stderr)
    else:
        report = evaluate_on_test(args.data, reports_dir, models_dir, include_non_selected=args.all)
        print(json.dumps({family: {"role": r["role"], "test_macro_f1": r["test"]["macro_f1"],
                                   "test_accuracy": r["test"]["accuracy"]}
                          for family, r in report["results"].items()}, indent=2))
        print(f"Report: {relative_to_root(reports_dir / 'test_report.md')}", file=sys.stderr)


if __name__ == "__main__":
    main()
