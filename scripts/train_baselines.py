"""Train Model A (TF-IDF + LogReg) and Model B (TF-IDF + LinearSVM) with validation-based tuning.

Usage (PowerShell):
    python scripts/train_baselines.py --data data/processed/amazon_electronics
    python scripts/train_baselines.py --data data/processed/synthetic_fixture --config configs/models/baselines_fixture.json

Saves artifacts/models/<family>/ and reports/<dataset>/validation/<family>.json.
The test split is NOT read. Next step: scripts/evaluate_models.py compare.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from review_intel.models.training import train_baselines  # noqa: E402
from review_intel.utils import ARTIFACTS_DIR, CONFIGS_DIR, REPORTS_DIR, read_json, setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Processed dataset dir (has train.csv, val.csv)")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "models" / "baselines.json")
    parser.add_argument("--family", action="append", help="Train only this family (repeatable)")
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    args = parser.parse_args()
    setup_logging()

    dataset_name = read_json(args.data / "manifest.json")["name"]
    models_dir = args.models_dir or ARTIFACTS_DIR / "models" / dataset_name
    reports_dir = args.reports_dir or REPORTS_DIR / dataset_name
    results = train_baselines(args.data, read_json(args.config), models_dir, reports_dir, args.family)
    print(json.dumps({family: {"val_macro_f1": r["validation"]["macro_f1"], "val_accuracy": r["validation"]["accuracy"],
                               "best_params": r["best_params"], "training_time_seconds": r["training_time_seconds"],
                               "model_dir": r["model_dir"]}
                      for family, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
