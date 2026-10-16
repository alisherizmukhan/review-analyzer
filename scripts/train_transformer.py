"""Optional: fine-tune DistilBERT (Model C). Requires `pip install -e .[transformer]` (see README for torch).

Usage (PowerShell):
    python scripts/train_transformer.py --data data/processed/amazon_electronics
    # CPU-friendly smoke run on a subsample:
    python scripts/train_transformer.py --data data/processed/amazon_electronics `
        --device cpu --max-train-samples 2000 --epochs 1 --max-length 128

Downloads the base model (~270 MB) from Hugging Face on first use. Writes
artifacts/models/<dataset>/distilbert/ and reports/<dataset>/validation/distilbert.json;
then rerun `scripts/evaluate_models.py compare` to include it in model selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from review_intel.utils import ARTIFACTS_DIR, CONFIGS_DIR, REPORTS_DIR, read_json, setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "models" / "distilbert.json")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"])
    args = parser.parse_args()
    setup_logging()

    try:
        from review_intel.models.transformer import TransformerTrainingConfig, train_transformer
    except ImportError as exc:
        print(f"error: transformer dependencies missing ({exc}). Install with: pip install -e .[transformer]",
              file=sys.stderr)
        return 2

    values = read_json(args.config)
    for f in fields(TransformerTrainingConfig):
        override = getattr(args, f.name, None)
        if override is not None:
            values[f.name] = override
    cfg = TransformerTrainingConfig(**values)

    dataset_name = read_json(args.data / "manifest.json")["name"]
    report = train_transformer(args.data, cfg, ARTIFACTS_DIR / "models" / dataset_name, REPORTS_DIR / dataset_name)
    print(json.dumps({"val_macro_f1": report["validation"]["macro_f1"], "val_accuracy": report["validation"]["accuracy"],
                      "best_epoch": report["best_params"]["best_epoch"], "history": report["history"],
                      "training_time_seconds": report["training_time_seconds"], "model_dir": report["model_dir"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
