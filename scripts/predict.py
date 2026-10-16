"""Standalone sentiment prediction (no MCP involved).

Usage (PowerShell):
    python scripts/predict.py "The battery died after two days, very disappointed."
    python scripts/predict.py --model-dir artifacts/models/amazon_electronics/tfidf_linearsvm "Great sound!"
    python scripts/predict.py --backend keyword "Great sound!"      # explicit demo baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from review_intel.inference.service import ServiceConfig, SentimentService  # noqa: E402
from review_intel.models.base import ModelNotAvailableError  # noqa: E402
from review_intel.utils import PROJECT_ROOT, setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="+", help="One or more texts to classify")
    parser.add_argument("--backend", choices=["trained", "keyword"], default="trained")
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    setup_logging()

    model_dir = args.model_dir if args.model_dir is None or args.model_dir.is_absolute() else PROJECT_ROOT / args.model_dir
    try:
        service = SentimentService.from_config(ServiceConfig(backend=args.backend, model_dir=model_dir))
    except ModelNotAvailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    results = []
    for text in args.text:
        try:
            results.append(service.analyze(text))
        except ValueError as exc:
            results.append({"text": text, "error": str(exc)})
    print(json.dumps(results if len(results) > 1 else results[0], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
