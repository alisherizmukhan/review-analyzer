"""Small shared helpers: paths, logging, hashing, JSON I/O, environment capture."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

# src/review_intel/utils.py -> project root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRACKED_PACKAGES = (
    "numpy", "scikit-learn", "scipy", "joblib", "torch", "transformers",
    "tokenizers", "sentence-transformers", "mcp",
)


def setup_logging(level: int = logging.INFO) -> None:
    """Log to stderr (stdout stays clean for data output and MCP stdio)."""
    logging.basicConfig(level=level, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "packages": package_versions(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:  # optional
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def relative_to_root(path: Path) -> str:
    """Portable path for manifests (no machine-specific absolute paths)."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name
