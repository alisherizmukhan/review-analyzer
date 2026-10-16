"""Model C: fine-tuned DistilBERT (optional; needs ``pip install -e .[transformer]``).

Plain PyTorch training loop (no ``Trainer``) so every step is visible:
seeded shuffling, AdamW + linear warmup/decay, optional class-weighted loss,
bf16/fp16 autocast on GPU, evaluation on the VALIDATION split after every
epoch, and the best epoch (validation macro-F1) is the one saved. The test
split is never read here.

Saved artifact (``artifacts/models/<dataset>/distilbert/``): tokenizer files,
config.json + model.safetensors (weights, id2label/label2id), metadata.json
(training config, dataset manifest reference, history, versions).
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from review_intel.data.labels import LABELS
from review_intel.data.pipeline import load_split
from review_intel.evaluation.metrics import classification_metrics, evaluate_model, measure_latency
from review_intel.models.artifacts import make_version, model_info
from review_intel.models.base import Prediction
from review_intel.utils import environment_info, package_versions, read_json, relative_to_root, sha256_file, utc_now, write_json

logger = logging.getLogger(__name__)

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


@dataclass
class TransformerTrainingConfig:
    base_model: str = "distilbert/distilbert-base-uncased"
    max_length: int = 256
    batch_size: int = 32
    eval_batch_size: int = 128
    epochs: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    class_weight: str | None = "balanced"
    max_train_samples: int | None = None  # subsample train for quick/CPU runs (seeded)
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    precision: str = "auto"  # "auto" | "fp32" | "bf16" | "fp16"
    seed: int = 42
    name: str = "distilbert"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    return torch.device(device)


def resolve_precision(precision: str, device: torch.device) -> torch.dtype | None:
    """Autocast dtype, or None for full fp32."""
    if device.type != "cuda" or precision == "fp32":
        return None
    if precision == "fp16":
        return torch.float16
    if precision in ("bf16", "auto"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else (torch.float16 if precision == "auto" else None)
    raise ValueError(f"Unknown precision {precision!r}")


class TransformerSentimentModel:
    model_type = "transformer"
    is_trained_model = True
    score_type = "softmax_probability"

    def __init__(self, model, tokenizer, metadata: dict[str, Any], device: torch.device | None = None,
                 max_length: int | None = None, batch_size: int = 64):
        self.device = device or resolve_device("auto")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.name = metadata["name"]
        self.version = metadata["version"]
        self.labels = tuple(metadata.get("labels", LABELS))
        self.max_length = max_length or metadata.get("training_config", {}).get("max_length", 256)
        self.batch_size = batch_size

    @classmethod
    def load(cls, directory: Path, device: str = "auto") -> TransformerSentimentModel:
        metadata = read_json(directory / "metadata.json")
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True)
        return cls(model, tokenizer, metadata, resolve_device(device))

    @torch.inference_mode()
    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        out: list[Prediction] = []
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(list(texts[start:start + self.batch_size]), truncation=True,
                                   max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.model(**batch).logits.float(), dim=-1).cpu().numpy()
            for row in probs:
                out.append(Prediction(label=id2label[int(row.argmax())],
                                      scores={id2label[i]: round(float(v), 6) for i, v in enumerate(row)}))
        return out

    def info(self) -> dict[str, Any]:
        info = model_info(self.metadata, self.score_type)
        info["device"] = str(self.device)
        return info


def _batches(tokenizer, texts: list[str], labels: list[int], cfg: TransformerTrainingConfig,
             generator: torch.Generator, shuffle: bool, batch_size: int) -> DataLoader:
    indices = list(range(len(texts)))

    def collate(idx: list[int]):
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=cfg.max_length, padding=True,
                        return_tensors="pt")
        enc["labels"] = torch.tensor([labels[i] for i in idx], dtype=torch.long)
        return enc

    return DataLoader(indices, batch_size=batch_size, shuffle=shuffle, generator=generator, collate_fn=collate)


def _evaluate_logits(model, loader, device, amp_dtype) -> tuple[list[int], list[int]]:
    model.eval()
    preds, gold = [], []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch = batch.to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(**batch).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            gold.extend(labels.tolist())
    model.train()
    return gold, preds


def train_transformer(processed_dir: Path, cfg: TransformerTrainingConfig, models_dir: Path, reports_dir: Path,
                      model=None, tokenizer=None) -> dict[str, Any]:
    """Fine-tune, keep the best epoch by validation macro-F1, save the artifact and a validation report.

    ``model``/``tokenizer`` can be injected (tests use a tiny random model: no download).
    """
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    amp_dtype = resolve_precision(cfg.precision, device)
    manifest = read_json(processed_dir / "manifest.json")
    dataset = {"name": manifest["name"], "is_synthetic": manifest["is_synthetic"],
               "processed_dir": relative_to_root(processed_dir), "source_sha256": manifest["source"]["sha256"],
               "split_sizes": manifest["split_sizes"]}

    x_train, y_train = load_split(processed_dir, "train")
    x_val, y_val = load_split(processed_dir, "val")
    if cfg.max_train_samples and cfg.max_train_samples < len(x_train):
        rng = random.Random(cfg.seed)
        keep = sorted(rng.sample(range(len(x_train)), cfg.max_train_samples))
        x_train, y_train = [x_train[i] for i in keep], [y_train[i] for i in keep]
    train_ids = [LABEL2ID[y] for y in y_train]
    val_ids = [LABEL2ID[y] for y in y_val]
    logger.info("Device %s, precision %s, train=%d val=%d", device, amp_dtype or "fp32", len(x_train), len(x_val))

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.base_model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    model.to(device).train()

    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = _batches(tokenizer, x_train, train_ids, cfg, generator, True, cfg.batch_size)
    val_loader = _batches(tokenizer, x_val, val_ids, cfg, generator, False, cfg.eval_batch_size)

    weight = None
    if cfg.class_weight == "balanced":
        counts = np.bincount(train_ids, minlength=len(LABELS)).astype(float)
        weight = torch.tensor(len(train_ids) / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(cfg.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    out_dir = models_dir / cfg.name
    history, best_f1, best_epoch = [], -1.0, 0
    started = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        epoch_start, running, steps = time.perf_counter(), 0.0, 0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = batch.to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            steps += 1
            if steps % 200 == 0:
                logger.info("  epoch %d step %d/%d loss %.4f", epoch, steps, len(train_loader), running / steps)
        gold, preds = _evaluate_logits(model, val_loader, device, amp_dtype)
        f1 = classification_metrics([ID2LABEL[g] for g in gold], [ID2LABEL[p] for p in preds])["macro_f1"]
        history.append({"epoch": epoch, "train_loss": round(running / max(steps, 1), 4), "val_macro_f1": f1,
                        "epoch_seconds": round(time.perf_counter() - epoch_start, 1)})
        logger.info("Epoch %d: train loss %.4f, val macro-F1 %.4f", epoch, running / max(steps, 1), f1)
        if f1 > best_f1:  # keep the best epoch (validation-based model selection)
            best_f1, best_epoch = f1, epoch
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
    training_seconds = time.perf_counter() - started

    weights = out_dir / "model.safetensors"
    metadata = {
        "name": cfg.name,
        "model_type": "transformer",
        "algorithm": f"Fine-tuned {cfg.base_model} (sequence classification)",
        "labels": list(LABELS),
        "label2id": LABEL2ID,
        "score_type": "softmax_probability",
        "dataset": dataset,
        "seed": cfg.seed,
        "training_config": asdict(cfg),
        "params": {"training_config": asdict(cfg)},
        "train_samples_used": len(x_train),
        "best_epoch": best_epoch,
        "history": history,
        "device": str(device) + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        "precision": str(amp_dtype or "fp32"),
        "training_time_seconds": round(training_seconds, 1),
        "weights_sha256": sha256_file(weights) if weights.exists() else None,
        "created_utc": utc_now(),
        "package_versions": package_versions(),
    }
    metadata["version"] = make_version(cfg.name, dataset["name"], metadata["weights_sha256"] or "00000000")
    write_json(out_dir / "metadata.json", metadata)

    # Reload the saved (best-epoch) artifact and evaluate it exactly as inference will use it.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    saved = TransformerSentimentModel.load(out_dir, device=str(device))
    val_metrics, _ = evaluate_model(saved, x_val, y_val, batch_size=cfg.eval_batch_size)
    latency = measure_latency(saved, x_val, seed=cfg.seed)
    metadata["validation_metrics"] = {"macro_f1": val_metrics["macro_f1"], "accuracy": val_metrics["accuracy"]}
    write_json(out_dir / "metadata.json", metadata)

    report = {
        "family": cfg.name,
        "model_dir": relative_to_root(out_dir),
        "model_version": metadata["version"],
        "algorithm": metadata["algorithm"],
        "score_type": "softmax_probability",
        "dataset": dataset,
        "selection_within_family": "best epoch by validation macro-F1",
        "best_params": {"training_config": asdict(cfg), "best_epoch": best_epoch},
        "training_time_seconds": metadata["training_time_seconds"],
        "validation": val_metrics,
        "latency": latency,
        "history": history,
        "created_utc": utc_now(),
        "environment": environment_info(),
    }
    write_json(reports_dir / "validation" / f"{cfg.name}.json", report)
    return report
