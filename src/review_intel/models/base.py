"""Common interface for every sentiment model (keyword baseline, scikit-learn, transformer).

Inference, analytics and the MCP server depend only on ``SentimentModel``,
so models can be swapped without touching those layers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# What the numbers in Prediction.scores mean. Reported with every prediction
# so no consumer mistakes an SVM margin for a probability.
SCORE_TYPES: dict[str, str] = {
    "probability": (
        "Class probabilities estimated by the model (predict_proba). They sum to 1 but are not guaranteed to be "
        "well calibrated; see the expected calibration error (ECE) in the evaluation report."
    ),
    "softmax_probability": (
        "Softmax over the transformer's output logits. They sum to 1 but fine-tuned transformers are often "
        "over-confident; see the ECE in the evaluation report before reading them as probabilities."
    ),
    "decision_score": (
        "Uncalibrated one-vs-rest decision scores (signed distances to the SVM hyperplanes). Higher means a "
        "stronger vote for that class. These are NOT probabilities and do not sum to 1."
    ),
    "keyword_count": "Number of matched sentiment keywords per class (demonstration baseline, not a model score).",
}


class ModelNotAvailableError(RuntimeError):
    """No usable trained model artifact; the message says how to create one."""


@dataclass
class Prediction:
    label: str
    scores: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SentimentModel(Protocol):
    name: str
    version: str
    model_type: str  # "keyword_baseline" | "sklearn" | "transformer"
    labels: tuple[str, ...]
    score_type: str  # key of SCORE_TYPES
    is_trained_model: bool

    def predict(self, texts: Sequence[str]) -> list[Prediction]: ...

    def info(self) -> dict[str, Any]: ...
