# Held-out test results: amazon_electronics

Test size: 9659. Selected model: **`distilbert`** (selected on validation macro-F1 = 0.7444).

Rows marked 'post-selection' were evaluated on test only after the selection was frozen; they were not used for any decision.

| Model | Accuracy | Macro-P | Macro-R | Macro-F1 | F1 neg | F1 neu | F1 pos | Train time (s) | Latency median / p95 (ms) | Batch texts/s | Score type |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distilbert | 0.8927 | 0.7322 | 0.7874 | **0.7522** | 0.8493 | 0.4513 | 0.9559 | 572.4 | 3.437 / 5.533 | 340.6 | softmax_probability |
| tfidf_logreg (post-selection) | 0.8521 | 0.6796 | 0.7462 | **0.7009** | 0.7926 | 0.3802 | 0.9299 | 52.37 | 1.084 / 2.085 | 3940.9 | probability |
| tfidf_linearsvm (post-selection) | 0.8833 | 0.6991 | 0.7067 | **0.7021** | 0.8134 | 0.3453 | 0.9477 | 29.0 | 1.075 / 1.988 | 3726.5 | decision_score |

## distilbert: confusion matrix (test)

| true \ predicted | negative | neutral | positive |
|---|---|---|---|
| negative | 1494 | 223 | 27 |
| neutral | 190 | 378 | 89 |
| positive | 90 | 417 | 6751 |

Calibration (top-label ECE, 15 bins): {'ece': 0.0324, 'bins': 15, 'mean_confidence': 0.925, 'accuracy': 0.8927}

## tfidf_logreg: confusion matrix (test)

| true \ predicted | negative | neutral | positive |
|---|---|---|---|
| negative | 1427 | 261 | 56 |
| neutral | 195 | 349 | 113 |
| positive | 235 | 569 | 6454 |

Calibration (top-label ECE, 15 bins): {'ece': 0.0423, 'bins': 15, 'mean_confidence': 0.8097, 'accuracy': 0.8521}

## tfidf_linearsvm: confusion matrix (test)

| true \ predicted | negative | neutral | positive |
|---|---|---|---|
| negative | 1482 | 145 | 117 |
| neutral | 227 | 216 | 214 |
| positive | 191 | 233 | 6834 |

## Environment

`{'python': '3.12.10', 'numpy': '2.5.4', 'scikit-learn': '1.9.1', 'scipy': '1.18.1', 'joblib': '1.6.0', 'torch': '2.11.0+cu128', 'transformers': '5.19.0', 'tokenizers': '0.23.3', 'sentence-transformers': '6.1.0', 'mcp': '2.3.0'}`

Platform: Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel; GPU: NVIDIA GeForce RTX 5060
