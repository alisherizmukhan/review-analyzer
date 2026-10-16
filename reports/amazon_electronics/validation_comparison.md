# Validation comparison: amazon_electronics

Synthetic data: **False**. Split sizes: {'train': 77268, 'val': 9659, 'test': 9659}.

Selection rule: Highest validation macro-F1; ties (equal to 4 decimals) broken by the fixed preference order ['tfidf_logreg', 'tfidf_linearsvm', 'distilbert'] (cheaper model first). The test split is not used for selection.

**Selected: `distilbert`**

All numbers below are on the VALIDATION split.

| Model | Accuracy | Macro-P | Macro-R | Macro-F1 | F1 neg | F1 neu | F1 pos | Train time (s) | Latency median / p95 (ms) | Batch texts/s | Score type |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distilbert | 0.8905 | 0.7260 | 0.7760 | **0.7444** | 0.8481 | 0.4302 | 0.9549 | 572.4 | 3.313 / 5.392 | 341.2 | softmax_probability |
| tfidf_logreg | 0.8521 | 0.6741 | 0.7399 | **0.6966** | 0.7832 | 0.3753 | 0.9313 | 52.37 | 1.113 / 1.954 | 4250.8 | probability |
| tfidf_linearsvm | 0.8805 | 0.6916 | 0.6957 | **0.6920** | 0.8038 | 0.3259 | 0.9463 | 29.0 | 1.085 / 1.839 | 4489.1 | decision_score |

## distilbert: confusion matrix (validation)

| true \ predicted | negative | neutral | positive |
|---|---|---|---|
| negative | 1493 | 221 | 31 |
| neutral | 199 | 356 | 102 |
| positive | 84 | 421 | 6752 |

Best parameters: `{'training_config': {'base_model': 'distilbert/distilbert-base-uncased', 'max_length': 256, 'batch_size': 32, 'eval_batch_size': 128, 'epochs': 2, 'learning_rate': 5e-05, 'weight_decay': 0.01, 'warmup_ratio': 0.06, 'class_weight': 'balanced', 'max_train_samples': None, 'device': 'auto', 'precision': 'auto', 'seed': 42, 'name': 'distilbert'}, 'best_epoch': 2}`
Calibration (top-label ECE, 15 bins): {'ece': 0.0338, 'bins': 15, 'mean_confidence': 0.9239, 'accuracy': 0.8905}

## tfidf_logreg: confusion matrix (validation)

| true \ predicted | negative | neutral | positive |
|---|---|---|---|
| negative | 1432 | 264 | 49 |
| neutral | 212 | 334 | 111 |
| positive | 268 | 525 | 6464 |

Best parameters: `{'features': {'id': 'word_1-2+char_3-5', 'word': {'ngram_range': [1, 2], 'min_df': 2, 'max_df': 0.95, 'max_features': 200000}, 'char': {'ngram_range': [3, 5], 'min_df': 3, 'max_features': 300000}}, 'classifier': {'model_type': 'logreg', 'C': 0.5, 'class_weight': 'balanced', 'max_iter': 3000}}`
Calibration (top-label ECE, 15 bins): {'ece': 0.0425, 'bins': 15, 'mean_confidence': 0.8103, 'accuracy': 0.8521}

## tfidf_linearsvm: confusion matrix (validation)

| true \ predicted | negative | neutral | positive |
|---|---|---|---|
| negative | 1479 | 136 | 130 |
| neutral | 243 | 196 | 218 |
| positive | 213 | 214 | 6830 |

Best parameters: `{'features': {'id': 'word_1-2+char_3-5', 'word': {'ngram_range': [1, 2], 'min_df': 2, 'max_df': 0.95, 'max_features': 200000}, 'char': {'ngram_range': [3, 5], 'min_df': 3, 'max_features': 300000}}, 'classifier': {'model_type': 'linear_svm', 'C': 0.05, 'class_weight': 'balanced', 'max_iter': 5000}}`

## Latency method

Single-text latency: 300 texts drawn with a fixed seed from the evaluation split are predicted one at a time (batch size 1) after 20 warm-up calls; we report the median and 95th percentile wall-clock time per call (time.perf_counter), including vectorization/tokenization. Batch throughput: the same texts are predicted in batches of 256; we report texts per second. Measured in-process on the machine listed under 'environment'; no network or disk I/O is included.
