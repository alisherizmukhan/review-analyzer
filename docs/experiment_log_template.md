# Experiment record: template

Copy this file for each new experiment (e.g. `docs/experiments/2026-11-02_group-split.md`).
**Rules:** fill in only values copied from files under `reports/`, `data/processed/*/manifest.json` or `artifacts/models/*/metadata.json`; write *not executed* for anything that was not run; never round, estimate or "expect" a number. For the standard tables, prefer `python scripts/render_results.py --dataset <name>`, which reads the reports directly.

## 1. Identification

| Field | Value |
|---|---|
| Experiment id / date | |
| Research question addressed | RQ1 / RQ2 / RQ3 / other: |
| Hypothesis (written before running) | |
| Git commit or code snapshot | |
| Operator / machine (CPU, GPU, RAM) | |

## 2. Data

| Field | Value (source file) |
|---|---|
| Dataset name / config file | `configs/data/...` |
| Synthetic? | (`manifest.json` → `is_synthetic`) |
| Input SHA-256 | (`manifest.json` → `source.sha256`) |
| Rows: input → after cleaning | |
| Exclusions by reason | |
| Split strategy, seed, sizes | |
| Class distribution (train / val / test) | |
| Leakage checks | |

## 3. Models and configuration

| Model | Config / hyperparameters (from `metadata.json`) | Version string | Seed |
|---|---|---|---|
| | | | |

Changes compared with the previous experiment, and **why** (decisions may use validation data only):

## 4. Validation results (used for selection)

| Model | Accuracy | Macro-P | Macro-R | Macro-F1 | F1 neg | F1 neu | F1 pos | ECE | Train time (s) | Latency median / p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |

Selection rule applied: highest validation macro-F1 (ties: LogReg → SVM → DistilBERT). Selected model:

## 5. Test results (run once, after selection)

| Model | Role (selected / post-selection) | Accuracy | Macro-F1 | F1 neg | F1 neu | F1 pos | ECE |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

Confusion matrix of the selected model (rows = true, columns = predicted):

| | negative | neutral | positive |
|---|---|---|---|
| negative | | | |
| neutral | | | |
| positive | | | |

## 6. Interpretation

- Result vs hypothesis:
- Effect size, and whether it exceeds run-to-run noise (seed variation / bootstrap CI, if computed):
- Error analysis (typical misclassified examples, from the test predictions):
- Threats to validity specific to this run:

## 7. Not executed / deviations

List every planned step that was not run, and every deviation from the methodology, with the reason.
