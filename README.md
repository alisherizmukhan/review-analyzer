# Review Analyzer

**NLP/ML system for analyzing Amazon Electronics customer reviews, with an MCP server that lets Claude Code query the trained model and the analytics as tools.**

It covers the whole path from raw data to a usable service: a reproducible data pipeline, three sentiment classifiers compared under one protocol (TF-IDF + Logistic Regression, TF-IDF + Linear SVM, fine-tuned DistilBERT), held-out evaluation, product-level analytics with evidence, similar-review retrieval, and a typed, read-only MCP integration layer with 219 automated tests.

| | |
|---|---|
| **Selected model** | DistilBERT fine-tuned on 77,268 reviews: test accuracy **0.8927**, macro-F1 **0.7522** (n = 9,659) |
| **Data** | Amazon Reviews 2023, Electronics: a uniform random sample of 100,000 reviews out of 43.9M, plus a separate 49,958-review corpus of 50 products |
| **Interface** | 9 MCP tools over stdio (sentiment, statistics, keyword and similar-review search, product summaries, comparisons, negative aspects, model info) |
| **Status** | Research/portfolio project, run locally. Not deployed as a production service. |

---

## Contents

1. [Problem](#problem)
2. [Capabilities](#capabilities)
3. [Architecture](#architecture)
4. [Repository structure](#repository-structure)
5. [Data](#data)
6. [Models](#models)
7. [Training and evaluation methodology](#training-and-evaluation-methodology)
8. [Results](#results)
9. [Sentiment labels vs star ratings](#sentiment-labels-vs-star-ratings)
10. [Similar-review search](#similar-review-search)
11. [MCP server and its nine tools](#mcp-server-and-its-nine-tools)
12. [Installation](#installation)
13. [Getting the data and models](#getting-the-data-and-models)
14. [Configuration](#configuration)
15. [Running the tests](#running-the-tests)
16. [Using the MCP server in Claude Code](#using-the-mcp-server-in-claude-code)
17. [Example usage](#example-usage)
18. [Limitations and future work](#limitations-and-future-work)
19. [Reproducibility](#reproducibility)
20. [Technology stack](#technology-stack)
21. [Licensing and acknowledgements](#licensing-and-acknowledgements)
22. [License](#license)

---

## Problem

Product reviews are abundant, unstructured and noisy. A product team wants to know (a) whether a review is positive, neutral or negative, and (b) **which aspects** of a product drive complaints, with evidence, without reading thousands of reviews. Doing this credibly requires more than a classifier: clean data without leakage, an honest evaluation on held-out data, analytics that report their sample sizes, and an interface that other tools (here, an LLM assistant) can call safely.

## Capabilities

| Capability | Implementation | Status |
|---|---|---|
| Streaming sampler for the 22.6 GB Amazon review file (uniform reservoir sample) | `src/review_intel/data/amazon.py` | executed |
| Resumable per-product collection (byte-range chunks, order-independent bottom-k sampling) | `src/review_intel/data/product_collection.py` | executed |
| Dataset pipeline: cleaning, star-rating labels, de-duplication, stratified split, manifest with leakage checks | `src/review_intel/data/pipeline.py` | executed |
| Sentiment models A/B/C behind one interface | `src/review_intel/models/` | trained and evaluated |
| Validation-based model selection, single held-out test evaluation, calibration (ECE), latency | `src/review_intel/evaluation/` | executed |
| Inference service: input validation, explicit score types, no silent fallback | `src/review_intel/inference/service.py` | used by MCP |
| Product statistics, summaries with 95% Wilson intervals, comparisons, negative aspects | `src/review_intel/analytics/` | implemented, descriptive only |
| Similar-review search: LSA (TF-IDF + truncated SVD) with an optional sentiment filter | `src/review_intel/retrieval/semantic.py` | implemented |
| Optional dense retriever (sentence-transformers) | same file | implemented and unit-tested; **not run on real data** |
| MCP server with 9 typed, read-only tools | `review-analyzer-mcp/server.py` | verified over stdio |
| Aspect-level accuracy, confidence intervals for model metrics | none | **not done**: no gold aspect labels; single split |

## Architecture

```mermaid
flowchart LR
    subgraph data["Data pipeline"]
        AMZ[("Amazon Reviews 2023<br/>Electronics, 43.9M reviews")] -->|"stream + reservoir sample<br/>scripts/download_amazon.py"| RAW["data/raw/*.jsonl"]
        RAW --> PREP["scripts/prepare_data.py<br/>clean, label, dedup, split, manifest"]
        PREP --> SPLITS["train / val / test CSV<br/>+ manifest.json"]
    end
    subgraph ml["Models and evaluation"]
        SPLITS --> A["A: TF-IDF + LogReg"]
        SPLITS --> B["B: TF-IDF + LinearSVM"]
        SPLITS --> C["C: DistilBERT fine-tune"]
        A & B & C -->|"validation macro-F1"| SEL["model selection<br/>artifacts/models/selected.json"]
        SEL -->|"then, once"| TEST["held-out test report<br/>reports/"]
    end
    subgraph svc["review_intel services"]
        SEL --> INF["SentimentService"]
        INF --> ASP["Aspect analyzer<br/>lexicon + sentence sentiment"]
        INF --> STATS["Product summary / compare"]
        RET["LSA retriever<br/>+ sentiment filter"]
        CORPUS[("Analytics corpus CSV")] --> STATS & ASP & RET
    end
    subgraph mcp["review-analyzer-mcp"]
        TOOLS["9 MCP tools<br/>schema validation, ToolError,<br/>structured output"]
    end
    INF & ASP & STATS & RET --> TOOLS
    TOOLS <-->|"JSON-RPC over stdio"| HOST["Claude Code<br/>(or another MCP client)"]
```

The ML code (`src/review_intel`) does not depend on MCP. The MCP server is a thin layer that validates arguments, calls the package and labels where every value comes from.

## Repository structure

```
review-analyzer/
├── README.md                    # this file
├── pyproject.toml               # package "review-intel" + extras [mcp] [transformer] [semantic] [data] [dev]
├── requirements-lock.txt        # exact versions of the environment that produced the results (Windows, CUDA 12.8)
├── configs/
│   ├── data/                    # dataset configs: column mapping, label source, split, seed
│   ├── models/                  # baseline grids, DistilBERT training config
│   └── aspects.json             # aspect lexicon (configuration, not a trained model)
├── src/review_intel/
│   ├── data/                    # amazon.py (sampler), product_collection.py, labels.py, pipeline.py
│   ├── preprocessing/text.py    # stateless cleaning, star-title removal, sentence split
│   ├── models/                  # common interface, sklearn models, DistilBERT, training, artifacts
│   ├── evaluation/              # metrics, calibration, latency, selection + test protocol
│   ├── inference/service.py     # SentimentService (independent of MCP)
│   ├── analytics/               # corpus loading, aspects, statistics, summaries, comparisons
│   ├── retrieval/semantic.py    # LSA + optional dense retriever + sentiment filter
│   └── app.py                   # facade used by the MCP server
├── scripts/                     # command-line entry points for every pipeline stage
├── review-analyzer-mcp/         # MCP server, its README and its tests
├── tests/                       # tests for the ML/data package
├── reports/amazon_electronics/  # machine-generated validation and test reports (JSON + Markdown)
├── docs/                        # methodology, full results, experiment log template
└── data/
    ├── fixtures/                # 73 synthetic reviews used by the tests
    └── processed/*/manifest.json  # dataset manifests (the data files themselves are not committed)
```

Not in the repository (git-ignored, rebuilt by the scripts): `data/raw/`, the processed CSV files and `artifacts/` (trained models, about 285 MB). See [Getting the data and models](#getting-the-data-and-models).

## Data

**Source:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UC San Diego; Hou et al., 2024, arXiv:2403.03952), category Electronics. The streaming pass counted 43,886,944 review lines.

**Research corpus (training and evaluation).** The file is ordered by user, so taking its first lines would be biased. `scripts/download_amazon.py` streams the whole file once and keeps a uniform random sample of 100,000 reviews (reservoir sampling, seed 42). After cleaning and de-duplication, 96,586 remain:

| Split | Reviews | Negative | Neutral | Positive |
|---|---|---|---|---|
| train | 77,268 | 18.06% | 6.80% | 75.14% |
| validation | 9,659 | 18.07% | 6.80% | 75.13% |
| test | 9,659 | 18.06% | 6.80% | 75.14% |

Excluded during cleaning (counted in `data/processed/amazon_electronics/manifest.json`): 113 empty or too-short texts, 2,145 duplicate texts, 1,156 copies of texts whose duplicates carried conflicting labels. Text overlap between splits is 0.

**Analytics corpora (used only by the MCP tools, never for training).**

| Corpus | File | Contents | Use |
|---|---|---|---|
| Original (default) | `data/processed/amazon_electronics/analytics_corpus.csv` | The 9,659-review held-out **test split**; 8,177 products, 7,369 of them with a single review; titles for 25 products | Corpus-level statistics and search |
| Per-product | `data/processed/amazon_electronics_products/corpus.csv` | **49,958 reviews** for the 50 most-reviewed products (996–1,000 each), official titles for all 50; no review from the research sample | Product summaries and comparisons |

The per-product corpus was read from the official Hugging Face mirror of the same file (pinned commit `2b6d039e…`). 1,000 reviews per product is a uniform random sample of each product's reviews (bottom-k by `sha1(seed | review key)`), not all of them. Its label distribution is 80.54% positive, 13.08% negative, 6.38% neutral. Provenance and checks: [methodology §2.2](docs/research_methodology.md#22-per-product-analytics-corpus-collected-2026-10-15) and `data/processed/amazon_electronics_products/validation_report.json`.

**Limitations of the data:** English only, one category, one platform; labels come from star ratings, not human sentiment annotation; popular products appear in several splits (reported in the manifest), so generalization to unseen products is not measured.

## Models

All models implement one interface (`SentimentModel` in `src/review_intel/models/base.py`) and return a label plus scores whose type is stated explicitly.

| Model | Features | Classifier | Scores |
|---|---|---|---|
| **A** TF-IDF + Logistic Regression | word 1–2-grams + character 3–5-grams (`char_wb`), sublinear TF | `LogisticRegression` (lbfgs), C = 0.5, balanced class weights | probabilities |
| **B** TF-IDF + Linear SVM | same feature union | `LinearSVC`, C = 0.05, balanced class weights | decision scores (**not** probabilities) |
| **C** DistilBERT | WordPiece tokens, max length 256 | `distilbert/distilbert-base-uncased` + classification head, fine-tuned end to end | softmax probabilities |
| Reference | keyword counts | none (the original v1 demo) | keyword counts; used only when explicitly configured |

DistilBERT training (`src/review_intel/models/transformer.py`, plain PyTorch loop): AdamW (lr 5e-5, weight decay 0.01), linear warmup (6%) and decay, class-weighted cross-entropy, gradient clipping at 1.0, batch 32, 2 epochs, bf16 autocast on an RTX 5060. The best epoch by validation macro-F1 is saved (epoch 2; training took 572 s).

## Training and evaluation methodology

1. **Split once:** stratified 80/10/10, seed 42, after de-duplication (so no text appears in two splits).
2. **Tune on validation only:** grids over feature sets, C and class weights for A and B; best epoch for C. The criterion is **macro-F1**, because 75% of reviews are positive and accuracy would reward ignoring the minority classes.
3. **Select across families on validation:** DistilBERT 0.7444, LogReg 0.6966, SVM 0.6920. The selection is written to `reports/amazon_electronics/model_selection.json`.
4. **Open the test set once:** `scripts/evaluate_models.py test` refuses to run before a selection is recorded. The other two models were evaluated on test afterwards for comparison only.

Learned preprocessing (TF-IDF vocabulary and IDF) is fitted on the training split only, and a unit test checks that training code never reads the test split. Full method: [`docs/research_methodology.md`](docs/research_methodology.md).

## Results

Held-out test set, n = 9,659 (`reports/amazon_electronics/test_report.md`):

| Model | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive | ECE | Train time | Median latency (1 text) |
|---|---|---|---|---|---|---|---|---|
| **DistilBERT** (selected) | **0.8927** | **0.7522** | 0.8493 | 0.4513 | 0.9559 | 0.0324 | 572 s (GPU) | 3.44 ms (GPU) |
| TF-IDF + LinearSVM | 0.8833 | 0.7021 | 0.8134 | 0.3453 | 0.9477 | n/a | 29 s | 1.08 ms |
| TF-IDF + LogReg | 0.8521 | 0.7009 | 0.7926 | 0.3802 | 0.9299 | 0.0423 | 52 s | 1.08 ms |

DistilBERT confusion matrix on test (rows = true label):

| true \ predicted | negative | neutral | positive |
|---|---|---|---|
| negative (1,744) | 1,494 | 223 | 27 |
| neutral (657) | 190 | 378 | 89 |
| positive (7,258) | 90 | 417 | 6,751 |

How to read these numbers:
- DistilBERT is about 5 macro-F1 points better than both linear models. That is one split and one seed, with no significance test.
- **Neutral is the weak class for every model** (F1 0.35–0.45). Only 6.8% of reviews are 3-star, and a 3-star rating is a noisy label for the sentiment of the text. The confusion matrix shows neutral errors going in both directions.
- The SVM has higher accuracy than LogReg but lower macro-recall, because it favors the majority class. This is why macro-F1, not accuracy, is the selection metric.
- These metrics measure agreement with **star-rating labels**, not with human judgments of the text.

All numbers were checked against the JSON reports on 2026-10-16. The recorded model version is `distilbert@amazon_electronics-20261014-79b5c36e`. The validation macro-F1 logged during training (0.7441, bf16 autocast) differs from the reported 0.7444 (the saved artifact re-evaluated in fp32); the difference is numerical. Validation tables, grids and latency method: [`docs/experiment_results.md`](docs/experiment_results.md).

## Sentiment labels vs star ratings

Labels are derived from star ratings with a fixed rule (`src/review_intel/data/labels.py`): **1–2 stars → negative, 3 → neutral, 4–5 → positive**. Fractional or missing ratings are excluded, not rounded. Auto-generated titles such as "Five Stars" are removed from the text because they would leak the label.

A star rating is a proxy: a 4-star review can complain about the battery, and a 3-star review can read as clearly positive. The project therefore keeps two sources of sentiment strictly separate, and every MCP response names its source in a `sentiment_sources` field:

| Source | Meaning | Used by |
|---|---|---|
| `star_rating_label` | The stored label derived from the customer's rating (verified against the rating for every review) | `get_review_statistics`, `search_reviews`, `search_similar_reviews` (`sentiment`), sentiment distributions in summaries and comparisons |
| `model_prediction` | DistilBERT's prediction for the text, with the model version | `analyze_sentiment`, `get_negative_aspects`, aspect sentiment, `predicted_sentiment` in filtered similar-review search |

## Similar-review search

`search_similar_reviews` ranks reviews by **cosine similarity of LSA vectors**: TF-IDF over word 1–2-grams (English stop words removed), reduced to 200 dimensions with truncated SVD (`LsaRetriever`). LSA captures co-occurrence ("drains" near "battery life"); it is **lexical-topical similarity, not semantic understanding**, and it ignores negation and polarity.

Measured on the per-product corpus (2026-10-16): for *"The battery drains too quickly."* the top unfiltered hit (similarity 0.967) is a short review praising the battery, the opposite of the query (illustration, not the original text: [Synthetic review: "The battery runs for ages between charges."]), and 4 of the top 5 results are 5-star reviews.

The optional sentiment filter does not change the ranking. It only removes reviews whose sentiment does not match:

| `sentiment_source` | Filters on | Candidates |
|---|---|---|
| `star_rating_label` | the stored star-rating label | the whole ranking (exact top-`limit`) |
| `model_prediction` | DistilBERT's prediction for the review text | only the `candidate_pool` most similar reviews (default 200, max 1,000), because classifying the whole corpus per query is too slow |

With `model_prediction`, fewer results than `limit` means the pool held fewer matches, not that the corpus has none. In the run above, 44 of the 200 candidates were predicted negative, including a 4-star review whose text complains about battery life. A filter without an explicit source is rejected with a validation error.

## MCP server and its nine tools

[MCP (Model Context Protocol)](https://modelcontextprotocol.io/) is a protocol for an AI application to discover and call external tools. It is **not** a model and does no NLP itself. Here, Claude Code launches `review-analyzer-mcp/server.py` as a subprocess and exchanges JSON-RPC messages over stdin/stdout. The server declares typed tools; arguments are validated against JSON Schema (pydantic) before any Python code runs, and expected failures return a readable `isError: true` result.

| Tool | Arguments (defaults) | Returns |
|---|---|---|
| `analyze_sentiment` | `text` | label, class scores, score type, model version |
| `get_review_statistics` | `product=""` | counts and percentages per label, average rating |
| `search_reviews` | `keyword`, `product=""`, `limit=20` | reviews containing the keyword, total match count |
| `list_products` | `limit=20` | products by review count, plus corpus coverage |
| `get_product_summary` | `product` | counts, rating and sentiment distribution with 95% Wilson intervals, sample-size check, top and most-negative aspects, representative reviews |
| `get_negative_aspects` | `product=""`, `limit=5` | aspects ranked by negatively classified mentions, with excerpts |
| `compare_products` | `product_a`, `product_b` | side-by-side statistics, differences, shared aspects, `comparison_reliable` |
| `search_similar_reviews` | `query`, `product=""`, `limit=5`, `sentiment_filter="any"`, `sentiment_source=""`, `candidate_pool=200` | reviews ranked by LSA cosine similarity, optionally filtered by sentiment |
| `get_model_info` | none | active model, validation and test metrics, corpus, retriever, lexicon |

All tools are read-only. There are no file, shell or network tools. Full schemas, example responses, performance notes and troubleshooting: [`review-analyzer-mcp/README.md`](review-analyzer-mcp/README.md).

## Installation

Requires Python 3.11+ (developed and tested on Python 3.12.10, Windows 11).

**Windows (PowerShell)**

```powershell
git clone https://github.com/alisherizmukhan/review-analyzer.git
cd review-analyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[mcp,dev]"          # core ML package + MCP server + pytest
```

To train or serve DistilBERT, install PyTorch first from the index that matches your hardware, then the transformer extra:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU (CUDA 12.8)
# or: pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[transformer]"
```

**Linux / macOS (not tested in this project)**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[mcp,dev]"
pip install torch && pip install -e ".[transformer]"
```

The code uses `pathlib` throughout and resolves all configured paths from the project root, so no path changes should be needed. `requirements-lock.txt` pins the exact Windows/CUDA environment used for the reported results.

## Getting the data and models

The repository does **not** include the review data or the trained model weights:
- **Data:** the Amazon Reviews 2023 pages do not state a licence, so raw and processed reviews are not redistributed. Only manifests (counts, SHA-256 hashes, configuration) and aggregate reports are committed.
- **Models:** the artifacts total about 285 MB (DistilBERT `model.safetensors` alone is 268 MB, over GitHub's 100 MB file limit) and are derived from that data.

Without them you can still run the full test suite (it uses the synthetic fixture) and the MCP server in demo mode (`REVIEW_ANALYZER_DATA=demo`, `REVIEW_ANALYZER_SENTIMENT_BACKEND=keyword`).

To rebuild everything (commands run from the project root; the first step streams about 6.5 GB):

```powershell
# 1. Data: uniform sample of 100,000 reviews, then clean/label/split
python scripts/download_amazon.py --category Electronics --sample-size 100000 --top-products 0 --seed 42
python scripts/prepare_data.py --config configs/data/amazon_electronics.json

# 2. Models: baselines (minutes on CPU), DistilBERT (about 10 minutes on an RTX 5060; much longer on CPU)
python scripts/train_baselines.py --data data/processed/amazon_electronics
python scripts/train_transformer.py --data data/processed/amazon_electronics

# 3. Select on validation and activate for serving, then evaluate on test once
python scripts/evaluate_models.py compare --data data/processed/amazon_electronics --activate
python scripts/evaluate_models.py test --data data/processed/amazon_electronics --all

# 4. Analytics corpora
pip install -e ".[data]"
python scripts/fetch_product_titles.py --from-csv data/processed/amazon_electronics/test.csv --top 25 --out data/raw/amazon_electronics/product_metadata.json
python scripts/build_corpus.py --split data/processed/amazon_electronics/test.csv --names data/raw/amazon_electronics/product_metadata.json --out data/processed/amazon_electronics/analytics_corpus.csv
python scripts/download_amazon.py --products-only --category Electronics --top-from-sample 50 --per-product-cap 1000 --seed 42
python scripts/fetch_product_titles.py --asin-file data/raw/amazon_electronics_products/selected_products.json --out data/raw/amazon_electronics_products/product_metadata.json
python scripts/prepare_data.py --config configs/data/amazon_electronics_products.json
python scripts/validate_product_corpus.py --verify-source 200
```

Compare the `source.sha256` in the new `manifest.json` with the committed one to confirm you have the same sample. A rerun should reproduce the dataset exactly; DistilBERT metrics can differ slightly because GPU kernels are not forced to be deterministic. Expected locations: models in `artifacts/models/amazon_electronics/<family>/`, the active model pointer in `artifacts/models/selected.json`.

## Configuration

The application reads optional environment variables from the process environment (no `.env` file is loaded). Relative paths resolve from the project root.

| Variable | Default | Meaning |
|---|---|---|
| `REVIEW_ANALYZER_DATA` | `data/processed/amazon_electronics/analytics_corpus.csv` | Review corpus (`.csv` or `.json`); `demo` selects the 5-review synthetic file |
| `REVIEW_ANALYZER_SENTIMENT_BACKEND` | `trained` | `trained`, or `keyword` for the explicit demo baseline |
| `REVIEW_ANALYZER_MODEL_DIR` | from `artifacts/models/selected.json` | Serve a specific model artifact |
| `REVIEW_ANALYZER_MAX_TEXT_CHARS` | `5000` | Maximum `analyze_sentiment` input length |
| `REVIEW_ANALYZER_MIN_PRODUCT_REVIEWS` | `30` | Reviews below which a product summary is flagged as a small sample (a heuristic) |
| `REVIEW_ANALYZER_RETRIEVER` | `lsa` | `lsa`, or `sentence-transformers` (needs `.[semantic]`) |
| `REVIEW_ANALYZER_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model for the dense retriever |
| `REVIEW_ANALYZER_ASPECTS` | `configs/aspects.json` | Aspect lexicon |

No API keys or credentials are needed.

## Running the tests

```powershell
python -m pytest                              # everything: ML package + MCP server
python -m pytest review-analyzer-mcp/tests    # MCP layer only
```

Last run (2026-10-16, Windows 11, Python 3.12.10, PyTorch installed): **219 passed**. The tests are offline: they build their own data from `data/fixtures/synthetic_reviews.csv`, train small models on it, use a tiny randomly initialized transformer (no download) and start the MCP server as a real stdio subprocess. The transformer test is skipped if PyTorch is not installed. Passing tests show that the code behaves as specified; they do not re-measure model quality on real data.

## Using the MCP server in Claude Code

Register the server once (PowerShell, from the project root; `-s user` makes it available in every project):

```powershell
claude mcp add review-analyzer -s user `
  -e REVIEW_ANALYZER_DATA="$PWD\data\processed\amazon_electronics_products\corpus.csv" `
  -- "$PWD\.venv\Scripts\python.exe" "$PWD\review-analyzer-mcp\server.py"
```

- Use the **absolute path of the virtual environment's Python**. Claude Code starts the process itself, without your activated shell, so a bare `python` may resolve to another interpreter that cannot import `review_intel`.
- Check the connection with `/mcp` inside Claude Code, or `claude mcp list`.
- The server loads DistilBERT before answering the handshake (8.5 s warm on the development machine; a cold start can approach 30 s). If it times out, start Claude Code with `$env:MCP_TIMEOUT = "120000"`.
- After moving the project folder, re-run `pip install -e .` and update the registered paths (`claude mcp remove review-analyzer -s user`, then add it again).

Without trained artifacts, register it with `-e REVIEW_ANALYZER_DATA=demo -e REVIEW_ANALYZER_SENTIMENT_BACKEND=keyword` to try the tools on the synthetic demo data.

## Example usage

**Standalone prediction (no MCP):**

```powershell
python scripts/predict.py "The remote stopped working after two weeks."
```

**In Claude Code**, natural-language requests map to tool calls, for example:

- *"Which aspects of the Fire TV Stick (B075X8471B) do customers complain about? Show evidence and the sample size."* → `get_negative_aspects`
- *"Find complaints similar to 'the battery drains too quickly', using the model's predictions."* → `search_similar_reviews` with `sentiment_filter="negative"`, `sentiment_source="model_prediction"`

A real `search_similar_reviews` result (per-product corpus, abbreviated). Review texts in this example are synthetic placeholders written for this documentation (labelled `[Synthetic example]`); the original Amazon review texts are not reproduced. All other values (ids, ratings, scores, similarities, counts) are from the real response and were computed on the original texts.

```json
{
  "method": "lsa (tf-idf + truncated SVD, 200 dims, cosine)",
  "sentiment_filter": {"sentiment": "negative", "source": "model_prediction",
                       "model_version": "distilbert@amazon_electronics-20261014-79b5c36e",
                       "candidate_pool": 200, "candidates_scanned": 200, "candidates_matching": 44},
  "results": [
    {"text": "[Synthetic example] Weak battery: it needs charging far more often than advertised. …", "rating": 2,
     "similarity": 0.8902, "sentiment": "negative", "predicted_sentiment": "negative", "predicted_score": 0.922163},
    {"text": "[Synthetic example] Nice screen, but I keep having to recharge it every couple of days. …", "rating": 4,
     "similarity": 0.8631, "sentiment": "positive", "predicted_sentiment": "negative", "predicted_score": 0.866549}
  ]
}
```

More real responses for every tool: [`review-analyzer-mcp/README.md`](review-analyzer-mcp/README.md#example-calls-real-responses-abbreviated-with-).

## Limitations and future work

**Known limitations**
- Labels are star-rating proxies; neutral (3-star) is noisy, and every model is weakest on it.
- One split, one seed, no confidence intervals or significance tests for model metrics.
- DistilBERT ran with a fixed 2-epoch budget and no hyperparameter search; validation F1 was still rising at epoch 2.
- Aspect analysis is lexicon matching plus sentence-level predictions from a review-level model. Its precision, recall and sentence-level accuracy are **not measured** (no gold labels exist).
- LSA retrieval is lexical-topical; the dense retriever is implemented but has not been evaluated on real data. There is no retrieval-quality benchmark.
- English, Electronics and Amazon only; generalization to unseen products is not measured.
- The original analytics corpus has very few reviews per product; product-level conclusions need the per-product corpus.

**Future work**
- A manually labelled, independent test sample (including sentence- and aspect-level labels) to measure agreement with human judgment.
- Bootstrap confidence intervals and repeated runs with several seeds; a product-grouped split.
- Hyperparameter search and longer training for DistilBERT; calibration (temperature scaling).
- Retrieval evaluation (labelled query–review pairs) comparing LSA with dense embeddings.
- A learned aspect extractor or ABSA model behind the existing `AspectExtractor` interface.

## Reproducibility

- Fixed seeds (42) for sampling, splitting, classifiers and fine-tuning.
- Every stage writes a JSON record: download manifest, dataset manifest (config, input SHA-256, exclusion counts, class distributions, leakage checks), model `metadata.json` (parameters, dataset reference, weight SHA-256, package versions), validation and test reports with hardware details.
- Model versions encode the family, dataset, date and the first 8 hex digits of the weights' SHA-256 (e.g. `distilbert@amazon_electronics-20261014-79b5c36e`). scikit-learn artifacts are checksum-verified on load.
- The active model pointer stores a relative path, so the project folder can be moved.
- Known nondeterminism: GPU kernels for DistilBERT. Details: [methodology §9](docs/research_methodology.md#9-reproducibility).

## Technology stack

Python 3.12 · scikit-learn (TF-IDF, LogisticRegression, LinearSVC, TruncatedSVD) · PyTorch + Hugging Face Transformers (DistilBERT) · NumPy · MCP Python SDK 2.x (`MCPServer`, stdio) · pydantic (schema validation) · pytest · optional: sentence-transformers, pyarrow / huggingface_hub (metadata lookup).

## Licensing and acknowledgements

- **Code:** the original source code of this project is released under the [MIT License](LICENSE), copyright (c) 2026 Alisher Izmukhan. The licence covers only this project's own code and documentation; it grants no rights to the dataset, the pretrained model or third-party libraries listed below, which remain under their own terms.
- **Dataset:** Amazon Reviews 2023 by the McAuley Lab (Hou, Y., Li, J., He, Z., Yan, A., Chen, X., McAuley, J., *Bridging Language and Items for Retrieval and Recommendation*, arXiv:2403.03952, 2024). No explicit licence is stated by the dataset providers; it is used here for non-commercial research and is not redistributed, and it is not covered by this repository's MIT License. The documentation quotes a few short review excerpts as examples of tool output.
- **Pretrained model:** [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) (Sanh et al., 2019; Apache-2.0), fine-tuned in this project. The fine-tuned weights are not included in the repository.
- **Libraries:** scikit-learn, PyTorch, Hugging Face Transformers, the MCP Python SDK and the others listed in `pyproject.toml`, under their own licences.
- The project was originally developed as a dissertation project, *"Development of an Intelligent User Review Analysis System Using Natural Language Processing and Machine Learning Methods"*.

## License

The original code of this project is licensed under the MIT License; see [LICENSE](LICENSE). Datasets, pretrained models and third-party libraries are not covered by it (see [Licensing and acknowledgements](#licensing-and-acknowledgements)).
