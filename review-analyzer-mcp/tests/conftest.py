"""Make the MCP tests deterministic and independent of local training runs.

Set before ``server`` is imported: the v1 tests expect the 5-review synthetic
corpus (REVIEW_ANALYZER_DATA=demo) and the keyword baseline, both explicitly
configured, never a silent fallback. Tests that need a trained model build one from the fixture.
"""

import os

os.environ["REVIEW_ANALYZER_SENTIMENT_BACKEND"] = "keyword"
os.environ["REVIEW_ANALYZER_DATA"] = "demo"  # explicit demo mode; the default is the Amazon corpus
os.environ.pop("REVIEW_ANALYZER_MODEL_DIR", None)
os.environ["REVIEW_ANALYZER_RETRIEVER"] = "lsa"
