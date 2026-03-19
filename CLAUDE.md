# Website-Based Financial Prediction

Predicts financial metrics (e.g. revenue, employee count) of UK companies from their website content, using ML models trained on structured financial filings (XBRL) as ground truth.

## Environment

Always activate the local venv before running Python scripts or installing pip packages:
```
source env/bin/activate
```

## Directory Structure

- `common_crawl_pipeline/` — Fetches and processes company website HTML from Common Crawl.
- `html_sanitization_pipeline/` — Cleans, re-indents, and tags numbers in raw HTML.
- `xbrl_parsing_pipeline/` — Parses XBRL filings (Companies House) to extract ground-truth financials.
- `cc_data/` — Cached Common Crawl data: raw HTML, converted markdown, and stats.
- `output/` — Enriched JSON datasets with company info and financials.
- `checkpoints/` — Saved model checkpoints.
- `tests/` — Ad-hoc test and analysis scripts.

## Key Root Files

- `financial_bert.py` / `hierarchical_encoder.py` — Model architectures.
- `financial_dataset.py` — Dataset loading and preprocessing.
- `train_financial.py` / `validate_financial.py` — Training and validation scripts.
- `train_arithmetic.py` — Auxiliary arithmetic pre-training task.

