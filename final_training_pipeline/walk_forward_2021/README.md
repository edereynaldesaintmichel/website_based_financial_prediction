# Walk-forward 2021 data pipeline

A parallel, time-aware pipeline that sits alongside the existing 2018→2025 flow
without overwriting any of its outputs. It produces two disjoint folds:

- **Train fold:** 2017/18 base filings → growth to 2021
- **Test fold:**  2021 base filings → growth to 2025

## Outputs produced here

After running every step below, this directory holds:

- `growth_rates_2018_to_2021.json` — labels for the train fold (UK + SEC merged)
- `growth_rates_2021_to_2025.json` — labels for the test fold  (UK + SEC merged)

The matching tokenized documents go to the existing dataset layout:

- `training_data/processed/SEC_10k_markdown_tagged/`       (2018 SEC, existing)
- `training_data/processed/companies_house_markdown_tagged/` (2017/18 UK, existing)
- `training_data/processed/SEC_10k_markdown_tagged_2021/`       (new)
- `training_data/processed/companies_house_markdown_tagged_2021/` (new)

## Step-by-step

### 1. Growth rates (CPU-only, runnable locally)

Already parameterized — see each script's `--year` / `--early-dir` / `--late-dir`:

```bash
source env/bin/activate

# SEC: parse 2021 DERA quarterly ZIPs into per-company JSON
python SEC_pipeline/parse_financials.py \
    --year 2021 \
    --output-dir SEC_pipeline/output/financials_json_2021

# SEC: growth rates for both windows
python SEC_pipeline/compute_growth_rates.py \
    --early-dir SEC_pipeline/output/financials_json_2018 \
    --late-dir  SEC_pipeline/output/financials_json_2021 \
    --output SEC_pipeline/growth_scores_2018_to_2021.json \
    --growth-rates-output SEC_pipeline/annual_growth_rates_2018_to_2021.json

python SEC_pipeline/compute_growth_rates.py \
    --early-dir SEC_pipeline/output/financials_json_2021 \
    --late-dir  SEC_pipeline/output/financials_json_2025 \
    --output SEC_pipeline/growth_scores_2021_to_2025.json \
    --growth-rates-output SEC_pipeline/annual_growth_rates_2021_to_2025.json

# UK: parse 2021 monthly iXBRL archives for reference (2018-baseline) companies.
# The --keep-html-dir flag preserves the matched raw HTMLs in a single pass so
# they can later be OCR'd for the test fold's training data.
python companies_house_pipeline/xbrl_parsing_pipeline/xbrl_parser_no_survivorship_bias.py \
    --year 2021 \
    --reference-dir /Users/eloireynal/Downloads/json_50plus_employees \
    --output-dir companies_house_pipeline/xbrl_parsing_pipeline/output \
    --keep-html-dir training_data/raw/companies_house_html_2021

# UK: growth rates for both windows (same compute_growth_scores.py, different dirs)
python companies_house_pipeline/assess_company_growth/compute_growth_scores.py \
    --dir-2018 /Users/eloireynal/Downloads/json_50plus_employees \
    --dir-2025 companies_house_pipeline/xbrl_parsing_pipeline/output/json_reference_companies_2021 \
    --fields companies_house_pipeline/assess_company_growth/common_fields.json \
    --output companies_house_pipeline/assess_company_growth/growth_scores_2018_to_2021.json \
    --growth-rates-output companies_house_pipeline/assess_company_growth/annual_growth_rates_2018_to_2021.json

python companies_house_pipeline/assess_company_growth/compute_growth_scores.py \
    --dir-2018 companies_house_pipeline/xbrl_parsing_pipeline/output/json_reference_companies_2021 \
    --dir-2025 companies_house_pipeline/xbrl_parsing_pipeline/output/json_reference_companies \
    --fields companies_house_pipeline/assess_company_growth/common_fields.json \
    --output companies_house_pipeline/assess_company_growth/growth_scores_2021_to_2025.json \
    --growth-rates-output companies_house_pipeline/assess_company_growth/annual_growth_rates_2021_to_2025.json

# Merge UK + SEC for each window
python final_training_pipeline/walk_forward_2021/prepare_growth_rates.py
```

### 2. Raw HTML for 2021 base filings

Already fetched by the steps above:

- **SEC 2021 10-Ks:** `SEC_pipeline/output/10k_html_raw_2021/`
  - Produced by `fetch_10k_index.py --year 2021` then `download_10k_html.py --year 2021`.
- **UK 2021 iXBRL HTMLs:** `training_data/raw/companies_house_html_2021/`
  - Populated by the `--keep-html-dir` flag on `xbrl_parser_no_survivorship_bias.py`.

### 3. Sanitize / OCR the raw HTML

SEC 10-Ks are clean HTML, so the Playwright-based `SEC_pipeline/sanitize_html.py`
is enough — no GPU needed, runs locally in ~15–20 min:

```bash
python SEC_pipeline/sanitize_html.py \
    --input  SEC_pipeline/output/10k_html_raw_2021 \
    --output SEC_pipeline/output/10k_markdown_2021
```

UK iXBRL filings are the messy ones, so they still go through GLM-OCR on a
rented GPU. Zip the UK raw-HTML dir and transfer it to a vast.ai GLM-OCR node:

```bash
# On your machine:
cd training_data/raw
zip -r companies_house_html_2021.zip companies_house_html_2021

# On the GPU node (after setup_companies_house_ocr.sh):
python glm_ocr_pipeline/convert.py companies_house_html_2021.zip \
    --output companies_house_html_2021_cleaned_up.jsonl
```

Bring the JSONL back.

### 4. Tag numbers and land in training_data/processed/*_2021/

SEC 2021 — the same regex the existing 2018 corpus was tagged with. Point
`mlm_training_pipeline/tag_numbers.py` at the sanitized directory and it tags
every `.md` in one shot:

```bash
python mlm_training_pipeline/tag_numbers.py \
    SEC_pipeline/output/10k_markdown_2021 \
    -o training_data/processed/SEC_10k_markdown_tagged_2021
```

UK 2021 — expand the GLM-OCR JSONL into one `.md` per filing, then tag with
the same script:

```bash
python -c "
import json
from pathlib import Path
out = Path('SEC_pipeline/output/companies_house_markdown_2021')
out.mkdir(parents=True, exist_ok=True)
for line in Path('companies_house_html_2021_cleaned_up.jsonl').open():
    rec = json.loads(line)
    (out / f\"{rec['stem']}.md\").write_text(rec['markdown'], encoding='utf-8')
"
python mlm_training_pipeline/tag_numbers.py \
    SEC_pipeline/output/companies_house_markdown_2021 \
    -o training_data/processed/companies_house_markdown_tagged_2021
```

### 5. Training with walk-forward evaluation

The existing `final_training_pipeline/train.py` already takes `--data` and
`--growth-rates` as CLI args. Train on the 2018 fold:

```bash
python final_training_pipeline/train.py \
    --data mlm_data/documents_base2018.pt \
    --growth-rates final_training_pipeline/walk_forward_2021/growth_rates_2018_to_2021.json \
    --output-dir checkpoints/wf_2021_train
```

Then evaluate on the 2021 fold with the same checkpoint.
The MD5-hash train/val split inside `split_utils.py` still kicks in on the
`--data` file, so pass `--val-fraction 0` (or similar) when you want the entire
2018 corpus to train and the 2021 corpus to serve as the only held-out
evaluation.
