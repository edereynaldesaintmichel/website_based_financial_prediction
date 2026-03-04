# xbrl_parsing_pipeline

Local, multithreaded iXBRL parser for UK Companies House 2025 monthly filing data.

## What it does

1. Downloads each monthly zip sequentially (one at a time to keep disk usage low)
2. Extracts it to a temporary work directory
3. **Immediately deletes the zip** after extraction
4. Parses all `.html` iXBRL files **in parallel** across your CPU cores
5. Saves a JSON file for each company with ≥ 50 employees
6. **Immediately deletes extracted files** after that month's parsing is complete
7. Writes an intermediate `sorted_by_employees.json` after every month (crash-safe)

Net peak disk usage ≈ **size of one uncompressed monthly archive** (a few GB at most).

## Setup

```bash
cd xbrl_parsing_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Default run (auto-detects CPU count, threshold = 50 employees)
python xbrl_parser.py

# Custom options
python xbrl_parser.py \
    --workers 10 \
    --threshold 100 \
    --output-dir ~/Desktop/ch_output

# Resume from a specific month (e.g. start from April = month 4)
python xbrl_parser.py --start-from 4
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--workers` | CPU count − 1 | Number of parallel file-parsing threads |
| `--threshold` | `50` | Minimum employee count to keep a filing |
| `--output-dir` | `./output` | Where JSON files and the ranking are saved |
| `--start-from` | `1` | Skip the first N-1 months (useful to resume after a crash) |

## Output structure

```
output/
├── json_50plus_employees/        ← one JSON per qualifying filing
│   ├── CH12345678_parsed.json
│   └── ...
├── sorted_by_employees.json      ← all qualifying companies, sorted by headcount
└── run.log                       ← full run log
```

## Threading model

- **Outer loop**: sequential per zip (controls disk usage — only one archive lives on disk at a time)
- **Inner pool**: `ThreadPoolExecutor(max_workers=N)` parallelises parsing of individual HTML files within each zip
- `lxml` releases the GIL during XML parsing, so threads genuinely run in parallel on multiple cores
