"""Merge UK + SEC per-window annual growth rates for walk-forward folds.

Produces two JSON files:
    growth_rates_2018_to_2021.json  — train fold labels
    growth_rates_2021_to_2025.json  — test fold labels

Each merges the UK and SEC annual-growth-rate JSONs for that window.

Default input paths match:
    SEC: SEC_pipeline/annual_growth_rates_<early>_to_<late>.json
    UK:  companies_house_pipeline/assess_company_growth/annual_growth_rates_<early>_to_<late>.json
"""

import argparse
import json
import os


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def default_sec_path(early: int, late: int) -> str:
    return os.path.join(
        REPO_ROOT, "SEC_pipeline",
        f"annual_growth_rates_{early}_to_{late}.json",
    )


def default_uk_path(early: int, late: int) -> str:
    return os.path.join(
        REPO_ROOT, "companies_house_pipeline", "assess_company_growth",
        f"annual_growth_rates_{early}_to_{late}.json",
    )


def merge_window(early: int, late: int, uk_path: str, sec_path: str, output: str) -> None:
    with open(uk_path) as f:
        uk = json.load(f)
    with open(sec_path) as f:
        sec = json.load(f)

    merged = {**uk, **sec}  # SEC wins on (unlikely) ID collision

    with open(output, "w") as f:
        json.dump(merged, f)

    print(f"[{early}->{late}] UK: {len(uk):>6,}  SEC: {len(sec):>6,}  Total: {len(merged):>6,}  -> {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--uk-train", default=default_uk_path(2018, 2021))
    parser.add_argument("--sec-train", default=default_sec_path(2018, 2021))
    parser.add_argument("--uk-test", default=default_uk_path(2021, 2025))
    parser.add_argument("--sec-test", default=default_sec_path(2021, 2025))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    merge_window(
        2018, 2021, args.uk_train, args.sec_train,
        os.path.join(args.output_dir, "growth_rates_2018_to_2021.json"),
    )
    merge_window(
        2021, 2025, args.uk_test, args.sec_test,
        os.path.join(args.output_dir, "growth_rates_2021_to_2025.json"),
    )


if __name__ == "__main__":
    main()
