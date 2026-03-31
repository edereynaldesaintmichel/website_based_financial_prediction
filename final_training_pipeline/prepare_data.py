"""Aggregate UK and SEC growth-rate files into a single JSON."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Merge UK and SEC annual growth rates.")
    parser.add_argument(
        "--uk-rates",
        default=os.path.join(
            os.path.dirname(__file__),
            "..", "companies_house_pipeline", "assess_company_growth", "annual_growth_rates.json",
        ),
        help="Path to UK annual_growth_rates.json",
    )
    parser.add_argument(
        "--sec-rates",
        default=os.path.join(
            os.path.dirname(__file__),
            "..", "SEC_pipeline", "annual_growth_rates.json",
        ),
        help="Path to SEC annual_growth_rates.json",
    )
    parser.add_argument(
        "--output",
        default="growth_rates.json",
        help="Output path (default: growth_rates.json)",
    )
    args = parser.parse_args()

    with open(args.uk_rates) as f:
        uk = json.load(f)
    with open(args.sec_rates) as f:
        sec = json.load(f)

    merged = {**uk, **sec}  # SEC overwrites on (unlikely) overlap

    with open(args.output, "w") as f:
        json.dump(merged, f)

    print(f"UK:  {len(uk):>6,}")
    print(f"SEC: {len(sec):>6,}")
    print(f"Total: {len(merged):>4,}")


if __name__ == "__main__":
    main()
