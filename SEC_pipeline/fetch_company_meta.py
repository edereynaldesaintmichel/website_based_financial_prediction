"""
Fetch Company Metadata: Ticker + Website (2018)
=================================================
Builds a CIK → {ticker, website} mapping for all 10-K filers.

Sources (in order):
  1. SEC company_tickers.json  →  CIK-to-ticker mapping
  2. Wikidata SPARQL           →  ticker-to-website (bulk, free)
  3. yfinance                  →  fallback for missing websites
  4. Serper (Google search)    →  fallback for still-missing websites

Output: output/company_meta.json
  {
    "320193": {"ticker": "AAPL", "company_name": "APPLE INC", "website": "https://www.apple.com"},
    ...
  }

Usage:
    pip install requests tqdm yfinance
    python fetch_company_meta.py
"""

import json
import os
import time
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from config import (
    OUTPUT_DIR, INDEX_DIR, YEAR, sec_headers, REQUEST_DELAY,
)

META_OUTPUT = OUTPUT_DIR / "company_meta.json"

# ============================================================================
# Step 1: SEC CIK → ticker mapping
# ============================================================================

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def fetch_cik_ticker_map(headers: dict) -> dict:
    """Download SEC company_tickers.json and return {cik_str: {ticker, company_name}}."""
    resp = requests.get(SEC_TICKERS_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    mapping = {}
    for entry in data.values():
        cik = str(entry["cik_str"])
        mapping[cik] = {
            "ticker": entry["ticker"],
            "company_name": entry["title"],
        }
    return mapping


# ============================================================================
# Step 2: Wikidata SPARQL — bulk ticker → website
# ============================================================================

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

SPARQL_QUERY = """
SELECT ?ticker ?website WHERE {
  ?company wdt:P414 ?exchange .
  VALUES ?exchange { wd:Q13677 wd:Q82059 wd:Q1478358 }
  ?company wdt:P856 ?website .
  ?company wdt:P249 ?ticker .
}
"""
# wd:Q13677 = NYSE, wd:Q82059 = NASDAQ, wd:Q1478358 = NYSE American (AMEX)


def fetch_wikidata_websites() -> dict:
    """Query Wikidata for ticker → website mapping. Returns {TICKER: url}."""
    print("Querying Wikidata SPARQL for company websites ...")
    resp = requests.get(
        WIKIDATA_ENDPOINT,
        params={"query": SPARQL_QUERY, "format": "json"},
        headers={"User-Agent": "SEC_pipeline/1.0 (financial research)"},
        timeout=120,
    )
    resp.raise_for_status()
    results = resp.json()["results"]["bindings"]

    ticker_to_website = {}
    for row in results:
        ticker = row["ticker"]["value"].strip().upper()
        website = row["website"]["value"].strip()
        # Keep the first website found per ticker
        if ticker not in ticker_to_website:
            ticker_to_website[ticker] = website

    print(f"  Wikidata returned websites for {len(ticker_to_website):,} tickers")
    return ticker_to_website


# ============================================================================
# Step 3: yfinance fallback for missing websites
# ============================================================================

def fetch_yfinance_websites(tickers: list[str]) -> dict:
    """Fetch websites from Yahoo Finance for a list of tickers.
    Returns {TICKER: url}."""
    import yfinance as yf

    results = {}
    print(f"Fetching {len(tickers):,} missing websites from Yahoo Finance ...")

    for ticker in tqdm(tickers, desc="yfinance"):
        try:
            info = yf.Ticker(ticker).info
            website = info.get("website")
            if website:
                results[ticker] = website
        except Exception:
            pass
        time.sleep(0.2)  # be polite to Yahoo

    print(f"  yfinance returned websites for {len(results):,} tickers")
    return results


# ============================================================================
# Step 4: Serper (Google search) fallback for still-missing websites
# ============================================================================

SERPER_API_KEY = "55011ddd640ce2dbc6a1f123e29201c49f92f61a"
SERPER_URL = "https://google.serper.dev/search"


def fetch_serper_websites(companies: list[tuple[str, str]]) -> dict:
    """Google-search each company name and extract the domain of the first result.

    Args:
        companies: list of (ticker, company_name) tuples.

    Returns:
        {TICKER: url} for companies where a website was found.
    """
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    results = {}
    print(f"Searching {len(companies):,} missing websites via Serper (Google) ...")

    for ticker, name in tqdm(companies, desc="serper"):
        try:
            resp = requests.post(
                SERPER_URL,
                headers=headers,
                json={"q": f"{name} website"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            organic = data.get("organic", [])
            if organic:
                link = organic[0].get("link", "")
                if link:
                    parsed = urlparse(link)
                    domain = f"{parsed.scheme}://{parsed.netloc}"
                    results[ticker] = domain
        except Exception:
            pass
        time.sleep(0.15)

    print(f"  Serper returned websites for {len(results):,} tickers")
    return results


# ============================================================================
# Main
# ============================================================================

def load_10k_ciks() -> set:
    """Load the set of CIKs that filed 10-K in our target year."""
    index_path = INDEX_DIR / f"10k_filings_{YEAR}.csv"
    if not index_path.exists():
        return set()

    import csv
    with open(index_path, "r", encoding="utf-8") as fh:
        return {row["cik"] for row in csv.DictReader(fh)}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    headers = sec_headers()

    print("=" * 60)
    print("Fetch Company Metadata: Ticker + Website")
    print("=" * 60)

    # 1. CIK → ticker from SEC
    print("\n[1/4] Fetching CIK-to-ticker mapping from SEC ...")
    cik_map = fetch_cik_ticker_map(headers)
    print(f"  SEC lists {len(cik_map):,} companies total")

    # Filter to 10-K filers if index exists
    ten_k_ciks = load_10k_ciks()
    if ten_k_ciks:
        # Keep only companies that filed 10-K
        cik_map = {cik: v for cik, v in cik_map.items() if cik in ten_k_ciks}
        print(f"  Filtered to {len(cik_map):,} companies with 10-K filings in {YEAR}")
    else:
        print(f"  (10-K index not found — keeping all companies. "
              f"Run fetch_10k_index.py first for filtering.)")

    # 2. Wikidata bulk query
    print("\n[2/4] Wikidata SPARQL ...")
    wikidata_websites = fetch_wikidata_websites()

    # Match tickers to websites
    matched = 0
    missing_tickers = []
    for cik, info in cik_map.items():
        ticker = info["ticker"]
        website = wikidata_websites.get(ticker)
        if website:
            info["website"] = website
            matched += 1
        else:
            missing_tickers.append(ticker)

    print(f"  Matched {matched:,} / {len(cik_map):,} companies via Wikidata")

    # 3. yfinance fallback
    if missing_tickers:
        print(f"\n[3/4] yfinance fallback for {len(missing_tickers):,} remaining ...")
        yf_websites = fetch_yfinance_websites(missing_tickers)

        yf_matched = 0
        for cik, info in cik_map.items():
            if "website" not in info:
                website = yf_websites.get(info["ticker"])
                if website:
                    info["website"] = website
                    yf_matched += 1

        print(f"  Matched {yf_matched:,} more via yfinance")
    else:
        print("\n[3/4] No missing tickers — skipping yfinance")

    # 4. Serper fallback
    still_missing = [
        (info["ticker"], info["company_name"])
        for info in cik_map.values()
        if "website" not in info
    ]
    if still_missing:
        print(f"\n[4/4] Serper fallback for {len(still_missing):,} remaining ...")
        serper_websites = fetch_serper_websites(still_missing)

        serper_matched = 0
        for cik, info in cik_map.items():
            if "website" not in info:
                website = serper_websites.get(info["ticker"])
                if website:
                    info["website"] = website
                    serper_matched += 1

        print(f"  Matched {serper_matched:,} more via Serper")
    else:
        print("\n[4/4] No missing tickers — skipping Serper")

    # Save
    with open(META_OUTPUT, "w", encoding="utf-8") as fh:
        json.dump(cik_map, fh, indent=2, ensure_ascii=False)

    total_with_website = sum(1 for v in cik_map.values() if "website" in v)
    total_without = len(cik_map) - total_with_website

    print(f"\n{'=' * 60}")
    print(f"Done! Saved {len(cik_map):,} companies to {META_OUTPUT}")
    print(f"  With website:    {total_with_website:,}")
    print(f"  Without website: {total_without:,}")
    print(f"{'=' * 60}")


def enrich_serper():
    """Load existing company_meta.json and use Serper to fill in missing websites."""
    if not META_OUTPUT.exists():
        print(f"Error: {META_OUTPUT} not found. Run the full pipeline first.")
        return

    with open(META_OUTPUT, "r", encoding="utf-8") as fh:
        cik_map = json.load(fh)

    missing = [
        (info["ticker"], info["company_name"])
        for info in cik_map.values()
        if "website" not in info
    ]

    if not missing:
        print("All companies already have websites!")
        return

    print(f"Found {len(missing):,} companies without websites. Searching via Serper ...")
    serper_websites = fetch_serper_websites(missing)

    matched = 0
    for cik, info in cik_map.items():
        if "website" not in info:
            website = serper_websites.get(info["ticker"])
            if website:
                info["website"] = website
                matched += 1

    with open(META_OUTPUT, "w", encoding="utf-8") as fh:
        json.dump(cik_map, fh, indent=2, ensure_ascii=False)

    total_with = sum(1 for v in cik_map.values() if "website" in v)
    total_without = len(cik_map) - total_with

    print(f"\nDone! Matched {matched:,} more via Serper")
    print(f"  With website:    {total_with:,}")
    print(f"  Without website: {total_without:,}")


if __name__ == "__main__":
    import sys
    if "--enrich-serper" in sys.argv:
        enrich_serper()
    else:
        main()
