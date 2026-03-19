import random
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

JS_SANITIZE = open("tests/sanitize_sec_tables.js").read()
URL = "file:///Users/eloireynal/Documents/My%20projects/website_based_financial_prediction/SEC_pipeline/output/10k_html_raw/1584509_2018-11-21.html"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(URL)

    # Run sanitization and measure time
    elapsed = page.evaluate(f"(() => {{ const t0 = performance.now(); {JS_SANITIZE}; return performance.now() - t0; }})()")
    print(f"Sanitization took {elapsed:.2f}ms")

    # Sample 10 random tables
    tables = page.query_selector_all("table")
    sample = random.sample(range(len(tables)), min(10, len(tables)))
    for i in sorted(sample):
        text = tables[i].inner_text().strip()
        if text:
            print(f"\n{'='*60}")
            print(f"TABLE {i}")
            print(f"{'='*60}")
            print(text[:2000])

    browser.close()
