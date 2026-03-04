"""
OpenRouter: Further sanitize HTML financial filings.

Takes the pre-sanitized HTML files (from sanitize_html.py) and uses
GPT-OSS-120B via OpenRouter to:
  - Remove remaining boilerplate / non-financial content
  - Clean up formatting artefacts
  - Produce well-structured, minimal Markdown

Providers: DeepInfra (deepinfra/fp4) primary, AtlasCloud (atlas-cloud/fp8) fallback.
Up to 50 concurrent requests (no batch API needed).

Usage:
    python gemini_sanitize_html.py [input_dir] [output_dir]
"""

import asyncio
import os
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

INPUT_DIR = "/Users/eloireynal/Downloads/html_50plus_employees_sanitized"
OUTPUT_DIR = "/Users/eloireynal/Downloads/html_50plus_employees_gemini_sanitized"

OPENROUTER_MODEL = "openai/gpt-oss-120b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Provider routing: DeepInfra fp4 first, AtlasCloud fp8 as fallback
PROVIDER_ORDER = ["deepinfra/fp4", "atlas-cloud/fp8"]

# Reasoning effort
REASONING_EFFORT = "medium"

# Max concurrent requests
MAX_CONCURRENT = 150

SYSTEM_INSTRUCTION = """Format this whole document in markdown. Output it all in one shot.
Remove all bullshit, boilerplate and CYA, but strictly preserve all percentages, interest rates, and primary financial statement tables. Preserve non-bullshit text too.

Remove the "Notes" column from every table you ever come across.

Remove the table of contents.
Make sure the markdown tables are properly formatted.
"""


# ──────────────────────────────────────────────────────────────
# Client setup
# ──────────────────────────────────────────────────────────────

def load_api_key() -> str:
    """Load OPENROUTER_API_KEY from environment or .env file."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()

    # Fall back to parsing .env from the project root or cwd
    for env_path in [Path(__file__).parent.parent / ".env", Path.cwd() / ".env"]:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY"):
                    _, _, val = line.partition("=")
                    return val.strip().strip('"').strip("'")

    raise RuntimeError(
        "OPENROUTER_API_KEY not found. Set it in your environment or .env file."
    )


def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=load_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ──────────────────────────────────────────────────────────────
# Core: process a single file
# ──────────────────────────────────────────────────────────────

async def process_file(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    html_path: Path,
    output_dir: Path,
) -> tuple[str, bool, str]:
    """Process one HTML file. Returns (output_filename, success, error_msg)."""
    filename = html_path.stem + ".md"
    html_content = read_file(str(html_path))

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": html_content},
                ],
                extra_body={
                    "provider": {
                        "order": PROVIDER_ORDER,
                        "allow_fallbacks": True,
                    },
                    "reasoning": {
                        "effort": REASONING_EFFORT,
                    },
                },
            )
            text = response.choices[0].message.content or ""
            write_file(str(output_dir / filename), text)
            return filename, True, ""
        except Exception as exc:
            return filename, False, str(exc)


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

async def main_async(input_dir: str, output_dir: str) -> None:
    html_files = sorted(Path(input_dir).glob("*.html"))
    print(f"📂 Found {len(html_files)} HTML files in {input_dir}")

    if not html_files:
        print("No HTML files found!")
        sys.exit(1)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    already_done = {p.name for p in out_path.iterdir() if p.is_file()}

    pending = [f for f in html_files if (f.stem + ".md") not in already_done]
    skipped = len(html_files) - len(pending)

    if skipped:
        print(f"⏭️  Skipped (already done): {skipped}")
    print(f"🔄 To process: {len(pending)}")
    print()

    if not pending:
        print("✅ Nothing to process — all files already have output!")
        return

    client = get_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [process_file(client, semaphore, p, out_path) for p in pending]

    saved = 0
    errors = 0
    total = len(pending)
    start_time = time.time()

    for coro in asyncio.as_completed(tasks):
        filename, success, error_msg = await coro
        done = saved + errors + 1
        if success:
            saved += 1
            print(f"  ✅ [{done}/{total}] {filename}")
        else:
            errors += 1
            print(f"  ❌ [{done}/{total}] {filename}: {error_msg}")

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"\n{'=' * 60}")
    print(f"🏁 Done in {elapsed}")
    print(f"✅ Saved: {saved}")
    if errors:
        print(f"⚠️  Errors: {errors}")


def main():
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    print("=" * 60)
    print("🧹 OpenRouter HTML Sanitizer")
    print("=" * 60)
    print(f"📂 Input:       {input_dir}")
    print(f"📂 Output:      {output_dir}")
    print(f"🤖 Model:       {OPENROUTER_MODEL}")
    print(f"🔀 Providers:   {' → '.join(PROVIDER_ORDER)}")
    print(f"🧠 Reasoning:   {REASONING_EFFORT}")
    print(f"⚡ Concurrency: {MAX_CONCURRENT}")
    print()

    asyncio.run(main_async(input_dir, output_dir))


if __name__ == "__main__":
    main()
