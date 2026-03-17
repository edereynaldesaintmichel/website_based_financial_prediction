#!/usr/bin/env python3
"""
Phase 2: OCR table screenshots via vLLM (runs on GPU server).

Reads PNG table images, sends them to a vLLM server running GLM-OCR,
and saves the recognized table content as .html files.

Input:  directory of {stem}__t{i}.png files
Output: {input}_cleaned_up/ with {stem}__t{i}.html files (same filenames, .html ext)

Usage:
    python ocr_tables.py <tables_dir> [--port 8000]
"""

import argparse
import asyncio
import base64
import sys
from pathlib import Path

import aiohttp
from tqdm import tqdm

SERVED_MODEL_NAME = "glm-ocr"


async def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: OCR table images via vLLM"
    )
    parser.add_argument("input", help="Directory of table PNG screenshots")
    parser.add_argument("--output", help="Output directory (default: {input}_cleaned_up)")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_cleaned_up"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(input_dir.glob("*.png"))
    existing = {p.stem for p in output_dir.glob("*.html")}
    todo = [img for img in images if img.stem not in existing]

    print(f"Input:  {input_dir} ({len(images)} images, {len(existing)} already done)")
    print(f"Output: {output_dir}")
    print(f"To process: {len(todo)}")

    if not todo:
        print("All done!")
        return

    # Cap in-flight requests to limit client-side memory (each payload holds a full
    # base64 PNG). vLLM handles batching/scheduling on its own.
    sem = asyncio.Semaphore(512)
    n_done = 0
    n_errors = 0

    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=len(todo), desc="OCR", unit="table")

        async def ocr_one(img_path: Path):
            nonlocal n_done, n_errors
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            payload = {
                "model": SERVED_MODEL_NAME,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": "Table Recognition:"},
                    ],
                }],
                "max_tokens": 8192,
                "temperature": 0.1,
            }
            async with sem:
                try:
                    async with session.post(
                        f"http://localhost:{args.port}/v1/chat/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=180),
                    ) as resp:
                        if resp.status != 200:
                            tqdm.write(f"  vLLM {resp.status} for {img_path.name}")
                            n_errors += 1
                            return
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        (output_dir / f"{img_path.stem}.html").write_text(
                            content, encoding="utf-8",
                        )
                        n_done += 1
                except Exception as e:
                    tqdm.write(f"  Error {img_path.name}: {e}")
                    n_errors += 1
                finally:
                    pbar.update(1)

        await asyncio.gather(*[ocr_one(img) for img in todo])
        pbar.close()

    print(f"\nDone! {n_done} tables OCR'd, {n_errors} errors")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
