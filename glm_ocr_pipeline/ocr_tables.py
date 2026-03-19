#!/usr/bin/env python3
"""
Phase 2: OCR table screenshots via vLLM (runs on GPU server).

Reads PNG table images from a zip file (loaded entirely into RAM),
sends them to a vLLM server running GLM-OCR, and saves the recognized
table content as a single JSONL file.

Input:  zip file containing {stem}__t{i}.png files
Output: {zip_parent}/{zip_stem}_ocred.jsonl  — one JSON line per table:
        {"stem": "...", "html": "..."}

Usage:
    python ocr_tables.py <tables.zip> [--port 8000]
"""

import argparse
import asyncio
import base64
import io
import json
import sys
import zipfile
from pathlib import Path

import aiofiles
import aiohttp
from tqdm import tqdm

SERVED_MODEL_NAME = "glm-ocr"


async def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: OCR table images via vLLM"
    )
    parser.add_argument("input", help="Zip file of table PNG screenshots")
    parser.add_argument("--output", help="Output JSONL file (default: {input_stem}_ocred.jsonl)")
    parser.add_argument("--port", type=int, default=8001, help="vLLM server port")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file() or input_path.suffix.lower() != ".zip":
        print(f"Error: {input_path} is not a zip file")
        sys.exit(1)

    # Read entire zip into RAM
    print(f"Loading {input_path} into memory...")
    zip_bytes = input_path.read_bytes()
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    print(f"Loaded {len(zip_bytes) / 1e9:.2f} GB into RAM")

    # Collect PNG entries (skip directories and non-png files)
    all_png_names = sorted(
        name for name in zf.namelist()
        if name.lower().endswith(".png") and not name.endswith("/")
        and not name.startswith("__MACOSX") and "/._" not in name
    )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_ocred.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check which ones are already done by reading existing JSONL
    existing = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing.add(json.loads(line)["stem"])
                    except (json.JSONDecodeError, KeyError):
                        pass

    todo = [
        name for name in all_png_names
        if Path(name).stem not in existing
    ]

    print(f"Input:  {input_path.name} ({len(all_png_names)} images, {len(existing)} already done)")
    print(f"Output: {output_path}")
    print(f"To process: {len(todo)}")

    if not todo:
        print("All done!")
        return

    # Cap in-flight requests to limit client-side memory (each payload holds a full
    # base64 PNG). vLLM handles batching/scheduling on its own.
    sem = asyncio.Semaphore(1024)
    write_lock = asyncio.Lock()
    n_done = 0
    n_errors = 0

    async with aiofiles.open(output_path, "a", encoding="utf-8") as out_f:
        async with aiohttp.ClientSession() as session:
            pbar = tqdm(total=len(todo), desc="OCR", unit="table")

            async def ocr_one(zip_entry_name: str):
                nonlocal n_done, n_errors
                img_bytes = zf.read(zip_entry_name)
                b64 = base64.b64encode(img_bytes).decode()
                stem = Path(zip_entry_name).stem
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
                            timeout=aiohttp.ClientTimeout(total=600),
                        ) as resp:
                            if resp.status != 200:
                                err_text = await resp.text()
                                tqdm.write(f"  vLLM {resp.status} for {stem}: {err_text[:300]}")
                                n_errors += 1
                                return
                            data = await resp.json()
                            content = data["choices"][0]["message"]["content"]
                    except Exception as e:
                        tqdm.write(f"  Error {stem}: {type(e).__name__}: {e}")
                        n_errors += 1
                        return
                    finally:
                        pbar.update(1)

                async with write_lock:
                    await out_f.write(json.dumps({"stem": stem, "html": content}, ensure_ascii=False) + "\n")
                n_done += 1

            await asyncio.gather(*[ocr_one(name) for name in todo])
            pbar.close()

    print(f"\nDone! {n_done} tables OCR'd, {n_errors} errors")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
