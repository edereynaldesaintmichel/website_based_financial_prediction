#!/usr/bin/env python3
import sys
import os
import torch

# Fix for PyTorch 2.6+ weights_only default change
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# Ensure HuggingFace uses cache and doesn't re-download
os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE', '0')  # Set to '1' to force offline mode

import fitz
from ultralyticsplus import YOLO
from PIL import Image


def load_model():
    """Load the YOLO model (uses cached weights after first download)"""
    model_name = "keremberke/yolov8m-table-extraction"
    
    print(f"Loading model '{model_name}'...")
    print(f"  Cache location: {os.environ.get('HF_HOME', '~/.cache/huggingface')}")
    
    model = YOLO(model_name)
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    
    print("  Model ready!\n")
    return model


def detect_tables(page, model, render_scale=2.0):
    """
    Detect tables on a page.
    Returns list of (rect, confidence) tuples in page coordinates.
    """
    # Render page at higher resolution for better detection
    pix = page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Run YOLO detection
    results = model.predict(img, save=False, verbose=False)
    
    # Convert to page coordinates
    scale_x = page.rect.width / pix.width
    scale_y = page.rect.height / pix.height
    
    tables = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        
        rect = fitz.Rect(
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y
        )
        tables.append((rect, conf))
    
    return tables


def calculate_coverage(tables, page_rect):
    """Calculate total table area as percentage of page area"""
    if not tables:
        return 0.0
    
    total_table_area = sum(rect.width * rect.height for rect, _ in tables)
    page_area = page_rect.width * page_rect.height
    
    return (total_table_area / page_area * 100) if page_area > 0 else 0.0


def highlight_tables(page, tables):
    """Draw semi-transparent highlights over detected tables"""
    for rect, conf in tables:
        # Draw highlighted rectangle
        shape = page.new_shape()
        shape.draw_rect(rect)
        shape.finish(
            color=(1, 0, 0),       # Red border
            fill=(1, 1, 0),        # Yellow fill
            fill_opacity=0.3,      # Semi-transparent
            width=2,
        )
        shape.commit()
        
        # Add confidence badge in corner
        badge_text = f"{conf:.0%}"
        badge_fontsize = 9
        badge_x = rect.x0 + 3
        badge_y = rect.y0 + badge_fontsize + 3
        
        # Badge background
        badge_width = len(badge_text) * badge_fontsize * 0.6
        badge_rect = fitz.Rect(
            badge_x - 2,
            badge_y - badge_fontsize - 1,
            badge_x + badge_width + 2,
            badge_y + 3
        )
        shape = page.new_shape()
        shape.draw_rect(badge_rect)
        shape.finish(color=None, fill=(1, 0, 0), fill_opacity=0.85)
        shape.commit()
        
        # Badge text
        page.insert_text(
            point=(badge_x, badge_y),
            text=badge_text,
            fontsize=badge_fontsize,
            fontname="helv",
            color=(1, 1, 1),  # White text
        )


def add_coverage_banner(page, coverage_pct, num_tables):
    """Add a banner at the top showing coverage stats"""
    text = f"Tables: {num_tables} | Coverage: {coverage_pct:.1f}%"
    
    font_size = 18
    text_width = len(text) * font_size * 0.45
    x_pos = (page.rect.width - text_width) / 2
    y_pos = 22
    
    # Banner background
    padding_x = 15
    padding_y = 6
    banner_rect = fitz.Rect(
        x_pos - padding_x,
        y_pos - font_size - padding_y + 5,
        x_pos + text_width + padding_x,
        y_pos + padding_y
    )
    
    shape = page.new_shape()
    shape.draw_rect(banner_rect)
    shape.finish(
        color=(0.7, 0, 0),
        fill=(1, 1, 1),
        fill_opacity=0.95,
        width=1.5
    )
    shape.commit()
    
    # Banner text
    page.insert_text(
        point=(x_pos, y_pos),
        text=text,
        fontsize=font_size,
        fontname="helv",
        color=(0.7, 0, 0),
    )


if __name__ == "__main__":
    pdf_path = "/Users/eloireynal/Downloads/companies_house_document.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    # Load model (cached after first download)
    model = load_model()
    
    # Open PDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Processing {total_pages} page(s)...\n")
    
    total_coverage = 0.0
    total_tables = 0
    
    for page_idx in range(total_pages):
        page = doc[page_idx]
        
        # Detect tables
        tables = detect_tables(page, model)
        coverage = calculate_coverage(tables, page.rect)
        
        total_coverage += coverage
        total_tables += len(tables)
        
        # Print progress
        print(f"  Page {page_idx + 1:3d}: {len(tables):2d} table(s), {coverage:5.1f}% coverage")
        sys.stdout.flush()
        
        # Annotate page
        highlight_tables(page, tables)
        add_coverage_banner(page, coverage, len(tables))
    
    # Save output
    output_path = pdf_path.replace(".pdf", "_annotated.pdf")
    doc.save(output_path)
    doc.close()
    
    # Summary
    avg_coverage = total_coverage / total_pages if total_pages > 0 else 0
    print(f"\n{'='*50}")
    print(f"  Total tables found: {total_tables}")
    print(f"  Average coverage:   {avg_coverage:.1f}%")
    print(f"{'='*50}")
    print(f"\n✓ Annotated PDF saved to:\n  {output_path}")