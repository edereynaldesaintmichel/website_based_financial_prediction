#!/usr/bin/env python3
"""
Parse raw HTML files per domain: convert to markdown, strip repeated
headers/footers via git diff, concatenate, and compute combined statistics.

For each domain directory in cc_data/raw_html/<domain>/:
  1. Convert each HTML → Markdown (html2text).
  2. Sort pages by URL depth then alphabetically.
  3. Git-diff each page against its parent-path page to strip shared
     headers/footers.
  4. Concatenate all pages with ########## separators.
  5. Compute per-page structural stats and combine across the domain.

Output:
  cc_data/markdown/<domain>.md   — concatenated, deduped markdown
  cc_data/stats/<domain>.json    — combined HTML statistics

Usage:
  python parse_html.py \\
      [--input-dir cc_data/raw_html] \\
      [--markdown-dir cc_data/markdown] \\
      [--stats-dir cc_data/stats] \\
      [--workers 4]
"""

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import html2text
from bs4 import BeautifulSoup, Comment, NavigableString, Tag, XMLParsedAsHTMLWarning
from tqdm import tqdm

import warnings

from utils import url_depth, url_to_slug

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Tag sets ──────────────────────────────────────────────────────────────────

SEMANTIC_TAGS = {"article", "section", "nav", "header", "footer", "main", "aside", "figure"}
TABLE_TAGS    = {"table", "tr", "td", "th"}
LIST_TAGS     = {"ul", "ol", "li"}
HEADING_TAGS  = {"h1", "h2", "h3", "h4", "h5", "h6"}


# ── DOM traversal helpers ─────────────────────────────────────────────────────

def iter_nodes(root: Tag):
    """Iterative DFS over all BS4 nodes, yielding (node, depth)."""
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        yield node, depth
        if isinstance(node, Tag):
            # Reverse so document order is preserved when popping from the right
            for child in reversed(list(node.children)):
                stack.append((child, depth + 1))


def max_nested_list_depth(nav_el: Tag) -> int:
    """Max nesting level of <ul>/<ol> elements inside a <nav> element."""
    max_depth = 0
    stack = [(nav_el, 0)]
    while stack:
        el, depth = stack.pop()
        if isinstance(el, Tag):
            if el.name in ("ul", "ol"):
                depth += 1
                max_depth = max(max_depth, depth)
            for child in el.children:
                if isinstance(child, Tag):
                    stack.append((child, depth))
    return max_depth


# ── Base-domain detection ─────────────────────────────────────────────────────

def get_base_domain(soup: BeautifulSoup) -> str | None:
    """Infer the page's own domain from canonical / og:url / base tags."""
    for link in soup.find_all("link"):
        rel = link.get("rel", [])
        if isinstance(rel, str):
            rel = [rel]
        if "canonical" in [r.lower() for r in rel] and link.get("href"):
            netloc = urlparse(link["href"]).netloc
            if netloc:
                return netloc

    og = soup.find("meta", property="og:url")
    if og and og.get("content"):
        netloc = urlparse(og["content"]).netloc
        if netloc:
            return netloc

    base = soup.find("base", href=True)
    if base:
        netloc = urlparse(base["href"]).netloc
        if netloc:
            return netloc

    return None


# ── Core stats computation ────────────────────────────────────────────────────

def compute_stats(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    # ── Full node traversal ───────────────────────────────────────────────────
    all_nodes = list(iter_nodes(soup))

    element_nodes  = [(n, d) for n, d in all_nodes if isinstance(n, Tag)]
    text_nodes     = [(n, d) for n, d in all_nodes
                      if isinstance(n, NavigableString) and not isinstance(n, Comment)]

    total_node_count    = len(element_nodes) + len(text_nodes)
    total_element_count = len(element_nodes)

    # ── Depth stats (leaf nodes only) ─────────────────────────────────────────
    leaf_depths = [d for _, d in text_nodes]
    for n, d in element_nodes:
        if not list(n.children):
            leaf_depths.append(d)

    max_dom_depth  = max(leaf_depths, default=0)
    mean_dom_depth = sum(leaf_depths) / len(leaf_depths) if leaf_depths else 0.0
    std_dom_depth  = (
        math.sqrt(sum((d - mean_dom_depth) ** 2 for d in leaf_depths) / len(leaf_depths))
        if len(leaf_depths) > 1 else 0.0
    )

    # ── Children counts (element children only — branching factor) ────────────
    children_counts = [
        sum(1 for c in n.children if isinstance(c, Tag))
        for n, _ in element_nodes
    ]
    max_children_count  = max(children_counts, default=0)
    mean_children_count = (
        sum(children_counts) / len(children_counts) if children_counts else 0.0
    )

    # ── Document scale ────────────────────────────────────────────────────────
    document_byte_size = len(html.encode("utf-8", errors="replace"))
    body               = soup.body
    inner_text_length  = len(body.get_text()) if body else len(soup.get_text())
    inner_html_length  = len(str(body)) if body else len(str(soup))
    html_to_text_ratio = (
        inner_html_length / inner_text_length if inner_text_length > 0 else 0.0
    )

    # ── Tag distribution ──────────────────────────────────────────────────────
    tag_names  = [n.name.lower() for n, _ in element_nodes if n.name]
    tag_counts = Counter(tag_names)
    unique_tag_count = len(tag_counts)
    total_tags       = sum(tag_counts.values())

    tag_entropy = 0.0
    if total_tags > 0:
        for count in tag_counts.values():
            p = count / total_tags
            if p > 0:
                tag_entropy -= p * math.log2(p)

    semantic_count   = sum(tag_counts.get(t, 0) for t in SEMANTIC_TAGS)
    semantic_tag_ratio = (
        semantic_count / total_element_count if total_element_count > 0 else 0.0
    )
    table_element_count = sum(tag_counts.get(t, 0) for t in TABLE_TAGS)
    list_element_count  = sum(tag_counts.get(t, 0) for t in LIST_TAGS)
    heading_count       = sum(tag_counts.get(t, 0) for t in HEADING_TAGS)
    heading_distribution = {h: tag_counts.get(h, 0) for h in ["h1","h2","h3","h4","h5","h6"]}

    # ── Interactive complexity ────────────────────────────────────────────────
    form_count   = tag_counts.get("form", 0)
    input_count  = (tag_counts.get("input", 0)
                    + tag_counts.get("select", 0)
                    + tag_counts.get("textarea", 0))
    button_count = tag_counts.get("button", 0) + sum(
        1 for n, _ in element_nodes
        if n.name == "input" and n.get("type", "").lower() == "submit"
    )
    iframe_count = tag_counts.get("iframe", 0)
    total_interactive  = form_count + input_count + button_count + iframe_count
    interactive_density = (
        total_interactive / total_element_count if total_element_count > 0 else 0.0
    )

    # ── Media & resources ─────────────────────────────────────────────────────
    image_count  = tag_counts.get("img", 0)
    svg_count    = tag_counts.get("svg", 0)
    video_count  = tag_counts.get("video", 0)
    canvas_count = tag_counts.get("canvas", 0)
    total_image_byte_size  = None
    image_to_doc_size_ratio = None

    # ── External dependencies ─────────────────────────────────────────────────
    base_domain = get_base_domain(soup)

    script_tags       = soup.find_all("script")
    external_scripts  = [t for t in script_tags if t.get("src")]
    inline_scripts    = [t for t in script_tags if not t.get("src")]
    external_script_count = len(external_scripts)
    inline_script_count   = len(inline_scripts)
    total_script_byte_size = sum(
        len((t.string or "").encode("utf-8")) for t in inline_scripts
    )

    stylesheet_count = 0
    for link in soup.find_all("link"):
        rel = link.get("rel", [])
        if isinstance(rel, str):
            rel = [rel]
        if "stylesheet" in [r.lower() for r in rel]:
            stylesheet_count += 1

    external_resource_urls: list[str] = []
    for t in external_scripts:
        src = t.get("src", "")
        if src.startswith("http"):
            external_resource_urls.append(src)
    for t in soup.find_all("link", href=True):
        href = t["href"]
        if href.startswith("http"):
            external_resource_urls.append(href)
    for t in soup.find_all("iframe", src=True):
        src = t["src"]
        if src.startswith("http"):
            external_resource_urls.append(src)

    external_domains: set[str] = set()
    for url in external_resource_urls:
        netloc = urlparse(url).netloc
        if netloc and (not base_domain or netloc != base_domain):
            external_domains.add(netloc)
    unique_external_domains = len(external_domains)

    # ── Styling signals ───────────────────────────────────────────────────────
    inline_style_count = sum(1 for n, _ in element_nodes if n.get("style"))

    style_content = " ".join(t.string for t in soup.find_all("style") if t.string)
    total_css_byte_size = len(style_content.encode("utf-8"))
    media_query_count   = len(re.findall(r"@media\s", style_content, re.IGNORECASE))
    at_rule_openers = len(re.findall(r"@(?:media|keyframes|supports|document|layer)\s", style_content, re.I))
    css_rule_count  = max(0, style_content.count("{") - at_rule_openers)

    # ── Structured data & SEO ─────────────────────────────────────────────────
    json_ld_tags = soup.find_all("script", type="application/ld+json")
    has_json_ld  = len(json_ld_tags) > 0

    json_ld_types: set[str] = set()
    for tag in json_ld_tags:
        try:
            data = json.loads(tag.string or "")
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    t = item.get("@type")
                    if isinstance(t, list):
                        json_ld_types.update(t)
                    elif t:
                        json_ld_types.add(t)
        except Exception:
            pass
    json_ld_type_count = len(json_ld_types)

    meta_tags            = soup.find_all("meta")
    meta_tag_count       = len(meta_tags)
    open_graph_tag_count = sum(
        1 for m in meta_tags
        if str(m.get("property", "")).lower().startswith("og:")
    )

    canonical_tag_present = False
    for link in soup.find_all("link"):
        rel = link.get("rel", [])
        if isinstance(rel, str):
            rel = [rel]
        if "canonical" in [r.lower() for r in rel]:
            canonical_tag_present = True
            break

    robots_meta = soup.find("meta", attrs={"name": re.compile(r"^robots$", re.I)})
    robots_meta_content = robots_meta.get("content") if robots_meta else None

    # ── Accessibility ─────────────────────────────────────────────────────────
    aria_attribute_count = sum(
        1 for n, _ in element_nodes
        if any(attr.startswith("aria-") for attr in n.attrs)
    )
    imgs           = soup.find_all("img")
    imgs_with_alt  = [i for i in imgs if (i.get("alt") or "").strip()]
    alt_text_coverage = (
        len(imgs_with_alt) / len(imgs) if imgs else None
    )
    role_attribute_count = sum(1 for n, _ in element_nodes if n.get("role"))
    tabindex_count       = sum(
        1 for n, _ in element_nodes if n.get("tabindex") is not None
    )

    # ── Navigation ────────────────────────────────────────────────────────────
    all_links       = soup.find_all("a")
    total_link_count = len(all_links)

    nav_elements  = soup.find_all("nav")
    nav_link_count = sum(len(nav.find_all("a")) for nav in nav_elements)

    footer_elements  = soup.find_all("footer")
    footer_link_count = sum(len(f.find_all("a")) for f in footer_elements)

    internal_link_count = 0
    for a in all_links:
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#"):
            internal_link_count += 1
        elif href.startswith("/"):
            internal_link_count += 1
        elif base_domain and base_domain in urlparse(href).netloc:
            internal_link_count += 1
    internal_link_ratio = (
        internal_link_count / total_link_count if total_link_count > 0 else None
    )

    nav_depth_levels = max(
        (max_nested_list_depth(nav) for nav in nav_elements), default=0
    )

    # ── Assemble output ───────────────────────────────────────────────────────
    def r(v, n=4):
        return round(v, n) if isinstance(v, float) else v

    return {
        # Document Scale
        "total_node_count":       total_node_count,
        "total_element_count":    total_element_count,
        "document_byte_size":     document_byte_size,
        "inner_text_length":      inner_text_length,
        "inner_html_length":      inner_html_length,
        "html_to_text_ratio":     r(html_to_text_ratio),
        # Tree Structure
        "max_dom_depth":          max_dom_depth,
        "mean_dom_depth":         r(mean_dom_depth, 3),
        "std_dom_depth":          r(std_dom_depth, 3),
        "max_children_count":     max_children_count,
        "mean_children_count":    r(mean_children_count, 3),
        # Tag Distribution
        "tag_entropy":            r(tag_entropy),
        "unique_tag_count":       unique_tag_count,
        "semantic_tag_ratio":     r(semantic_tag_ratio),
        "table_element_count":    table_element_count,
        "list_element_count":     list_element_count,
        "heading_count":          heading_count,
        "heading_distribution":   heading_distribution,
        # Interactive Complexity
        "form_count":             form_count,
        "input_count":            input_count,
        "button_count":           button_count,
        "iframe_count":           iframe_count,
        "interactive_density":    r(interactive_density),
        # Media & Resources
        "image_count":            image_count,
        "svg_count":              svg_count,
        "video_count":            video_count,
        "canvas_count":           canvas_count,
        "total_image_byte_size":  total_image_byte_size,
        "image_to_doc_size_ratio": image_to_doc_size_ratio,
        # External Dependencies
        "external_script_count":  external_script_count,
        "inline_script_count":    inline_script_count,
        "stylesheet_count":       stylesheet_count,
        "unique_external_domains": unique_external_domains,
        "total_script_byte_size": total_script_byte_size,
        # Styling Signals
        "inline_style_count":     inline_style_count,
        "total_css_byte_size":    total_css_byte_size,
        "media_query_count":      media_query_count,
        "css_rule_count":         css_rule_count,
        # Structured Data & SEO
        "has_json_ld":            has_json_ld,
        "json_ld_type_count":     json_ld_type_count,
        "meta_tag_count":         meta_tag_count,
        "open_graph_tag_count":   open_graph_tag_count,
        "canonical_tag_present":  canonical_tag_present,
        "robots_meta_content":    robots_meta_content,
        # Accessibility
        "aria_attribute_count":   aria_attribute_count,
        "alt_text_coverage":      r(alt_text_coverage) if alt_text_coverage is not None else None,
        "role_attribute_count":   role_attribute_count,
        "tabindex_count":         tabindex_count,
        # Navigation
        "nav_link_count":         nav_link_count,
        "total_link_count":       total_link_count,
        "internal_link_ratio":    r(internal_link_ratio) if internal_link_ratio is not None else None,
        "footer_link_count":      footer_link_count,
        "nav_depth_levels":       nav_depth_levels,
    }


# ── Multi-website merging ────────────────────────────────────────────────────

def combine_stats(stats_list: list[dict]) -> dict:
    """
    Merge stats from multiple HTML files for the same company.
    Counts are summed; ratios are re-derived from the merged counts;
    depths/entropy are averaged.
    """
    if len(stats_list) == 1:
        return stats_list[0]

    combined = {}
    sum_keys = {
        "total_node_count", "total_element_count", "document_byte_size",
        "inner_text_length", "inner_html_length", "table_element_count",
        "list_element_count", "heading_count", "form_count", "input_count",
        "button_count", "iframe_count", "image_count", "svg_count",
        "video_count", "canvas_count", "external_script_count",
        "inline_script_count", "stylesheet_count", "total_script_byte_size",
        "inline_style_count", "total_css_byte_size", "media_query_count",
        "css_rule_count", "json_ld_type_count", "meta_tag_count",
        "open_graph_tag_count", "aria_attribute_count", "role_attribute_count",
        "tabindex_count", "nav_link_count", "total_link_count", "footer_link_count",
    }
    max_keys  = {"max_dom_depth", "max_children_count", "nav_depth_levels"}
    avg_keys  = {
        "mean_dom_depth", "std_dom_depth", "mean_children_count",
        "tag_entropy", "html_to_text_ratio",
    }
    bool_or_keys = {"has_json_ld", "canonical_tag_present"}

    for k in sum_keys:
        combined[k] = sum(s.get(k) or 0 for s in stats_list)
    for k in max_keys:
        combined[k] = max((s.get(k) or 0) for s in stats_list)
    for k in avg_keys:
        vals = [s[k] for s in stats_list if s.get(k) is not None]
        combined[k] = sum(vals) / len(vals) if vals else None
    for k in bool_or_keys:
        combined[k] = any(s.get(k) for s in stats_list)

    # Heading distribution: element-wise sum
    combined["heading_distribution"] = {
        h: sum(s.get("heading_distribution", {}).get(h, 0) for s in stats_list)
        for h in ["h1","h2","h3","h4","h5","h6"]
    }

    # Re-derive ratios
    combined["html_to_text_ratio"] = (
        combined["inner_html_length"] / combined["inner_text_length"]
        if combined["inner_text_length"] else 0.0
    )
    combined["semantic_tag_ratio"] = (
        sum(s.get("semantic_tag_ratio", 0) * s.get("total_element_count", 0) for s in stats_list)
        / combined["total_element_count"]
        if combined["total_element_count"] else 0.0
    )
    combined["interactive_density"] = (
        (combined["form_count"] + combined["input_count"]
         + combined["button_count"] + combined["iframe_count"])
        / combined["total_element_count"]
        if combined["total_element_count"] else 0.0
    )
    combined["unique_external_domains"] = max(
        s.get("unique_external_domains", 0) for s in stats_list
    )

    # Alt text coverage: weighted average
    total_imgs = combined["image_count"]
    covered = sum(
        (s.get("alt_text_coverage") or 0) * (s.get("image_count") or 0)
        for s in stats_list
    )
    combined["alt_text_coverage"] = covered / total_imgs if total_imgs else None

    combined["internal_link_ratio"] = (
        sum(
            (s.get("internal_link_ratio") or 0) * s.get("total_link_count", 0)
            for s in stats_list
        ) / combined["total_link_count"]
        if combined["total_link_count"] else None
    )

    # Null-only fields
    combined["total_image_byte_size"]   = None
    combined["image_to_doc_size_ratio"] = None
    combined["unique_tag_count"]        = max(s.get("unique_tag_count", 0) for s in stats_list)
    combined["robots_meta_content"]     = next(
        (s["robots_meta_content"] for s in stats_list if s.get("robots_meta_content")), None
    )

    return combined


# ── HTML → Markdown ──────────────────────────────────────────────────────────

def html_to_markdown(html: str) -> str:
    """Convert raw HTML to markdown using html2text."""
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.ignore_emphasis = False
    h.body_width = 0  # Don't wrap lines
    h.unicode_snob = True
    return h.handle(html)


# ── Git-diff header/footer stripping ────────────────────────────────────────

def strip_header_footer(reference_text: str, target_text: str) -> str:
    """
    Use git diff --no-index to find content unique to target_text,
    effectively stripping repeated headers/footers.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_file = os.path.join(tmpdir, "ref.txt")
        target_file = os.path.join(tmpdir, "target.txt")

        with open(ref_file, "w", encoding="utf-8") as f:
            f.write(reference_text)
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(target_text)

        result = subprocess.run(
            ["git", "diff", "--no-index", "--no-prefix", ref_file, target_file],
            capture_output=True,
            text=True,
        )

        unique_lines = []
        for line in result.stdout.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                unique_lines.append(line[1:])  # Remove the '+' prefix

        return "\n".join(unique_lines)


# ── Per-domain processing ────────────────────────────────────────────────────

def process_domain_dir(
    domain_dir: Path,
    markdown_dir: Path,
    stats_dir: Path,
) -> tuple[str, str]:
    """Process one domain's HTML files into markdown + combined stats.

    Returns (domain, status).
    """
    domain   = domain_dir.name
    md_out   = markdown_dir / f"{domain}.md"
    stats_out = stats_dir / f"{domain}.json"

    # Skip if already done
    if md_out.exists() and stats_out.exists():
        return domain, "skipped"

    # Load per-domain manifest
    manifest_path = domain_dir / "manifest.json"
    if not manifest_path.exists():
        return domain, "no_manifest"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ok_pages = [p for p in manifest if p.get("status") in ("ok", "skipped")]

    if not ok_pages:
        return domain, "no_pages"

    # Convert each page to markdown and compute stats
    pages_data = []
    for page in ok_pages:
        html_path = domain_dir / f"{page['slug']}.html"
        if not html_path.exists():
            continue

        html = html_path.read_text(encoding="utf-8", errors="replace")
        if len(html) < 50:
            continue

        md    = html_to_markdown(html)
        stats = compute_stats(html)

        parsed_url = urlparse(page["url"])
        path = (parsed_url.netloc or "").lower().removeprefix("www.") + (parsed_url.path.rstrip("/") or "/")

        pages_data.append({
            "url":      page["url"],
            "path":     path,
            "slug":     page["slug"],
            "markdown": md,
            "stats":    stats,
            "depth":    url_depth(page["url"]),
        })

    if not pages_data:
        return domain, "no_valid_pages"

    # Sort by depth, then alphabetically by path
    pages_data.sort(key=lambda p: (p["depth"], p["path"]))

    # Build path→markdown dict for git-diff parent lookups
    path_to_md = {p["path"]: p["markdown"] for p in pages_data}

    # Git-diff deduplication + concatenation
    output_lines = []
    for idx, page in enumerate(pages_data):
        output_lines.append(f"########## {page['url']}")

        content = page["markdown"]

        # Find reference: walk up path hierarchy
        reference = None
        path = page["path"]
        for _ in range(5):
            if "/" not in path:
                break
            path = path[:path.rfind("/")]
            reference = path_to_md.get(path)
            if reference is not None:
                break

        # Fallback: use previous page if no parent found (skip root page)
        if reference is None and idx > 0:
            reference = pages_data[idx - 1]["markdown"]

        if reference is not None:
            content = strip_header_footer(reference, content)

        output_lines.append(content)

    # Write concatenated markdown
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(output_lines), encoding="utf-8")

    # Combine stats and write
    all_stats = [p["stats"] for p in pages_data]
    combined  = combine_stats(all_stats)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    return domain, f"ok:{len(pages_data)}_pages"


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default=str(Path(__file__).resolve().parents[1] / "cc_data" / "raw_html"),
        help="Directory containing <domain>/ subdirectories of .html files (default: %(default)s)",
    )
    parser.add_argument(
        "--markdown-dir",
        default=str(Path(__file__).resolve().parents[1] / "cc_data" / "markdown"),
        help="Output directory for concatenated .md files (default: %(default)s)",
    )
    parser.add_argument(
        "--stats-dir",
        default=str(Path(__file__).resolve().parents[1] / "cc_data" / "stats"),
        help="Output directory for combined stats JSON files (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Parallel worker processes (default: %(default)s)",
    )
    args = parser.parse_args()

    input_dir    = Path(args.input_dir)
    markdown_dir = Path(args.markdown_dir)
    stats_dir    = Path(args.stats_dir)
    markdown_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Find all domain directories that have a manifest
    domain_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and (d / "manifest.json").exists()
    ])
    logger.info("Found %d domain directories in %s", len(domain_dirs), input_dir)

    counts: dict[str, int] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_domain_dir, d, markdown_dir, stats_dir): d.name
            for d in domain_dirs
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing domains"):
            _, status = future.result()
            key = status.split(":")[0] if ":" in status else status
            counts[key] = counts.get(key, 0) + 1

    logger.info("Summary: %s", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
