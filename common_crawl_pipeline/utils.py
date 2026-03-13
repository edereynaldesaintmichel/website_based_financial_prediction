"""Shared utilities for the Common Crawl pipeline."""

import hashlib
import re
from urllib.parse import urlparse

# Crawl priority order — first match wins during dedup
CC_CRAWL_PRIORITY = [
    "CC-MAIN-2018-05",  # Jan 2018 ← original target
    "CC-MAIN-2017-51",  # Dec 2017
    "CC-MAIN-2018-09",  # Feb 2018
    "CC-MAIN-2018-13",  # Mar 2018
]

PAGE_CAP = 20

BULLSHIT_PATTERNS = [
    "privacy", "terms", "conditions", "certificate", "bylaws", "legal", "cookie",
    "disclaimer", "copyright",
    "sitemap", "accessibility", "gdpr", "compliance",
    "unsubscribe", "preferences", "settings", "login", "signup",
    "register", "forgot-password", "reset-password", "404",
    "error", "search", "tag/", "wp-admin", "wp-content",
]


def is_bullshit_page(url: str) -> bool:
    """Return True if URL path contains patterns indicating non-content pages."""
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in BULLSHIT_PATTERNS)


def url_depth(url: str) -> int:
    """Count path segments in a URL, excluding trailing slash."""
    path = urlparse(url).path.rstrip("/")
    if not path or path == "/":
        return 0
    return path.count("/")


def url_to_slug(url: str) -> str:
    """Convert a URL into a safe filename slug.

    Examples:
        'http://example.com/'             -> 'index'
        'http://example.com/about'        -> 'about'
        'http://example.com/about/team'   -> 'about__team'
        'http://example.com/page.html'    -> 'page'
    """
    path = urlparse(url).path.strip("/")
    if not path:
        return "index"
    # Remove common extensions
    path = re.sub(r"\.(html?|php|aspx?|jsp|cfm|do)$", "", path, flags=re.IGNORECASE)
    # Replace / with __
    slug = path.replace("/", "__")
    # Replace non-alphanumeric chars (except _ and -) with _
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", slug)
    # Collapse multiple underscores
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        return "index"
    # macOS HFS+/APFS limit is 255 bytes; cap slug and append a short hash
    # to avoid collisions when truncating
    max_len = 200
    if len(slug) > max_len:
        h = hashlib.md5(slug.encode()).hexdigest()[:12]
        slug = f"{slug[:max_len]}_{h}"
    return slug
