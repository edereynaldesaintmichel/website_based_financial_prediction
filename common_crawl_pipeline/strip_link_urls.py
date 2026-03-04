"""
Replace URLs inside markdown link syntax with "link" placeholder.
[text](url) or [text](url "title") => [text](link)
Standalone URLs (e.g. in headings) are left unchanged.
"""
import re
import sys
from pathlib import Path

# Matches [any text](anything inside parens)
LINK_RE = re.compile(r'(\[[^\]]*\])\([^)]+\)')

def strip_link_urls(text: str) -> str:
    return LINK_RE.sub(r'\1(link)', text)

def process_dir(md_dir: Path) -> None:
    files = list(md_dir.glob('*.md'))
    print(f"Processing {len(files)} files...")
    for path in files:
        original = path.read_text(encoding='utf-8', errors='replace')
        modified = strip_link_urls(original)
        if modified != original:
            path.write_text(modified, encoding='utf-8')
    print("Done.")

if __name__ == '__main__':
    md_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('cc_data/markdown')
    process_dir(md_dir)
