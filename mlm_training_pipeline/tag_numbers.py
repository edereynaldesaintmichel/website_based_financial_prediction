import re

NUMBER_RE = re.compile(
    r"""(?<![.\d"'])"""            # not preceded by digit, dot, or quote
    r'(?P<num>'
        r'\d{1,3}(?:,\d{3})+'   # comma-formatted (1,234 or 12,345,678)
        r'(?:\.\d+)?'            # optional decimal
    r'|'
        r'\d+'                   # plain integer
        r'(?:\.\d+)?'            # optional decimal
    r')'
    r'(?!\d|\.\d)'               # not followed by digit or decimal point+digit
    r"""(?!["'])"""               # not followed by quote
)


def normalize_number(raw: str) -> str:
    clean = raw.replace(",", "").strip()
    try:
        return str(int(clean))
    except ValueError:
        pass
    try:
        return f"{float(clean):.10f}".rstrip("0").rstrip(".")
    except ValueError:
        return raw


def strip_number_tags(text: str) -> str:
    return text.replace("<number>", "").replace("</number>", "")


def tag_numbers_in_text(text: str) -> str:
    text = strip_number_tags(text)
    def replace(m):
        return f"<number>{normalize_number(m.group('num'))}</number>"
    return NUMBER_RE.sub(replace, text)


if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Tag numbers in text.")
    parser.add_argument("input", nargs="?",
                        help="Input file or directory (default: stdin)")
    parser.add_argument("-o", "--output",
                        help="Output file or directory (default: stdout). "
                             "If input is a directory, output must also be a directory.")
    parser.add_argument("--glob", default="*.md",
                        help="Glob applied when input is a directory (default: *.md)")
    args = parser.parse_args()

    in_path = Path(args.input) if args.input else None

    if in_path and in_path.is_dir():
        if not args.output:
            parser.error("--output is required when input is a directory")
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(in_path.glob(args.glob))
        try:
            from tqdm import tqdm
            iterator = tqdm(files, desc="Tag", unit="file")
        except ImportError:
            iterator = files
        total_tags = 0
        n_written = 0
        for f in iterator:
            out_path = out_dir / f.name
            if out_path.exists():
                continue
            tagged = tag_numbers_in_text(f.read_text(encoding="utf-8"))
            out_path.write_text(tagged, encoding="utf-8")
            total_tags += tagged.count("<number>")
            n_written += 1
        print(f"--- {n_written}/{len(files)} files written, {total_tags} numbers tagged "
              f"-> {out_dir} ---", file=sys.stderr)
    else:
        text = open(args.input).read() if args.input else sys.stdin.read()
        tagged = tag_numbers_in_text(text)

        if args.output:
            with open(args.output, "w") as f:
                f.write(tagged)
        else:
            print(tagged)

        print(f"--- {tagged.count('<number>')} numbers tagged ---", file=sys.stderr)
