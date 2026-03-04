import json
from pathlib import Path

DIRECTORY = Path("/Users/eloireynal/Downloads/json_50plus_employees")

website_counts = []

for json_file in DIRECTORY.glob("*.json"):
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    websites = data.get("identifier", {}).get("websites", [])
    website_counts.append(len(websites))

total = len(website_counts)
if total == 0:
    print("No JSON files found.")
else:
    avg = sum(website_counts) / total
    two_plus = sum(1 for c in website_counts if c >= 2)
    none = sum(1 for c in website_counts if c == 0)

    print(f"Total files:            {total}")
    print(f"Average websites:       {avg:.2f}")
    print(f"Files with 2+ websites: {two_plus}")
    print(f"Files with no website:  {none}")
