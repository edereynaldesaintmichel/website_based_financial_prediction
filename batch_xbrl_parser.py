#!/usr/bin/env python3
"""
Batch script to parse all iXBRL files from a directory.
Processes files one at a time to avoid memory issues.
Outputs parsed JSON files and a sorted revenue ranking.
"""

import json
import os
from pathlib import Path
from lxml import etree
from ixbrl_parse.ixbrl import parse


# Configuration
INPUT_DIR = '/Users/eloireynal/Downloads/Accounts_Monthly_Data-August2024'
OUTPUT_DIR = '/Users/eloireynal/Downloads/Accounts_Monthly_Data-August2024_parsed'
REVENUE_OUTPUT_FILE = '/Users/eloireynal/Downloads/sorted_by_revenue.json'


def parse_single_file(file_path: str) -> tuple[dict | None, float | None]:
    """
    Parse a single iXBRL file and return the parsed data and revenue.
    
    Returns:
        tuple: (parsed_data dict or None on error, revenue float or None if not found)
    """
    try:
        if file_path == '/Users/eloireynal/Downloads/Accounts_Monthly_Data-August2024/Prod224_2466_13723269_20231130.html':
            yes = "yes"
        # Parse the HTML file with lxml
        tree = etree.parse(file_path)
        
        # Parse the iXBRL data
        xbrl = parse(tree)
        
        # Build a dictionary with all the parsed data
        parsed_data = {}
        revenue = None
        
        # Extract values (facts)
        for context_key, value in xbrl.values.items():
            # Filter to numeric values only
            unit_name = str(value.unit)
            is_number = unit_name in ['pure', 'GBP']
            if not is_number:
                continue
            
            try:
                value_string = float(str(value).replace(f'({str(value.unit)})', ''))
            except (ValueError, TypeError):
                continue
                
            unit_to_be_displayed = unit_name in ['GBP']
            value_name = ''.join([' ' + l if l.isupper() else l for l in value.name.localname]).strip()
            
            # Extract date
            date = None
            if value.context.period:
                date = str(value.context.period.end)
            elif value.context.instant:
                date = str(value.context.instant.instant)
            
            if not date:
                continue
            
            if date not in parsed_data:
                parsed_data[date] = {}
            
            complete_value_string = f"<number>{value_string}</number>"
            value_data = {
                "value": complete_value_string.strip(),
                "unit": str(value.unit) if unit_to_be_displayed else '',
            }
            
            parsed_data[date][value_name] = value_data
            
            # Check for revenue (common iXBRL names for revenue/turnover)
            revenue_keywords = ['turnover', 'revenue', 'sales']
            value_name_lower = value_name.lower()
            if any(kw in value_name_lower for kw in revenue_keywords):
                # Keep the highest revenue value found
                if revenue is None or value_string > revenue:
                    revenue = value_string
        
        return parsed_data, revenue
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None, None


def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of HTML files
    input_path = Path(INPUT_DIR)
    html_files = list(input_path.glob('*.html'))
    total_files = len(html_files)
    
    print(f"Found {total_files} HTML files to process")
    
    # Track revenue for each file
    revenue_data = {}
    
    # Process files one by one
    processed = 0
    errors = 0
    
    for i, html_file in enumerate(html_files):
        filename = html_file.name
        
        # Progress update every 1000 files
        if (i + 1) % 1000 == 0:
            print(f"Progress: {i + 1}/{total_files} ({(i + 1) / total_files * 100:.1f}%)")
        
        # Parse the file
        parsed_data, revenue = parse_single_file(str(html_file))
        
        if parsed_data is None:
            errors += 1
            continue
        
        # Save parsed JSON
        output_filename = filename.replace('.html', '_parsed.json')
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving {output_filename}: {e}")
            errors += 1
            continue
        
        # Track revenue if found
        if revenue is not None:
            revenue_data[filename] = revenue
        
        processed += 1
    
    print(f"\nProcessing complete!")
    print(f"  Successfully processed: {processed}")
    print(f"  Errors: {errors}")
    print(f"  Files with revenue data: {len(revenue_data)}")
    
    # Sort by revenue descending and save
    sorted_revenue = dict(sorted(revenue_data.items(), key=lambda x: x[1], reverse=True))
    
    with open(REVENUE_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sorted_revenue, f, indent=2, ensure_ascii=False)
    
    print(f"\nSorted revenue data saved to: {REVENUE_OUTPUT_FILE}")
    
    # Show top 10 by revenue
    print("\nTop 10 by revenue:")
    for i, (filename, rev) in enumerate(list(sorted_revenue.items())[:10]):
        print(f"  {i + 1}. {filename}: £{rev:,.2f}")


if __name__ == "__main__":
    main()
