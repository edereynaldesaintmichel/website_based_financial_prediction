#!/usr/bin/env python3
"""
Simple script to test the ixbrl-parse library.
Parses an iXBRL file and dumps the data as JSON.
"""

import json
from lxml import etree
from ixbrl_parse.ixbrl import parse


def main():
    # Hardcoded path to the iXBRL file
    file_path = '/Users/eloireynal/Documents/My projects/website_based_financial_prediction/financial_statements_100_biggest/Prod224_2466_00211587_20231231.html'
    
    print(f"Parsing iXBRL file: {file_path}")
    
    # Parse the HTML file with lxml
    tree = etree.parse(file_path)
    
    # Parse the iXBRL data
    xbrl = parse(tree)
    
    # Build a dictionary with all the parsed data
    parsed_data = {}
    
    # Extract values (facts)
    for context_key, value in xbrl.values.items():
        # context_key is a tuple: (context_dict, qname)
        unit_name = str(value.unit)
        is_number = unit_name in ['pure', 'GBP']
        if not is_number:
            continue
        # value_string = value.elements[0].text
        value_string = float(str(value).replace(f'({str(value.unit)})', ''))
        unit_to_be_displayed = unit_name in ['GBP']
        value_name = ''.join([' ' + l if l.isupper() else l for l in value.name.localname]).strip()
        if value.name.localname == 'EntityCurrentLegalOrRegisteredName':
            bof = "bof"
        date = str(value.context.period.end) if value.context.period else str(value.context.instant.instant) if value.context.instant else None
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
    
    
    json_output = json.dumps(parsed_data, indent=2, default=str, ensure_ascii=False)
    
    # Save to file
    output_path = file_path.replace('.html', '_parsed.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json_output)
    print(f"JSON output saved to: {output_path}")
    

if __name__ == "__main__":
    main()
