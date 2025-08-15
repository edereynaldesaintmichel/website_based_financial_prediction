import os
import re
from pathlib import Path

def count_slashes(url):
    """Count slashes in URL, excluding trailing slash"""
    url = url.rstrip('/')
    # Remove protocol if present
    url = re.sub(r'^https?://', '', url)
    return url.count('/')

def is_bullshit_page(url):
    """Check if URL contains patterns indicating non-content pages"""
    bullshit_patterns = [
        'privacy', 'terms', 'conditions', 'certificate', 'bylaws', 'legal', 'cookie',
        'disclaimer', 'copyright',
        'sitemap', 'accessibility', 'gdpr', 'compliance',
        'unsubscribe', 'preferences', 'settings', 'login', 'signup',
        'register', 'forgot-password', 'reset-password', '404',
        'error', 'search', 'tag/', 'category/', 'page/', 'author/',
        'wp-admin', 'wp-content'
    ]
    
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in bullshit_patterns)

def process_document(input_path, output_path):
    """Process a single document: filter and reorder pages"""
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by page separator
    pages = content.split('##########')
    
    # Parse pages into structured format
    parsed_pages = []
    for page in pages:
        if not page.strip():
            continue
        
        lines = page.strip().split('\n')
        if not lines:
            continue
            
        url = lines[0].strip()
        
        # Skip bullshit pages
        if is_bullshit_page(url):
            continue
            
        page_content = '\n'.join(lines[1:]) if len(lines) > 1 else ''
        
        parsed_pages.append({
            'url': url,
            'content': page_content,
            'slash_count': count_slashes(url),
            'url_length': len(url.rstrip('/'))
        })
    
    # Sort pages by slash count, then by URL length
    sorted_pages = sorted(parsed_pages, key=lambda x: (x['slash_count'], x['url_length']))
    
    # Reconstruct document
    output_lines = []
    for page in sorted_pages:
        output_lines.append(f"########## {page['url']}")
        if page['content']:
            output_lines.append(page['content'])
    
    # Save to output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    return len(sorted_pages)

def process_all_documents(input_dir, output_dir):
    """Process all .txt files in the input directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    for txt_file in input_path.glob('*.txt'):
        input_file = str(txt_file)
        output_file = str(output_path / txt_file.name)
        
        try:
            num_pages = process_document(input_file, output_file)
            print(f"Processed {txt_file.name}: {num_pages} pages kept")
        except Exception as e:
            print(f"Error processing {txt_file.name}: {e}")

# Run the script
if __name__ == "__main__":
    input_directory = "/Users/eloireynal/Documents/My projects/crawl_data/txt/"
    output_directory = "/Users/eloireynal/Documents/My projects/crawl_data/sanitized_txt/"
    
    process_all_documents(input_directory, output_directory)