import boto3
import gzip
from io import BytesIO
import warcio
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import pdfplumber
import os
from collections import defaultdict
from readability import Document
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from functools import partial
import logging
from datetime import datetime

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(message)s'
)

def create_s3_client():
    """Create S3 client for each process"""
    return boto3.client('s3')

def fetch_page_content(row, s3_client):
    """
    Fetch the actual page content from Common Crawl S3 bucket
    """
    bucket = 'commoncrawl'
    key = row['warc_filename']
    
    record_length = row['warc_record_length']
    buffer_size = 4096 
    
    # Calculate the byte range to fetch
    offset = row['warc_record_offset']
    end_offset = offset + record_length + buffer_size - 1
    byte_range = f"bytes={offset}-{end_offset}"
    
    try:
        # Fetch only the specific record using byte range
        response = s3_client.get_object(
            Bucket=bucket, 
            Key=key, 
            Range=byte_range,
            RequestPayer='requester'
        )
        
        # The data is gzipped
        raw_data = response['Body'].read()
        
        # Parse the WARC record
        stream = BytesIO(raw_data)
        for record in warcio.ArchiveIterator(stream):
            if record.rec_type == 'response':
                # Get the HTTP response
                content = record.content_stream().read()
                
                # Extract content type from headers
                headers = record.http_headers
                content_type = headers.get_header('Content-Type', '')
                
                return {
                    'content': content,
                    'content_type': content_type,
                    'url': row['url']
                }
    
    except Exception as e:
        logging.error(f"Error fetching {row['url']}: {str(e)}")
        return None

def extract_text_from_content(content_data, process_text=True):
    """Extract text from content data"""
    if not content_data or not content_data.get('content'):
        return None, None

    content = content_data['content']
    content_type = content_data.get('content_type', '').lower()
    url = content_data.get('url', '')

    if 'application/pdf' in content_type or url.endswith('.pdf'):
        try:
            with pdfplumber.open(BytesIO(content)) as pdf:
                full_text = "".join(page.extract_text() or "" for page in pdf.pages)
            return (full_text.strip(), full_text.strip()) if full_text.strip() else (None, None)
        except Exception:
            return (None, None)

    is_html_like = ('text/html' in content_type or
                    url.endswith(('.html', '.htm', '.asp', '.aspx')))

    if is_html_like:
        try:
            try:
                html_text = content.decode('utf-8')
            except UnicodeDecodeError:
                html_text = content.decode('latin-1', errors='ignore')
            
            if not process_text:
                return (None, html_text)
                
            doc = Document(html_text)
            
            cleaned_html = doc.summary()
            soup = BeautifulSoup(cleaned_html, 'html.parser')
            
            text = soup.get_text(separator='\n', strip=True)
            
            if not text.strip():
                soup = BeautifulSoup(html_text, 'html.parser')
                for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    script.decompose()
                body = soup.find('body')
                if body:
                    text = body.get_text(separator='\n', strip=True)

            return (text.strip(), html_text) if text.strip() else (None, html_text)

        except Exception:
            return None, None

    return None, None

def save_host_content(host_content, host_name, base_path='/Users/eloireynal/Documents/My projects/crawl_data', extension="txt"):
    """Save concatenated content for each host"""
    os.makedirs(base_path, exist_ok=True)
    safe_host = host_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    filepath = os.path.join(base_path, extension, f"{safe_host}.{extension}")
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for url, text in host_content.items():
                f.write(f"########## {url}\n")
                f.write(text)
                f.write("\n\n")
        
        logging.info(f"Saved content for {host_name} to {filepath}")
        return True
    except Exception as e:
        logging.error(f"Error saving content for {host_name}: {str(e)}")
        return False

def process_single_url(row, s3_client, process_text=True):
    """Process a single URL and return the result"""
    try:
        content_data = fetch_page_content(row, s3_client)
        text, html = extract_text_from_content(content_data, process_text)
        
        return row['url'], text, html, len(text) if text else 0
    except Exception as e:
        logging.error(f"Error processing {row['url']}: {str(e)}")
        return row['url'], None, 0

def process_host_group(host_data, save_text=False):
    """Process all URLs for a single host"""
    host_name, group_df = host_data
    
    logging.info(f"Starting processing for host: {host_name} ({len(group_df)} URLs)")
    
    # Create S3 client for this process
    s3_client = create_s3_client()
    
    host_content = {}
    host_content_html = {}
    successful_extractions = 0
    
    # Process URLs for this host using thread pool for I/O operations
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_row = {
            executor.submit(process_single_url, row, s3_client, save_text): row 
            for _, row in group_df.iterrows()
        }
        
        # Process completed tasks
        for future in as_completed(future_to_row):
            url, text, html, text_length = future.result()
            
            if html:
                host_content_html[url] = html
            if text:
                host_content[url] = text
                successful_extractions += 1
                logging.debug(f"Successfully extracted {text_length} characters from {url}")
    
    # Save the content for this host
    if host_content and save_text:
        save_host_content(host_content, host_name)
    if host_content_html:
        save_host_content(host_content_html, host_name, extension='.html')
    
    logging.info(f"Completed host: {host_name} - {successful_extractions}/{len(group_df)} URLs extracted")
    
    return host_name, successful_extractions, len(group_df)

def process_all_records_parallel(start_index=0, max_workers=None, save_text=True):
    """Process all records in parallel, grouped by host"""
    
    # Load the dataframe
    logging.info("Loading dataframe...")
    df_filtered = pd.read_parquet("commoncrawl_sanitized.parquet")
    
    if start_index > 0:
        df_filtered = df_filtered.iloc[start_index:]
    
    # Group by host
    logging.info("Grouping by host...")
    grouped = df_filtered.groupby('url_host_name')
    host_groups = list(grouped)
    
    logging.info(f"Found {len(host_groups)} unique hosts to process")
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 16)
    
    logging.info(f"Using {max_workers} worker processes")
    
    # Process hosts in parallel
    total_hosts = len(host_groups)
    completed_hosts = 0
    total_urls_processed = 0
    total_urls_extracted = 0
    
    start_time = datetime.now()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_host = {
            executor.submit(process_host_group, host_data, save_text): host_data[0] 
            for host_data in host_groups
        }
        
        # Process completed tasks
        for future in as_completed(future_to_host):
            try:
                host_name, extracted, total = future.result()
                completed_hosts += 1
                total_urls_processed += total
                total_urls_extracted += extracted
                
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = completed_hosts / elapsed if elapsed > 0 else 0
                eta = (total_hosts - completed_hosts) / rate if rate > 0 else 0
                
                logging.info(
                    f"Progress: {completed_hosts}/{total_hosts} hosts "
                    f"({total_urls_extracted}/{total_urls_processed} URLs extracted) "
                    f"- Rate: {rate:.2f} hosts/sec - ETA: {eta/60:.1f} minutes"
                )
                
            except Exception as e:
                host_name = future_to_host[future]
                logging.error(f"Failed to process host {host_name}: {str(e)}")
    
    elapsed_total = (datetime.now() - start_time).total_seconds()
    logging.info(
        f"\nProcessing complete!\n"
        f"Total time: {elapsed_total/60:.1f} minutes\n"
        f"Hosts processed: {completed_hosts}/{total_hosts}\n"
        f"URLs extracted: {total_urls_extracted}/{total_urls_processed}\n"
        f"Average rate: {completed_hosts/elapsed_total:.2f} hosts/sec"
    )

if __name__ == "__main__":
    # You can adjust max_workers based on your system capabilities
    # More workers = faster processing but more memory/CPU usage
    process_all_records_parallel(start_index=10019, max_workers=8, save_text=False)