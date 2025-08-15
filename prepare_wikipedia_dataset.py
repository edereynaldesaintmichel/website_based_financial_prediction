import os
import re
import random
import pickle
import logging
import torch
from tqdm import tqdm
import argparse
from typing import List, Tuple, Dict, Any
import gzip
import json
import requests
from datasets import load_dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    """Configuration class for processing parameters"""
    CACHE_DIR = "./cache"
    MAX_CONTEXT_LEN = 8192
    MIN_CONTEXT_LEN = 512
    MIN_SUB_SEQUENCES = 2
    MAX_SUB_SEQUENCES = 64
    BATCH_SIZE = 8
    
    # Wikipedia specific settings
    MIN_ARTICLE_LENGTH = 500  # Minimum characters for an article to be considered
    MAX_ARTICLES_TO_LOAD = 100000  # Limit for memory management

def ensure_cache_dir():
    """Ensure cache directory exists"""
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

def download_wikipedia_alternative():
    """
    Download Wikipedia-like dataset from alternative sources.
    Using wikimedia/wikipedia dataset or other alternatives.
    """
    logging.info("Downloading Wikipedia dataset from alternative source...")
    
    try:
        # Option 1: Try the new wikimedia/wikipedia dataset (most up-to-date)
        try:
            logging.info("Attempting to load wikimedia/wikipedia dataset...")
            wiki_dataset = load_dataset(
                "wikimedia/wikipedia", 
                "20231101.en",  # English Wikipedia snapshot
                split="train",
                streaming=True  # Use streaming to avoid loading entire dataset into memory
            )
            logging.info("Successfully loaded wikimedia/wikipedia dataset")
            return wiki_dataset, "wikimedia"
        except Exception as e:
            logging.warning(f"Could not load wikimedia/wikipedia: {e}")
        
        # Option 2: Try Wikipedia sample datasets
        try:
            logging.info("Attempting to load wikipedia sample dataset...")
            wiki_dataset = load_dataset(
                "Cohere/wikipedia-22-12-en-embeddings",
                split="train",
                streaming=True
            )
            logging.info("Successfully loaded Cohere Wikipedia dataset")
            return wiki_dataset, "cohere"
        except Exception as e:
            logging.warning(f"Could not load Cohere dataset: {e}")
        
        # Option 3: Use a general text dataset as fallback
        try:
            logging.info("Falling back to wikitext dataset...")
            wiki_dataset = load_dataset(
                "wikitext", 
                "wikitext-103-v1",
                split="train",
                streaming=False  # Wikitext is smaller, can load fully
            )
            logging.info("Successfully loaded wikitext dataset")
            return wiki_dataset, "wikitext"
        except Exception as e:
            logging.warning(f"Could not load wikitext: {e}")
            
        # Option 4: Use OpenWebText as final fallback
        logging.info("Final fallback to OpenWebText dataset...")
        wiki_dataset = load_dataset(
            "Skylion007/openwebtext",
            split="train",
            streaming=True
        )
        logging.info("Successfully loaded OpenWebText dataset")
        return wiki_dataset, "openwebtext"
        
    except Exception as e:
        logging.error(f"Error downloading any dataset: {e}")
        raise

def sensible_split(text: str, num_chunks: int) -> List[str]:
    """Splits text into a number of chunks at sensible points (punctuation/newlines)."""
    split_points = [m.start() for m in re.finditer(r'[.?!]\s|\n', text)]
    
    if len(split_points) < num_chunks - 1:
        chunk_size = len(text) // num_chunks
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)][:num_chunks]
    
    chosen_indices = sorted(random.sample(split_points, num_chunks - 1))
    chunks = []
    start_idx = 0
    
    for split_idx in chosen_indices:
        chunks.append(text[start_idx:split_idx+1].strip())
        start_idx = split_idx + 1
    
    chunks.append(text[start_idx:].strip())
    return [chunk for chunk in chunks if chunk]

def extract_text_chunks_from_article(article_text: str, num_chunks: int = 5) -> List[str]:
    """
    Extract multiple random chunks from a single article.
    Each chunk will be of varying length between MIN_CONTEXT_LEN and MAX_CONTEXT_LEN tokens.
    """
    chunks = []
    article_length = len(article_text)
    
    if article_length < Config.MIN_ARTICLE_LENGTH:
        return chunks
    
    for _ in range(num_chunks):
        # Randomly select chunk size in characters (approximate)
        # Assuming ~4 characters per token on average
        min_chars = Config.MIN_CONTEXT_LEN * 4
        max_chars = min(Config.MAX_CONTEXT_LEN * 4, article_length)
        
        if min_chars >= article_length:
            continue
            
        chunk_size = random.randint(min_chars, min(max_chars, article_length))
        
        # Random starting position
        max_start = article_length - chunk_size
        if max_start <= 0:
            start_pos = 0
        else:
            start_pos = random.randint(0, max_start)
        
        chunk = article_text[start_pos:start_pos + chunk_size]
        
        # Try to find sentence boundaries for cleaner chunks
        # Look for the first sentence start
        first_period = chunk.find('. ')
        if first_period > 0 and first_period < 100:
            chunk = chunk[first_period + 2:]
        
        # Look for the last complete sentence
        last_period = chunk.rfind('. ')
        if last_period > len(chunk) - 100 and last_period > 0:
            chunk = chunk[:last_period + 1]
        
        if len(chunk.strip()) >= Config.MIN_ARTICLE_LENGTH:
            chunks.append(chunk.strip())
    
    return chunks

def extract_text_from_dataset_item(item: Dict, dataset_type: str) -> Tuple[str, str]:
    """
    Extract text and title from different dataset formats.
    
    Returns:
        Tuple of (text, title)
    """
    if dataset_type == "wikimedia":
        # wikimedia/wikipedia format
        text = item.get('text', '')
        title = item.get('title', 'Unknown')
    elif dataset_type == "cohere":
        # Cohere Wikipedia format
        text = item.get('text', item.get('passage', ''))
        title = item.get('title', item.get('id', 'Unknown'))
    elif dataset_type == "wikitext":
        # Wikitext format - it's just raw text
        text = item.get('text', '')
        title = "WikiText Article"
    elif dataset_type == "openwebtext":
        # OpenWebText format
        text = item.get('text', '')
        title = "OpenWebText Document"
    else:
        # Generic fallback
        text = str(item.get('text', item))
        title = "Unknown Source"
    
    return text, title

def process_wikipedia_to_training_examples(
    num_examples: int = 5000,
    examples_per_article: int = 3,
    cache_file_prefix: str = "wikipedia"
) -> Tuple[List[Tuple[str, List[str]]], List[Dict[str, Any]]]:
    """
    Process Wikipedia articles into training examples with the same structure
    as the original function.
    
    Args:
        num_examples: Total number of training examples to create
        examples_per_article: Number of examples to extract from each article
        cache_file_prefix: Prefix for cache files
        
    Returns:
        Tuple of (texts_to_process, metadata) with same structure as original
    """
    ensure_cache_dir()
    
    # Check if processed data already exists
    texts_cache_file = os.path.join(Config.CACHE_DIR, f"{cache_file_prefix}_texts_to_process.pkl")
    metadata_cache_file = os.path.join(Config.CACHE_DIR, f"{cache_file_prefix}_metadata.pkl")
    
    if os.path.exists(texts_cache_file) and os.path.exists(metadata_cache_file):
        logging.info("Loading cached processed data...")
        with open(texts_cache_file, 'rb') as f:
            texts_to_process = pickle.load(f)
        with open(metadata_cache_file, 'rb') as f:
            metadata = pickle.load(f)
        logging.info(f"Loaded {len(texts_to_process)} cached examples")
        return texts_to_process, metadata
    
    # Download Wikipedia or alternative dataset
    dataset, dataset_type = download_wikipedia_alternative()
    
    texts_to_process = []
    metadata = []
    articles_processed = 0
    
    # Create progress bar
    pbar = tqdm(total=num_examples, desc="Creating training examples")
    
    # Handle both streaming and non-streaming datasets
    if hasattr(dataset, '__iter__'):
        dataset_iter = iter(dataset)
    else:
        dataset_iter = dataset
    
    # Process articles from the dataset
    for article in dataset_iter:
        if len(texts_to_process) >= num_examples:
            break
            
        if articles_processed >= Config.MAX_ARTICLES_TO_LOAD:
            break
            
        try:
            # Extract text and title based on dataset type
            article_text, article_title = extract_text_from_dataset_item(article, dataset_type)
            
            # Skip short articles
            if len(article_text.strip()) < Config.MIN_ARTICLE_LENGTH:
                continue
            
            # Extract chunks from this article
            chunks = extract_text_chunks_from_article(article_text, examples_per_article)
            
            for chunk_text in chunks:
                if len(texts_to_process) >= num_examples:
                    break
                
                # Create sub-sequences from the chunk
                num_sub_sequences = random.randint(
                    Config.MIN_SUB_SEQUENCES, 
                    min(Config.MAX_SUB_SEQUENCES, max(2, len(chunk_text) // 100))
                )
                
                sub_sequence_texts = sensible_split(chunk_text, num_sub_sequences)
                
                if len(sub_sequence_texts) < 2:
                    continue
                
                texts_to_process.append((chunk_text, sub_sequence_texts))
                metadata.append({
                    'source_file': f"{dataset_type}:{article_title}",
                    'num_sequences': len(sub_sequence_texts),
                    'article_id': article.get('id', 'unknown') if isinstance(article, dict) else 'unknown',
                    'chunk_length': len(chunk_text),
                    'dataset_type': dataset_type
                })
                
                pbar.update(1)
            
            articles_processed += 1
            
            # Log progress every 100 articles
            if articles_processed % 100 == 0:
                logging.info(f"Processed {articles_processed} articles, created {len(texts_to_process)} examples")
                
        except Exception as e:
            logging.warning(f"Error processing article: {e}")
            continue
    
    pbar.close()
    
    logging.info(f"Created {len(texts_to_process)} training examples from {articles_processed} articles")
    logging.info(f"Dataset type used: {dataset_type}")
    
    # Save processed data to cache
    logging.info("Saving processed data to cache...")
    with open(texts_cache_file, 'wb') as f:
        pickle.dump(texts_to_process, f)
    with open(metadata_cache_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    # # Also save in PyTorch format for compatibility
    # torch.save(texts_to_process, os.path.join(Config.CACHE_DIR, f"{cache_file_prefix}_texts_to_process.pt"))
    # torch.save(metadata, os.path.join(Config.CACHE_DIR, f"{cache_file_prefix}_metadata.pt"))
    
    return texts_to_process, metadata

def analyze_dataset(texts_to_process: List[Tuple[str, List[str]]], metadata: List[Dict[str, Any]]):
    """Analyze and print statistics about the created dataset"""
    logging.info("\n" + "="*50)
    logging.info("Dataset Statistics:")
    logging.info("="*50)
    
    total_examples = len(texts_to_process)
    logging.info(f"Total training examples: {total_examples}")
    
    # Check dataset types
    if metadata and 'dataset_type' in metadata[0]:
        dataset_types = set(m.get('dataset_type', 'unknown') for m in metadata)
        logging.info(f"Dataset types used: {', '.join(dataset_types)}")
    
    # Analyze chunk lengths
    chunk_lengths = [len(text[0]) for text in texts_to_process]
    avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    min_chunk_length = min(chunk_lengths) if chunk_lengths else 0
    max_chunk_length = max(chunk_lengths) if chunk_lengths else 0
    
    logging.info(f"Chunk length - Avg: {avg_chunk_length:.0f}, Min: {min_chunk_length}, Max: {max_chunk_length}")
    
    # Analyze sub-sequences
    num_subsequences = [len(text[1]) for text in texts_to_process]
    avg_subsequences = sum(num_subsequences) / len(num_subsequences) if num_subsequences else 0
    min_subsequences = min(num_subsequences) if num_subsequences else 0
    max_subsequences = max(num_subsequences) if num_subsequences else 0
    
    logging.info(f"Sub-sequences per example - Avg: {avg_subsequences:.1f}, Min: {min_subsequences}, Max: {max_subsequences}")
    
    # Show sample
    if texts_to_process:
        logging.info("\n" + "="*50)
        logging.info("Sample Example:")
        logging.info("="*50)
        sample_idx = random.randint(0, len(texts_to_process) - 1)
        sample_text, sample_subsequences = texts_to_process[sample_idx]
        sample_meta = metadata[sample_idx]
        
        logging.info(f"Source: {sample_meta['source_file']}")
        logging.info(f"Number of sub-sequences: {sample_meta['num_sequences']}")
        logging.info(f"Main chunk preview (first 200 chars): {sample_text[:200]}...")
        logging.info(f"First sub-sequence preview: {sample_subsequences[0][:100]}...")

def main():
    parser = argparse.ArgumentParser(description="Process Wikipedia into training examples")
    parser.add_argument('--num-examples', type=int, default=10000, 
                       help='Number of training examples to create')
    parser.add_argument('--examples-per-article', type=int, default=3,
                       help='Number of examples to extract from each article')
    parser.add_argument('--cache-prefix', type=str, default='wikipedia_val',
                       help='Prefix for cache files')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze existing cached data')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Load existing data and analyze
        texts_cache_file = os.path.join(Config.CACHE_DIR, f"{args.cache_prefix}_texts_to_process.pkl")
        metadata_cache_file = os.path.join(Config.CACHE_DIR, f"{args.cache_prefix}_metadata.pkl")
        
        if os.path.exists(texts_cache_file) and os.path.exists(metadata_cache_file):
            with open(texts_cache_file, 'rb') as f:
                texts_to_process = pickle.load(f)
            with open(metadata_cache_file, 'rb') as f:
                metadata = pickle.load(f)
            analyze_dataset(texts_to_process, metadata)
        else:
            logging.error("No cached data found. Run without --analyze-only first.")
    else:
        # Process Wikipedia and create training examples
        texts_to_process, metadata = process_wikipedia_to_training_examples(
            num_examples=args.num_examples,
            examples_per_article=args.examples_per_article,
            cache_file_prefix=args.cache_prefix
        )
        
        # Analyze the created dataset
        analyze_dataset(texts_to_process, metadata)
        
        logging.info(f"\nData saved to {Config.CACHE_DIR}/")
        logging.info("Files created:")
        logging.info(f"  - {args.cache_prefix}_texts_to_process.pkl")
        logging.info(f"  - {args.cache_prefix}_texts_to_process.pt")
        logging.info(f"  - {args.cache_prefix}_metadata.pkl")
        logging.info(f"  - {args.cache_prefix}_metadata.pt")

if __name__ == "__main__":
    main()