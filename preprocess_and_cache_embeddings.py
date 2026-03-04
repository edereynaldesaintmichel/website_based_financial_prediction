def preprocess_and_cache_embeddings(file_paths, tokenizer, base_model, device, cache_file_prefix,
                                    max_examples=None, examples_per_file=4, batch_size=8):
    """
    Pre-process files and create cached embeddings with batch processing.
    Now with checkpoint support, dynamic batch sizing, and memory-efficient processing.
    """
    gc.collect()

    # Check for existing checkpoints to determine start position
    start_idx = 0
    checkpoint_number = 6

    # Find the latest checkpoint
    # checkpoint_pattern = f"/content/drive/MyDrive/website predictor/{cache_file_prefix}_checkpoint_*.pkl"
    # import glob
    # existing_checkpoints = sorted(glob.glob(checkpoint_pattern))

    # if existing_checkpoints:
    #     # Get the latest checkpoint
    #     latest_checkpoint = existing_checkpoints[-1]
    #     # Extract checkpoint number from filename
    #     checkpoint_number = int(latest_checkpoint.split('_checkpoint_')[1].split('.pkl')[0])

    #     print(f"Found {len(existing_checkpoints)} existing checkpoint(s). Loading latest: {latest_checkpoint}")
    #     with open(latest_checkpoint, 'rb') as f:
    #         checkpoint_data = pickle.load(f)
    #         start_idx = checkpoint_data['last_processed_idx']
    #         total_processed = checkpoint_data.get('total_processed', start_idx)
    #     print(f"Resuming from index {start_idx}")
    #     checkpoint_number += 1  # Increment for next checkpoint
    # else:
    #     start_idx = 458988  # Your specified start index if no checkpoints exist
    #     print(f"No checkpoints found. Starting from index {start_idx}")

    start_idx = 540004  # Your specified start index if no checkpoints exist


    # Load and merge metadata files
    print("Loading metadata files...")
    with open("/content/drive/MyDrive/website predictor/website_metadata.pkl", 'rb') as f:
        website_metadata = pickle.load(f)
    with open("/content/drive/MyDrive/website predictor/wikipedia_metadata.pkl", 'rb') as f:
        wikipedia_metadata = pickle.load(f)
    metadata = website_metadata + wikipedia_metadata
    del website_metadata, wikipedia_metadata  # Free memory
    gc.collect()

    # Load and merge texts_to_process files
    print("Loading texts_to_process files...")
    with open("/content/drive/MyDrive/website predictor/website_texts_to_process.pkl", 'rb') as f:
        website_texts = pickle.load(f)
    with open("/content/drive/MyDrive/website predictor/wikipedia_texts_to_process.pkl", 'rb') as f:
        wikipedia_texts = pickle.load(f)
    texts_to_process = website_texts + wikipedia_texts
    del website_texts, wikipedia_texts  # Free memory
    gc.collect()

    total_data_points = len(texts_to_process)
    print(f"Total data points in dataset: {total_data_points}")

    # Sort by length
    print("Calculating text lengths for sorting...")
    lengths = [len(text_tuple[0]) for text_tuple in texts_to_process]
    sorted_data = sorted(zip(lengths, texts_to_process, metadata), key=lambda x: x[0])
    _, sorted_texts_to_process, sorted_metadata = zip(*sorted_data)

    # Convert to lists for slicing
    sorted_texts_to_process = list(sorted_texts_to_process)
    sorted_metadata = list(sorted_metadata)

    del lengths, sorted_data, texts_to_process, metadata  # Free memory
    gc.collect()

    # IMPORTANT: Discard all data before start_idx to save RAM
    if start_idx > 0:
        print(f"Discarding {start_idx} already processed items to save RAM...")
        sorted_texts_to_process = sorted_texts_to_process[start_idx:]
        sorted_metadata = sorted_metadata[start_idx:]
        gc.collect()
        print(f"Remaining items to process: {len(sorted_texts_to_process)}")

    print("Data sorted and trimmed. Starting embedding computation...")

    # Dynamic batch size function
    def get_dynamic_batch_size(text_length):
        """
        Dynamically adjust batch size based on text length.
        Shorter texts can have larger batch sizes.
        """
        return 4

    # Process embeddings in batches
    checkpoint_frequency = 4000  # Save checkpoint every N batches
    batches_since_checkpoint = 0
    all_examples = []  # This will be cleared after each checkpoint

    with torch.no_grad():
        # i now indexes into the trimmed list (starting from 0)
        # global_idx tracks position in the original dataset
        i = 0
        global_idx = start_idx

        pbar = tqdm(total=total_data_points, initial=start_idx, desc="Computing embeddings")

        while i < len(sorted_texts_to_process):
            # Get dynamic batch size based on current text length
            current_text_length = len(sorted_texts_to_process[i][0])
            current_batch_size = get_dynamic_batch_size(current_text_length)

            # Adjust batch size to not exceed remaining items
            actual_batch_size = min(current_batch_size, len(sorted_texts_to_process) - i)

            batch_texts = sorted_texts_to_process[i:i+actual_batch_size]
            batch_meta = sorted_metadata[i:i+actual_batch_size]

            # Process target texts
            target_texts = [t[0] for t in batch_texts]
            all_sub_texts = []
            sub_text_indices = []

            # Collect all sub_texts from the current batch and their original indices
            for batch_id_in_batch, (_, sub_texts) in enumerate(batch_texts):
                for sub_id_in_example, sub in enumerate(sub_texts):
                    all_sub_texts.append((sub, sub_id_in_example))
                    sub_text_indices.append(batch_id_in_batch)

            # Sort all collected sub_texts by length for efficiency
            zipped_sub_texts = sorted(list(zip(all_sub_texts, sub_text_indices)), key=lambda x: len(x[0][0]))

            target_tokens = tokenizer(target_texts, return_tensors='pt', padding=True,
                                     truncation=True, max_length=Config.MAX_CONTEXT_LEN).to(device)
            target_outputs = base_model(**target_tokens)
            target_embeddings = target_outputs.last_hidden_state[:, 0, :].cpu().numpy()
            del target_tokens, target_outputs

            batch_embeddings = {batch_id: [] for batch_id in range(actual_batch_size)}

            # Process sub-texts
            if zipped_sub_texts:
                sub_batch_size = 12
                for j in range(0, len(zipped_sub_texts), sub_batch_size):
                    current_sub_batch_data = zipped_sub_texts[j:j+sub_batch_size]

                    sub_batch_texts_to_tokenize = [element[0][0] for element in current_sub_batch_data]

                    sub_tokens = tokenizer(sub_batch_texts_to_tokenize, return_tensors='pt', padding=True,
                                         truncation=True, max_length=Config.MAX_CONTEXT_LEN).to(device)
                    sub_outputs = base_model(**sub_tokens)
                    embeddings = sub_outputs.last_hidden_state[:, 0, :].cpu().numpy()

                    for idx, embedding in enumerate(embeddings):
                        batch_id = current_sub_batch_data[idx][1]
                        sub_id = current_sub_batch_data[idx][0][1]
                        batch_embeddings[batch_id].append((embedding, sub_id))

                    del sub_tokens, sub_outputs

            # Reorganize embeddings by example
            for idx, meta in enumerate(batch_meta):
                example_sub_embeddings_list = [element[0] for element in
                                               sorted(batch_embeddings[idx], key=lambda x: x[1])]
                example_sub_embeddings = np.array(example_sub_embeddings_list)

                all_examples.append({
                    'input_embeddings': example_sub_embeddings,
                    'target_embedding': target_embeddings[idx],
                    'num_sequences': meta['num_sequences'],
                    'source_file': meta['source_file']
                })

            # Update progress
            i += actual_batch_size
            global_idx += actual_batch_size
            pbar.update(actual_batch_size)
            batches_since_checkpoint += 1

            # Save checkpoint periodically and clear all_examples
            if batches_since_checkpoint >= checkpoint_frequency or i >= len(sorted_texts_to_process):
                if all_examples:  # Only save if we have examples
                    checkpoint_file = f"/content/drive/MyDrive/website predictor/{cache_file_prefix}_checkpoint_{checkpoint_number:04d}.pkl"
                    print(f"\nSaving checkpoint {checkpoint_number} at global index {global_idx} with {len(all_examples)} examples...")

                    checkpoint_data = {
                        'examples': all_examples,
                        'last_processed_idx': global_idx,
                        'total_items': total_data_points,
                    }

                    # Save to temporary file first, then rename (atomic operation)
                    temp_checkpoint = checkpoint_file + '.tmp'
                    with open(temp_checkpoint, 'wb') as f:
                        pickle.dump(checkpoint_data, f)
                    # Atomic rename to avoid corruption
                    os.replace(temp_checkpoint, checkpoint_file)

                    print(f"Checkpoint {checkpoint_number} saved. Progress: {global_idx}/{total_data_points}")
                    print(f"Clearing {len(all_examples)} examples from memory...")

                    # Clear all_examples to free RAM
                    all_examples = []
                    checkpoint_number += 1
                    batches_since_checkpoint = 0

                    # Force garbage collection
                    gc.collect()

                    # Clear GPU cache
                    if device == 'cuda':
                        torch.cuda.empty_cache()

        pbar.close()

    print(f"Processing complete! Created {checkpoint_number} checkpoint files.")
    print(f"Total examples processed: {global_idx}")

    # Return information about the checkpoints created
    checkpoint_info = {
        'num_checkpoints': checkpoint_number,
        'total_processed': global_idx,
        'checkpoint_pattern': f"{cache_file_prefix}_checkpoint_*.pkl"
    }

    return checkpoint_info


def sensible_split(text, num_chunks):
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