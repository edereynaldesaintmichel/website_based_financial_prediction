"""
Deterministic train/val split by source filename.

Uses MD5 hashing so the same document always lands in the same split,
regardless of which pipeline is running or what order the data is loaded.

All three training pipelines import from here to guarantee consistent splits:
  - mlm_training_pipeline/train_mlm_full.py
  - t5_style_training_pipeline/prepare_data.py
  - cls_aggregator_training_pipeline/train.py
"""
import hashlib


def is_val_document(source_file: str, val_ratio: float = 0.1) -> bool:
    """Return True if source_file belongs to the validation set.

    The decision is purely a function of the filename string and val_ratio,
    so it is deterministic and consistent across pipelines / runs.
    """
    h = int(hashlib.md5(source_file.encode()).hexdigest(), 16) % 10000
    return h < val_ratio * 10000


def split_documents(documents, val_ratio=0.1, key="source_file"):
    """Split a list of document dicts into (train, val) lists.

    Each document must have a ``key`` field (default ``source_file``) whose
    value is hashed to decide the split.
    """
    train, val = [], []
    for doc in documents:
        if is_val_document(doc[key], val_ratio):
            val.append(doc)
        else:
            train.append(doc)
    return train, val


def split_source_files(source_files, val_ratio=0.1):
    """Split an iterable of source-file strings into (train, val) sets."""
    train, val = set(), set()
    for sf in source_files:
        if is_val_document(sf, val_ratio):
            val.add(sf)
        else:
            train.add(sf)
    return train, val
