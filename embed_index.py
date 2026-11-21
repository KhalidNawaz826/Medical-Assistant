"""
Semantic Embedding Index Generator
-----------------------------------
This script precomputes normalized dense embeddings for MedQuAD records
in 'records.jsonl' using the 'intfloat/multilingual-e5-small' model via
sentence-transformers.

The generated vector index is stored as 'embeddings.npy' alongside an
'embedding_metadata.json' file containing provenance and validation fingerprints.
"""

import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_RECORDS_FILE = "records.jsonl"
DEFAULT_OUTPUT_EMBEDDINGS = "embeddings.npy"
DEFAULT_OUTPUT_METADATA = "embedding_metadata.json"
DEFAULT_BATCH_SIZE = 32
DOCUMENT_PREFIX = "passage: "


def compute_file_sha256(filepath: Path) -> str:
    """Calculates SHA256 hash of a file for integrity and provenance tracking."""
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_records_fingerprint(records: List[Dict[str, Any]]) -> str:
    """Computes a SHA256 fingerprint over ordered record IDs to ensure 1-to-1 correspondence."""
    hasher = hashlib.sha256()
    for rec in records:
        hasher.update(rec.get("id", "").encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def load_records(records_path: str) -> List[Dict[str, Any]]:
    """Loads and validates JSONL records line by line."""
    path = Path(records_path)
    if not path.exists():
        print(f"Error: Records file '{records_path}' not found.", file=sys.stderr)
        sys.exit(1)

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                rec = json.loads(line_str)
                records.append(rec)
            except json.JSONDecodeError as err:
                print(f"Error: Malformed JSON on line {line_num}: {err}", file=sys.stderr)
                sys.exit(1)

    if not records:
        print(f"Error: No valid records found in '{records_path}'.", file=sys.stderr)
        sys.exit(1)

    return records


def format_passages(records: List[Dict[str, Any]], prefix: str = DOCUMENT_PREFIX) -> List[str]:
    """
    Constructs the passage text for each record using topic, question, and answer.
    multilingual-e5 strictly requires the 'passage: ' prefix for documents.
    """
    passages = []
    for rec in records:
        topic = rec.get("topic", "").strip()
        question = rec.get("question", "").strip()
        answer = rec.get("answer", "").strip()
        combined = f"{topic} {question} {answer}".strip()
        passages.append(f"{prefix}{combined}")
    return passages


def build_embedding_index(
    records_file: str = DEFAULT_RECORDS_FILE,
    output_embeddings: str = DEFAULT_OUTPUT_EMBEDDINGS,
    output_metadata: str = DEFAULT_OUTPUT_METADATA,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Builds the dense embedding index and saves embeddings.npy and metadata.json.
    Uses safe temporary file writing to avoid partial or corrupted index files.
    """
    records_path = Path(records_file)
    emb_path = Path(output_embeddings)
    meta_path = Path(output_metadata)

    print(f"[1/4] Loading records from '{records_path}'...")
    records = load_records(records_file)
    record_count = len(records)
    print(f"      Loaded {record_count} records.")

    # Calculate fingerprints for verification
    file_sha256 = compute_file_sha256(records_path)
    id_fingerprint = compute_records_fingerprint(records)
    print(f"      records.jsonl SHA256: {file_sha256[:12]}...")
    print(f"      Ordered ID fingerprint: {id_fingerprint[:12]}...")

    print(f"[2/4] Formatting document passages with '{DOCUMENT_PREFIX}' prefix...")
    passages = format_passages(records, prefix=DOCUMENT_PREFIX)
    if len(passages) != record_count:
        print("Error: Passage count does not match record count.", file=sys.stderr)
        sys.exit(1)

    print(f"[3/4] Loading model '{model_name}' on CPU...")
    start_time = time.time()
    try:
        model = SentenceTransformer(model_name, device="cpu")
    except Exception as err:
        print(f"Error: Failed to load SentenceTransformer model '{model_name}': {err}", file=sys.stderr)
        sys.exit(1)

    print(f"[4/4] Encoding {record_count} passages (batch_size={batch_size}, normalize=True)...")
    encoding_start = time.time()
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    encoding_duration = time.time() - encoding_start
    total_duration = time.time() - start_time

    # Validation checks on computed array
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype=np.float32)
    else:
        embeddings = embeddings.astype(np.float32)

    if embeddings.shape[0] != record_count:
        print(f"Error: Embedding row count {embeddings.shape[0]} != record count {record_count}", file=sys.stderr)
        sys.exit(1)

    if not np.all(np.isfinite(embeddings)):
        print("Error: Generated embeddings contain non-finite values (NaN or Inf).", file=sys.stderr)
        sys.exit(1)

    # Check vector norms
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print("Warning: Some embedding vectors do not have unit norm (~1.0).", file=sys.stderr)

    vector_dim = int(embeddings.shape[1])
    print(f"\nEncoding complete in {encoding_duration:.2f}s (Total: {total_duration:.2f}s).")
    print(f"Vector Matrix Shape: {embeddings.shape}, Data Type: {embeddings.dtype}")

    # Prepare metadata
    metadata = {
        "model_name": model_name,
        "record_count": record_count,
        "embedding_dimension": vector_dim,
        "dtype": str(embeddings.dtype),
        "normalized": True,
        "document_prefix": DOCUMENT_PREFIX,
        "source_records_file": str(records_path.name),
        "source_records_sha256": file_sha256,
        "ordered_record_id_fingerprint": id_fingerprint,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "encoding_duration_seconds": round(encoding_duration, 2),
    }

    # Safe atomic write via temporary files (must end in .npy for np.save)
    tmp_emb_path = emb_path.with_name(f"{emb_path.stem}_tmp.npy")
    tmp_meta_path = meta_path.with_name(f"{meta_path.stem}_tmp.json")

    try:
        np.save(tmp_emb_path, embeddings)
        with tmp_meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Atomic replace
        tmp_emb_path.replace(emb_path)
        tmp_meta_path.replace(meta_path)
    except Exception as err:
        print(f"Error saving index files: {err}", file=sys.stderr)
        if tmp_emb_path.exists():
            tmp_emb_path.unlink()
        if tmp_meta_path.exists():
            tmp_meta_path.unlink()
        sys.exit(1)

    print(f"Saved vector index: '{emb_path}' ({emb_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"Saved metadata:     '{meta_path}' ({meta_path.stat().st_size} bytes)")
    return embeddings, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Precompute semantic embeddings for MedQuAD records using E5"
    )
    parser.add_argument(
        "--records_file",
        type=str,
        default=DEFAULT_RECORDS_FILE,
        help=f"Path to input records.jsonl (default: {DEFAULT_RECORDS_FILE})"
    )
    parser.add_argument(
        "--output_embeddings",
        type=str,
        default=DEFAULT_OUTPUT_EMBEDDINGS,
        help=f"Path to output embeddings.npy (default: {DEFAULT_OUTPUT_EMBEDDINGS})"
    )
    parser.add_argument(
        "--output_metadata",
        type=str,
        default=DEFAULT_OUTPUT_METADATA,
        help=f"Path to output metadata JSON (default: {DEFAULT_OUTPUT_METADATA})"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"HuggingFace model identifier (default: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Inference batch size on CPU (default: {DEFAULT_BATCH_SIZE})"
    )

    args = parser.parse_args()
    build_embedding_index(
        records_file=args.records_file,
        output_embeddings=args.output_embeddings,
        output_metadata=args.output_metadata,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
