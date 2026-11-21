"""
Semantic Retrieval Module (Multilingual E5)
--------------------------------------------
This module performs standalone dense semantic retrieval using precomputed
normalized embeddings from 'embeddings.npy' and the 'intfloat/multilingual-e5-small'
model via sentence-transformers.

It enforces strict provenance validation against 'embedding_metadata.json'
and 'records.jsonl' to ensure no stale or mismatched index is ever queried.
"""

import sys
import json
import re
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# Reuse baseline record loading from search.py
from search import load_records


DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_RECORDS_FILE = "records.jsonl"
DEFAULT_EMBEDDINGS_FILE = "embeddings.npy"
DEFAULT_METADATA_FILE = "embedding_metadata.json"
QUERY_PREFIX = "query: "


def compute_file_sha256(filepath: Path) -> str:
    """Calculates SHA256 hash of a file for integrity verification."""
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_records_fingerprint(records: List[Dict[str, Any]]) -> str:
    """Computes a SHA256 fingerprint over ordered record IDs."""
    hasher = hashlib.sha256()
    for rec in records:
        hasher.update(rec.get("id", "").encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def validate_index_integrity(
    records: List[Dict[str, Any]],
    embeddings: np.ndarray,
    metadata: Dict[str, Any],
    records_path: Path,
    expected_model_name: str = DEFAULT_MODEL_NAME,
) -> None:
    """
    Validates that the loaded embeddings and metadata strictly match the active
    records.jsonl dataset and configuration. Raises ValueError if any mismatch is found.
    """
    # 1. Model name validation
    meta_model = metadata.get("model_name")
    if meta_model != expected_model_name:
        raise ValueError(
            f"Model mismatch: requested '{expected_model_name}', but index was built with '{meta_model}'."
        )

    # 2. Source file SHA256 validation
    computed_file_sha = compute_file_sha256(records_path)
    meta_file_sha = metadata.get("source_records_sha256")
    if computed_file_sha != meta_file_sha:
        raise ValueError(
            f"Corpus SHA256 mismatch: '{records_path.name}' hash ({computed_file_sha}) "
            f"does not match index metadata hash ({meta_file_sha}). Index is stale."
        )

    # 3. Record count and dimension validation
    rec_count = len(records)
    meta_count = metadata.get("record_count")
    if rec_count != meta_count:
        raise ValueError(
            f"Record count mismatch: '{records_path.name}' has {rec_count} records, but metadata specifies {meta_count}."
        )

    if embeddings.shape[0] != rec_count:
        raise ValueError(
            f"Embedding row count mismatch: matrix has {embeddings.shape[0]} rows, expected {rec_count}."
        )

    meta_dim = metadata.get("embedding_dimension")
    if embeddings.shape[1] != meta_dim:
        raise ValueError(
            f"Embedding dimension mismatch: matrix has {embeddings.shape[1]} dimensions, expected {meta_dim}."
        )

    # 4. Data type validation
    meta_dtype = metadata.get("dtype")
    if str(embeddings.dtype) != meta_dtype:
        raise ValueError(
            f"Embedding dtype mismatch: matrix is {embeddings.dtype}, expected {meta_dtype}."
        )

    # 5. Ordered record ID fingerprint validation
    computed_id_fp = compute_records_fingerprint(records)
    meta_id_fp = metadata.get("ordered_record_id_fingerprint")
    if computed_id_fp != meta_id_fp:
        raise ValueError(
            "Record ID ordering fingerprint mismatch. The order or content of record IDs has changed since indexing."
        )

    # 6. Finite values validation
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Embedding index contains non-finite values (NaN or Inf).")


class SemanticRetriever:
    """
    Handles lazy model loading, index validation, and cosine-similarity semantic search.
    """

    def __init__(
        self,
        records_file: str = DEFAULT_RECORDS_FILE,
        embeddings_file: str = DEFAULT_EMBEDDINGS_FILE,
        metadata_file: str = DEFAULT_METADATA_FILE,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
    ):
        self.records_path = Path(records_file)
        self.embeddings_path = Path(embeddings_file)
        self.metadata_path = Path(metadata_file)
        self.model_name = model_name
        self.device = device

        # Check required files
        for p in (self.records_path, self.embeddings_path, self.metadata_path):
            if not p.exists():
                raise FileNotFoundError(f"Required file not found: '{p}'")

        # Load records, metadata, and vectors
        self.records = load_records(str(self.records_path))
        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.embeddings = np.load(str(self.embeddings_path))

        # Perform strict validation
        validate_index_integrity(
            records=self.records,
            embeddings=self.embeddings,
            metadata=self.metadata,
            records_path=self.records_path,
            expected_model_name=self.model_name,
        )

        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-loads the SentenceTransformer model on demand."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Performs dense semantic search using E5 query encoding and cosine similarity.

        Args:
            query: User's question or search query.
            top_k: Maximum number of top records to return.

        Returns:
            List of (rank, score, record) tuples matching search.py convention.
        """
        # 1. Empty or punctuation-only check
        words = re.findall(r"\b\w+\b", query.lower())
        if not words:
            return []

        # 2. Format query with required E5 prefix
        formatted_query = f"{QUERY_PREFIX}{query.strip()}"

        # 3. Encode query to normalized 384-d vector
        query_vec = self.model.encode(
            formatted_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        if not np.all(np.isfinite(query_vec)):
            raise ValueError("Query embedding returned non-finite values.")

        query_vec = query_vec.astype(np.float32)

        # 4. Compute cosine similarity (dot product of L2-normalized vectors)
        scores = np.dot(self.embeddings, query_vec)

        # 5. Sort indices descending
        ranked_indices = np.argsort(scores)[::-1]

        # 6. Format results according to (rank, score, record) convention
        results = []
        for rank_num, idx in enumerate(ranked_indices[:top_k], start=1):
            score = float(scores[idx])
            record = self.records[idx]
            results.append((rank_num, score, record))

        return results


def display_results(query: str, results: List[Tuple[int, float, Dict[str, Any]]]) -> None:
    """
    Cleanly formats and displays semantic search results in terminal.
    """
    print(f"\n=======================================================")
    print(f" Semantic Search Results for: \"{query}\"")
    print(f" Total Results Returned: {len(results)}")
    print(f"=======================================================\n")

    if not results:
        print("Notice: Query was empty or no results were returned.")
        print("-" * 55)
        return

    for rank, score, record in results:
        print(f"Rank #{rank} | Cosine Similarity: {score:.4f}")
        print(f"Record ID:   {record.get('id', 'N/A')}")
        print(f"Collection:  {record.get('collection', 'N/A')}")
        print(f"Topic:       {record.get('topic', 'N/A')}")
        print(f"Source URL:  {record.get('source_url', 'N/A')}")
        print(f"Question:    {record.get('question', 'N/A')}")

        answer = record.get("answer", "")
        if len(answer) > 250:
            preview = answer[:250].rstrip() + "... [truncated]"
        else:
            preview = answer
        print(f"Answer:      {preview}")
        print("-" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Semantic Retrieval using Multilingual E5"
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Search query terms (e.g. 'What is asthma?')"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)"
    )
    parser.add_argument(
        "--records_file",
        type=str,
        default=DEFAULT_RECORDS_FILE,
        help=f"Path to records JSONL (default: {DEFAULT_RECORDS_FILE})"
    )
    parser.add_argument(
        "--embeddings_file",
        type=str,
        default=DEFAULT_EMBEDDINGS_FILE,
        help=f"Path to embeddings NPY (default: {DEFAULT_EMBEDDINGS_FILE})"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=DEFAULT_METADATA_FILE,
        help=f"Path to metadata JSON (default: {DEFAULT_METADATA_FILE})"
    )

    args = parser.parse_args()
    raw_query = " ".join(args.query).strip()

    if not raw_query:
        print("Error: No search query provided.", file=sys.stderr)
        print("Usage: python embed_search.py \"<your question or topic>\"", file=sys.stderr)
        print("Example: python embed_search.py \"What is asthma?\"", file=sys.stderr)
        sys.exit(1)

    retriever = SemanticRetriever(
        records_file=args.records_file,
        embeddings_file=args.embeddings_file,
        metadata_file=args.metadata_file,
    )

    results = retriever.search(raw_query, top_k=args.top_k)
    display_results(raw_query, results)


if __name__ == "__main__":
    main()
