"""
BM25 Keyword Search Module
--------------------------
This script searches ingested MedQuAD records in 'records.jsonl'
using the BM25Okapi algorithm from the rank-bm25 package.

It retrieves the most relevant medical records based on keyword matching
across the topic, question, and answer text.
"""

import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

# Small, maintainable English stopwords set (preserves medical terms like asthma, symptoms, treatment, patients)
ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your",
    "yours", "yourself", "yourselves"
}

# Generic quantity/modifier terms that should not alone qualify an unrelated document as relevant
GENERIC_MODIFIERS = {
    "exact", "number", "numbers", "amount", "many", "much", "total", "rate", "rates",
    "patient", "patients", "person", "people", "case", "cases", "level", "levels",
    "test", "tests", "good", "bad", "high", "low", "new", "old", "first", "last",
    "cause", "causes", "effect", "effects", "help", "information", "info", "use",
    "space", "repair", "make", "take", "get", "give", "know", "see", "come", "think",
    "tell", "show", "find", "need", "like", "time", "day", "year"
}


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Splits text into lowercase alphanumeric word tokens and filters common English stopwords.
    Ensures critical medical terms such as asthma, symptoms, treatment, and patients are preserved.

    Args:
        text: Input string to tokenize.
        remove_stopwords: If True, filters common English function words.

    Returns:
        List of lowercase string tokens.
    """
    if not text:
        return []
    # Convert to lowercase and extract words (letters and numbers)
    words = re.findall(r"\b\w+\b", text.lower())
    if remove_stopwords:
        filtered = [w for w in words if w not in ENGLISH_STOPWORDS]
        return filtered if filtered else words
    return words


def load_records(records_path: str = "records.jsonl") -> List[Dict[str, Any]]:
    """
    Reads records line-by-line from a JSONL file.
    Validates file existence and handles malformed JSON lines cleanly.

    Args:
        records_path: Path to the JSONL file.

    Returns:
        List of record dictionaries.
    """
    path = Path(records_path)

    # 1. Check if the file exists
    if not path.exists():
        print(f"Error: Records file '{records_path}' was not found.", file=sys.stderr)
        print("Please run 'python ingest.py' first to build the dataset index.", file=sys.stderr)
        sys.exit(1)

    records = []
    malformed_count = 0

    # 2. Read each line with UTF-8 encoding
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                records.append(record)
            except json.JSONDecodeError as err:
                malformed_count += 1
                print(f"Warning: Malformed JSON on line {line_num}: {err}", file=sys.stderr)

    # 3. Report any data problems found
    if malformed_count > 0:
        print(f"Notice: Encounted {malformed_count} malformed JSON line(s) in '{records_path}'.", file=sys.stderr)

    # 4. Validate that the dataset is not empty
    if not records:
        print(f"Error: No valid records found in '{records_path}'. The dataset is empty.", file=sys.stderr)
        sys.exit(1)

    return records


def build_bm25_index(records: List[Dict[str, Any]]) -> Tuple[BM25Okapi, List[List[str]]]:
    """
    Builds the search corpus and initializes the BM25Okapi model.
    Combines each record's topic, question, and answer for retrieval.

    Args:
        records: List of record dictionaries.

    Returns:
        A tuple of (BM25Okapi instance, tokenized corpus).
    """
    tokenized_corpus = []

    for record in records:
        # Combine topic, question, and answer to form the searchable document text
        topic = record.get("topic", "")
        question = record.get("question", "")
        answer = record.get("answer", "")

        combined_doc_text = f"{topic} {question} {answer}"
        tokens = tokenize(combined_doc_text)
        tokenized_corpus.append(tokens)

    # Initialize BM25 model
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


def search(
    query: str,
    records: List[Dict[str, Any]],
    bm25: BM25Okapi,
    top_k: int = 5,
) -> List[Tuple[int, float, Dict[str, Any]]]:
    """
    Performs BM25 search for a given query string against indexed records,
    with relevance filtering to avoid returning documents that only match
    incidental modifiers or generic non-medical words.

    Args:
        query: User search query.
        records: List of ingested records.
        bm25: Pre-built BM25Okapi index.
        top_k: Number of top results to return.

    Returns:
        List of tuples: (rank, score, record).
    """
    raw_words = re.findall(r"\b\w+\b", query.lower())
    if not raw_words:
        print("Error: Search query cannot be empty or contain only punctuation.", file=sys.stderr)
        sys.exit(1)

    # Detect queries containing only English stopwords
    content_words = [w for w in raw_words if w not in ENGLISH_STOPWORDS]
    if not content_words:
        print("Notice: Query contains only generic stop words. Please provide a specific medical question or condition.", file=sys.stderr)
        return []

    query_tokens = tokenize(query)

    # Compute BM25 relevance scores for all records
    scores = bm25.get_scores(query_tokens)

    # Sort record indices by score in descending order
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    # Identify core substantive query terms (excluding generic modifiers)
    core_query_tokens = [t for t in query_tokens if t not in GENERIC_MODIFIERS]

    results = []
    for idx in ranked_indices:
        score = float(scores[idx])
        if score <= 0.0:
            continue

        record = records[idx]
        doc_tokens = set(tokenize(f"{record.get('topic', '')} {record.get('question', '')} {record.get('answer', '')}"))

        # Filtering logic:
        # 1. If query contains core medical/content terms, the document MUST match at least one core term.
        #    This prevents unrelated documents matching only "exact" or "number" from appearing.
        #    It also ensures queries about absent topics (e.g. quantum rockets) return 0 results.
        if core_query_tokens:
            if not any(term in doc_tokens for term in core_query_tokens):
                continue
        else:
            # If query only has generic words, require matching at least half of the query terms
            matching = [t for t in query_tokens if t in doc_tokens]
            if len(matching) < max(1, len(query_tokens) // 2):
                continue

        results.append((len(results) + 1, score, record))
        if len(results) >= top_k:
            break

    return results


def display_results(query: str, results: List[Tuple[int, float, Dict[str, Any]]]) -> None:
    """
    Formats and prints search results cleanly to the console.

    Args:
        query: The user query searched.
        results: List of (rank, score, record) tuples.
    """
    print(f"\n=======================================================")
    print(f" Search Results for: \"{query}\"")
    print(f" Total Results Returned: {len(results)}")
    print(f"=======================================================\n")

    if not results:
        print("Notice: No sufficiently relevant records found in MedQuAD matching the core topic.")
        print("-" * 55)
        return

    for rank, score, record in results:
        print(f"Rank #{rank} | Score: {score:.4f}")
        print(f"Record ID:   {record.get('id', 'N/A')}")
        print(f"Collection:  {record.get('collection', 'N/A')}")
        print(f"Topic:       {record.get('topic', 'N/A')}")
        print(f"Source URL:  {record.get('source_url', 'N/A')}")
        print(f"Question:    {record.get('question', 'N/A')}")

        answer = record.get("answer", "")
        # Show full or clean answer preview
        if len(answer) > 300:
            answer_preview = answer[:300].rstrip() + "... [truncated]"
        else:
            answer_preview = answer
        print(f"Answer:      {answer_preview}")
        print("-" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="BM25 Keyword Search for MedQuAD Bilingual Medical Assistant"
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
        help="Maximum number of results to display (default: 5)"
    )
    parser.add_argument(
        "--records_file",
        type=str,
        default="records.jsonl",
        help="Path to records.jsonl file (default: records.jsonl)"
    )

    args = parser.parse_args()

    # Combine query arguments into a single string
    raw_query = " ".join(args.query).strip()

    # Validate non-empty query
    if not raw_query:
        print("Error: No search query provided.", file=sys.stderr)
        print("Usage: python search.py \"<your question or topic>\"", file=sys.stderr)
        print("Example: python search.py \"What is asthma?\"", file=sys.stderr)
        sys.exit(1)

    # 1. Load dataset records
    records = load_records(args.records_file)

    # 2. Build index
    bm25, _ = build_bm25_index(records)

    # 3. Perform search
    results = search(raw_query, records, bm25, top_k=args.top_k)

    # 4. Display results
    display_results(raw_query, results)


if __name__ == "__main__":
    main()
