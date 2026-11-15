"""
Bilingual Medical Assistant - Command-Line Chat (Phase 1: English RAG)
----------------------------------------------------------------------
This module implements the retrieval-augmented generation (RAG) pipeline:
1. Takes an English medical question from the user.
2. Retrieves the top relevant MedQuAD records using BM25 from search.py.
3. Formats the retrieved evidence (topic, question, answer, collection, URL).
4. Sends the evidence and user query to a local llama.cpp server running Qwen.
5. Enforces grounding: the model answers strictly from evidence without diagnosing.
6. Displays the generated medical response and citation source URLs.
"""

import sys
import json
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Reuse existing retrieval components from search.py
from search import load_records, build_bm25_index, search, tokenize

# Configuration for the local llama.cpp server
DEFAULT_API_URL = "http://127.0.0.1:8765/v1/chat/completions"
DEFAULT_MODEL_ALIAS = "health-fyp"
DEFAULT_RECORDS_FILE = "records.jsonl"
DEFAULT_TOP_K = 5
DEFAULT_TIMEOUT = 180

SYSTEM_PROMPT = (
    "You are a helpful, professional Medical Assistant providing accurate health "
    "information strictly based on the provided MedQuAD medical evidence.\n\n"
    "Strict Guidelines:\n"
    "1. Answer ONLY using the facts explicitly stated in the provided context evidence. "
    "Do NOT use outside knowledge, speculate, guess, or extrapolate.\n"
    "2. Specificity and Abstention: If the user asks about a specific village, city, location, "
    "or individual person that is not explicitly documented in the evidence, you MUST clearly state "
    "that the provided evidence does not contain statistics or information for that specific location or person. "
    "Do NOT substitute statistics from another country or population (such as United States national statistics) "
    "as an answer for a specific village, city, or individual.\n"
    "3. If the provided evidence does not contain sufficient information to answer the question, "
    "explicitly state: \"The provided medical evidence does not contain sufficient information to answer this question.\"\n"
    "4. Do NOT diagnose the user or prescribe individual treatment plans. Only state general medical facts as documented in the evidence.\n"
    "5. Keep your tone neutral, clear, and informative."
)


def format_evidence_block(results: List[Tuple[int, float, Dict[str, Any]]]) -> str:
    """
    Formats retrieved search results into a structured text block for the LLM prompt.

    Args:
        results: List of (rank, score, record) tuples from search().

    Returns:
        Formatted evidence string.
    """
    evidence_parts = []
    for rank, score, record in results:
        part = (
            f"--- Evidence [{rank}] ---\n"
            f"Topic: {record.get('topic', 'N/A')}\n"
            f"Collection: {record.get('collection', 'N/A')}\n"
            f"Source URL: {record.get('source_url', 'N/A')}\n"
            f"Medical Question: {record.get('question', 'N/A')}\n"
            f"Medical Answer: {record.get('answer', 'N/A')}\n"
        )
        evidence_parts.append(part)

    return "\n".join(evidence_parts)


def call_llm(
    query: str,
    evidence_text: str,
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL_ALIAS,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """
    Calls the local llama.cpp server via urllib.request (OpenAI-compatible chat completions).

    Args:
        query: The user's medical question.
        evidence_text: Structured evidence retrieved from MedQuAD.
        api_url: Local server endpoint URL.
        model: Model alias expected by the server.
        timeout: Network timeout in seconds.

    Returns:
        The generated answer text from the model.
    """
    user_prompt = (
        f"Context Evidence from MedQuAD:\n"
        f"{evidence_text}\n\n"
        f"User Question: {query}\n\n"
        f"Instructions: Based ONLY on the context evidence above, provide a clear and factual answer. "
        f"If the question asks about a specific location or population (such as a specific village, city, or person) "
        f"not present in the evidence, state clearly that the evidence does not contain statistics for that specific location. "
        f"Do not substitute other regional data. Do not provide personalized medical diagnosis."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=api_url,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_bytes = response.read()
    except urllib.error.HTTPError as err:
        error_body = ""
        try:
            error_body = err.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        print(f"\nError: Local LLM server returned HTTP {err.code}: {err.reason}", file=sys.stderr)
        if error_body:
            print(f"Details: {error_body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as err:
        print(
            f"\nError: Could not connect to local llama.cpp server at '{api_url}'.\n"
            f"Reason: {err.reason}\n\n"
            f"Please ensure your llama.cpp server is running on port 8765 with model '{model}'.\n"
            f"Example server start: llama-server -m <path-to-qwen.gguf> --port 8765 --alias {model}",
            file=sys.stderr,
        )
        sys.exit(1)
    except TimeoutError:
        print(f"\nError: Request to local LLM server timed out after {timeout} seconds.", file=sys.stderr)
        sys.exit(1)

    # Parse JSON response
    try:
        resp_json = json.loads(response_bytes.decode("utf-8"))
    except json.JSONDecodeError as err:
        print(f"\nError: Failed to parse JSON response from LLM server: {err}", file=sys.stderr)
        sys.exit(1)

    # Validate OpenAI-compatible response structure
    choices = resp_json.get("choices")
    if not choices or not isinstance(choices, list):
        print(f"\nError: Invalid API response format from LLM server. Expected 'choices' list.", file=sys.stderr)
        print(f"Server response: {resp_json}", file=sys.stderr)
        sys.exit(1)

    first_choice = choices[0]
    message = first_choice.get("message", {})
    content = message.get("content", "")

    if not content or not content.strip():
        print("\nError: The model returned an empty response.", file=sys.stderr)
        sys.exit(1)

    return content.strip()


def display_rag_output(
    query: str,
    answer: str,
    results: List[Tuple[int, float, Dict[str, Any]]],
) -> None:
    """
    Renders the final assistant response and citations in a clean terminal layout.

    Args:
        query: The user's question.
        answer: Generated answer from LLM.
        results: Retrieved search results with URLs.
    """
    print("\n" + "=" * 60)
    print(" Medical Assistant Response")
    print("=" * 60)
    print(f"\nQuestion:\n  {query}\n")
    print("Answer:")
    print(f"  {answer}\n")

    print("=" * 60)
    print(" Supporting MedQuAD Sources")
    print("=" * 60)

    seen_urls = set()
    source_index = 1
    for rank, score, record in results:
        url = record.get("source_url", "").strip()
        topic = record.get("topic", "General Topic")
        collection = record.get("collection", "MedQuAD")

        # Avoid printing duplicate URLs if multiple QA pairs come from the same article
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)

        print(f"[{source_index}] {topic} ({collection})")
        print(f"    URL: {url if url else 'N/A'}")
        source_index += 1

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Bilingual Medical Assistant RAG Chat (Phase 1: English)"
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Medical question in English (e.g. 'What is asthma?')"
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"llama.cpp API URL (default: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ALIAS,
        help=f"Model alias (default: {DEFAULT_MODEL_ALIAS})"
    )
    parser.add_argument(
        "--records_file",
        type=str,
        default=DEFAULT_RECORDS_FILE,
        help=f"Path to JSONL records (default: {DEFAULT_RECORDS_FILE})"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of evidence records to retrieve (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds for LLM generation (default: {DEFAULT_TIMEOUT}s)"
    )

    args = parser.parse_args()

    # 1. Validate non-empty question
    raw_query = " ".join(args.question).strip()
    if not raw_query:
        print("Error: No medical question provided.", file=sys.stderr)
        print("Usage: python chat.py \"<your medical question>\"", file=sys.stderr)
        print("Example: python chat.py \"What is asthma?\"", file=sys.stderr)
        sys.exit(1)

    # 2. Load dataset records (reusing search.py)
    records = load_records(args.records_file)

    # 3. Build/initialize BM25 search index
    bm25, _ = build_bm25_index(records)

    # 4. Retrieve top matching records
    results = search(raw_query, records, bm25, top_k=args.top_k)

    if not results:
        display_rag_output(
            raw_query,
            "The provided medical evidence does not contain sufficient information to answer this question. No relevant records were found in the dataset for this query.",
            []
        )
        return

    # 5. Format evidence for the prompt
    evidence_text = format_evidence_block(results)

    # 6. Call local llama.cpp server
    print(f"\n[1/2] Retrieved {len(results)} MedQuAD evidence records.")
    print(f"[2/2] Consulting '{args.model}' (CPU inference may take 30-90s)...")
    answer = call_llm(
        query=raw_query,
        evidence_text=evidence_text,
        api_url=args.api_url,
        model=args.model,
        timeout=args.timeout,
    )

    # 7. Display final answer and verified source links
    display_rag_output(raw_query, answer, results)


if __name__ == "__main__":
    main()
