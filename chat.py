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
import re
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Reuse existing retrieval components from search.py, embed_search.py, and hybrid_search.py
from search import load_records, build_bm25_index, search, tokenize
from embed_search import SemanticRetriever
from hybrid_search import HybridRetriever

# Configuration for the local llama.cpp server and retrievers
DEFAULT_API_URL = "http://127.0.0.1:8765/v1/chat/completions"
DEFAULT_MODEL_ALIAS = "health-fyp"
DEFAULT_RECORDS_FILE = "records.jsonl"
DEFAULT_TOP_K = 5
DEFAULT_TIMEOUT = 180
DEFAULT_RETRIEVER = "hybrid"
DEFAULT_CANDIDATE_POOL = 20
DEFAULT_RRF_K = 60

# Canonical abstention responses
ENGLISH_ABSTENTION = (
    "The provided medical evidence does not contain sufficient information to answer this question."
)
ROMAN_URDU_ABSTENTION = (
    "Faraham kardah tibbi saboot is sawal ka jawab denay kay liye kafi maloomat nahi rakhta."
)

# Characteristic Roman Urdu markers for fallback routing
ROMAN_URDU_MARKERS = {
    "kya", "kia", "hai", "hain", "ki", "ka", "kay", "ke", "ko", "se", "mein",
    "aur", "hota", "hoti", "hote", "waja", "wajah", "alamat", "ilaj", "ilaaj", "dard",
    "karein", "karo", "mera", "meri", "meray", "gaon", "gao", "bataiye", "batao",
    "kis", "kaise", "kese", "kyun", "kyu", "mareez", "kitnay", "kitna", "kitni"
}

# Initial verified mappings of Roman Urdu medical terms to MedQuAD English entities
ROMAN_URDU_MEDICAL_SYNONYMS = {
    # Multi-word compound phrases (longer phrases first)
    "pait ka dard": "abdominal pain",
    "pait dard": "abdominal pain",
    # Single-word terms
    "dama": "asthma",
    "khaansi": "cough",
    "khansi": "cough",
}


def normalize_medical_synonyms(text: str) -> str:
    """
    Safely normalizes recognized Roman Urdu medical terms to English MedQuAD
    topic entities for retrieval, using case-insensitive whole-word boundaries.
    Processes longer phrases before shorter ones to prevent partial matching.
    """
    normalized = text
    sorted_terms = sorted(ROMAN_URDU_MEDICAL_SYNONYMS.keys(), key=len, reverse=True)
    for term in sorted_terms:
        target = ROMAN_URDU_MEDICAL_SYNONYMS[term]
        pattern = r"\b" + re.escape(term) + r"\b"
        normalized = re.sub(pattern, target, normalized, flags=re.IGNORECASE)
    return normalized


def is_roman_urdu(text: str) -> bool:
    """Detects whether an input text contains characteristic Roman Urdu lexical markers."""
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    return bool(tokens & ROMAN_URDU_MARKERS)


SYSTEM_PROMPT = (
    "You are a helpful, professional bilingual Medical Assistant providing accurate health "
    "information strictly based on the provided MedQuAD medical evidence.\n\n"
    "Strict Guidelines:\n"
    "1. Evidence Grounding: Answer ONLY using the facts explicitly stated in the provided context evidence. "
    "Do NOT use outside medical knowledge, speculate, guess, or extrapolate. "
    "Do NOT diagnose the user or prescribe individual treatment plans. Only state general medical facts documented in the evidence.\n"
    "2. Language Matching & Translation:\n"
    "   - If the user asks in English, provide your answer in English.\n"
    "   - If the user asks in Roman Urdu (Urdu written in Latin script), provide your answer "
    "in clear, natural, grammatically correct Roman Urdu using Latin script. Translate the supported facts faithfully "
    "without altering their medical meaning or adding outside facts. Avoid redundant English parenthetical translations for every term unless necessary for clarity.\n"
    "   - Medical Terminology & Symptoms in Roman Urdu:\n"
    "     * Translate 'wheezing' clearly as 'saans lete waqt seeti jaisi awaaz aana'.\n"
    "     * Medically distinct concepts and anatomical regions must remain strictly distinct during translation. "
    "In particular, 'chest tightness' relates to the chest and must NOT be translated as abdominal pain ('pait dard' or 'pait ka dard'); "
    "use accurate Roman Urdu terms such as 'seene mein jakran' or 'seene mein tangi'. "
    "Conversely, 'pait dard' strictly refers to abdominal/stomach pain.\n"
    "     * When translating clinical nuances such as 'Not all people who have asthma have these symptoms. Having these symptoms doesn't always mean that you have asthma', "
    "express this in natural Roman Urdu (e.g., 'Zaroori nahi ke har mareez mein yeh tamam alamat zahir hon, aur in alamat ka hona laazmi tor par dama ki tasdeeq nahi karta').\n"
    "3. Semantic Equivalence for Roman Urdu Terms:\n"
    "   - Recognized Roman Urdu medical terms refer directly to English medical topics in the retrieved evidence "
    "(e.g., 'dama' refers to asthma; 'pait dard' and 'pait ka dard' refer to abdominal pain; 'khaansi' and 'khansi' refer to cough).\n"
    "   - When the retrieved evidence covers the matching medical condition and specifically answers the user's question, treat it as relevant and sufficient evidence. "
    "Do NOT treat evidence as missing or insufficient merely because the original Roman Urdu term does not literally appear in the English text.\n"
    "   - If the retrieved evidence covers a general topic (such as anatomy or general descriptions) but does NOT explicitly answer the specific question asked (such as causes or specific treatments), treat the evidence as insufficient.\n"
    "4. Specificity & Location Abstention:\n"
    "   - If the user asks about a specific village, city, location, or individual not explicitly documented in the evidence, "
    "you MUST treat this as insufficient evidence. Do NOT provide or append statistics from other populations or countries (such as United States data).\n"
    "   - For English questions, state ONLY: \"The provided medical evidence does not contain sufficient information to answer this question.\"\n"
    "   - For Roman Urdu questions, state ONLY: \"Faraham kardah tibbi saboot is sawal ka jawab denay kay liye kafi maloomat nahi rakhta.\"\n"
    "5. Mutually Exclusive Output (Answer vs. Abstain):\n"
    "   - Determine whether the evidence supports the requested information (by medical concept, not by literal word match).\n"
    "   - If the evidence supports the requested question, output the factual, grounded answer ONLY. Never append an abstention sentence to a supported answer.\n"
    "   - If the evidence is genuinely insufficient to answer the question, output ONLY the designated canonical abstention sentence for that language. "
    "Do NOT include any introductory sentence, preamble, conversational phrase, partial answer, explanation, or extra text before or after the sentence. "
    "The entire output must consist of nothing other than the exact canonical sentence:\n"
    "     * English: \"The provided medical evidence does not contain sufficient information to answer this question.\"\n"
    "     * Roman Urdu: \"Faraham kardah tibbi saboot is sawal ka jawab denay kay liye kafi maloomat nahi rakhta.\"\n"
    "6. Tone: Keep your tone neutral, clear, and informative."
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


def enforce_canonical_abstention(answer: str) -> str:
    """
    Enforces mutually exclusive output contract when the model signals abstention.

    If the model emits the canonical abstention sentence but prefixes or suffixes it
    with conversational preambles, introductory phrases, or extra commentary, this
    safeguard ensures the final output is strictly the exact canonical abstention sentence.

    Activation:
        Activates only when a canonical abstention sentence is detected in the response
        (matched case-insensitively to handle LLM-introduced capitalization shifts).
    Safety:
        Never removes an abstention sentence to certify claims; instead, it discards any
        extraneous text and strictly fails safe to pure canonical abstention.
    """
    if ROMAN_URDU_ABSTENTION.lower() in answer.lower():
        return ROMAN_URDU_ABSTENTION
    if ENGLISH_ABSTENTION.lower() in answer.lower():
        return ENGLISH_ABSTENTION
    return answer


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
    is_urdu = is_roman_urdu(query)

    if is_urdu:
        user_prompt = (
            f"Context Evidence from MedQuAD:\n"
            f"{evidence_text}\n\n"
            f"User Question: {query}\n\n"
            f"Instructions:\n"
            f"1. Language: Roman Urdu. The user question is in Roman Urdu. "
            f"Answer ONLY in clear, natural, grammatically correct Roman Urdu using Latin script.\n"
            f"2. Translation & Medical Accuracy:\n"
            f"   - Translate 'wheezing' clearly as 'saans lete waqt seeti jaisi awaaz aana'. Do not use awkward words like 'saise'.\n"
            f"   - 'Chest tightness' relates to the chest and must NEVER be translated as abdominal pain ('pait ka dard'); "
            f"use natural terms like 'seene mein jakran' or 'seene mein tangi'.\n"
            f"   - When explaining that having symptoms does not always mean someone has asthma, "
            f"use natural Roman Urdu (e.g., 'Zaroori nahi ke har mareez mein yeh tamam alamat hon, aur in alamat ka hona laazmi tor par dama ki tasdeeq nahi karta').\n"
            f"   - Recognized Roman Urdu terms (such as 'dama' for asthma, 'pait dard' for abdominal pain, 'khaansi' for cough) "
            f"correspond to the matching English evidence topics.\n"
            f"   - If the evidence covers a general topic (like anatomy) but does NOT explicitly answer the specific question asked (such as causes), it is insufficient.\n"
            f"3. Mutually Exclusive Output Modes:\n"
            f"   - If the evidence is insufficient or asks for undocumented local/village statistics, "
            f"output ONLY the exact canonical abstention sentence below and NOTHING else (no introductory phrase, preamble, explanation, or commentary):\n"
            f"     \"{ROMAN_URDU_ABSTENTION}\"\n"
            f"   - If and ONLY if the evidence explicitly answers the question, provide a factual grounded answer in Roman Urdu. "
            f"Never append an abstention sentence to an answer."
        )
    else:
        user_prompt = (
            f"Context Evidence from MedQuAD:\n"
            f"{evidence_text}\n\n"
            f"User Question: {query}\n\n"
            f"Instructions:\n"
            f"1. Language: English. The user question is in English. Answer strictly in English. Do NOT answer in Roman Urdu or any other language.\n"
            f"2. Evidence Grounding: Answer ONLY using facts explicitly stated in the context evidence above. "
            f"If the evidence covers a general topic (like anatomy) but does NOT explicitly answer the specific question asked (such as causes), it is insufficient.\n"
            f"3. Mutually Exclusive Output Modes:\n"
            f"   - If the evidence is insufficient or asks for undocumented local/village statistics, "
            f"output ONLY the exact canonical abstention sentence below and NOTHING else (no introductory phrase, preamble, explanation, or commentary):\n"
            f"     \"{ENGLISH_ABSTENTION}\"\n"
            f"   - If and ONLY if the evidence explicitly answers the question, provide a factual grounded answer in English. "
            f"Never append an abstention sentence to an answer."
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

    clean_content = content.strip()
    return enforce_canonical_abstention(clean_content)


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


def retrieve_evidence(
    query: str,
    retriever_type: str = DEFAULT_RETRIEVER,
    records_file: str = DEFAULT_RECORDS_FILE,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    rrf_k: int = DEFAULT_RRF_K,
) -> List[Tuple[int, float, Dict[str, Any]]]:
    """
    Retrieves evidence records from MedQuAD using the specified retrieval modality.

    Args:
        query: User search query.
        retriever_type: Retrieval modality ('hybrid', 'bm25', or 'semantic').
        records_file: Path to records.jsonl.
        top_k: Number of final records to return.
        candidate_pool: Candidate pool depth for hybrid search.
        rrf_k: RRF smoothing constant.

    Returns:
        List of (rank, score, record) tuples.
    """
    retrieval_query = normalize_medical_synonyms(query)

    if retriever_type == "bm25":
        records = load_records(records_file)
        bm25, _ = build_bm25_index(records)
        return search(retrieval_query, records, bm25, top_k=top_k)

    if retriever_type == "semantic":
        semantic_retriever = SemanticRetriever(records_file=records_file)
        return semantic_retriever.search(retrieval_query, top_k=top_k)

    if retriever_type == "hybrid":
        hybrid_retriever = HybridRetriever(
            records_file=records_file,
            candidate_pool=candidate_pool,
            rrf_k=rrf_k,
        )
        return hybrid_retriever.search(
            retrieval_query,
            top_k=top_k,
            candidate_pool=candidate_pool,
            rrf_k=rrf_k,
        )

    raise ValueError(f"Unknown retriever type: '{retriever_type}'. Expected 'hybrid', 'bm25', or 'semantic'.")


def main():
    parser = argparse.ArgumentParser(
        description="Bilingual Medical Assistant RAG Chat (English and Roman Urdu)"
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Medical question in English or Roman Urdu (e.g. 'What is asthma?' or 'Asthma ki kya alamat hoti hain?')"
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["hybrid", "bm25", "semantic"],
        default=DEFAULT_RETRIEVER,
        help=f"Retriever choice: hybrid, bm25, or semantic (default: {DEFAULT_RETRIEVER})"
    )
    parser.add_argument(
        "--candidate_pool",
        type=int,
        default=DEFAULT_CANDIDATE_POOL,
        help=f"Candidate pool depth for hybrid search (default: {DEFAULT_CANDIDATE_POOL})"
    )
    parser.add_argument(
        "--rrf_k",
        type=int,
        default=DEFAULT_RRF_K,
        help=f"RRF smoothing constant k for hybrid search (default: {DEFAULT_RRF_K})"
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

    # 1. Validate argument bounds
    if args.top_k <= 0:
        print("Error: --top_k must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    if args.candidate_pool <= 0:
        print("Error: --candidate_pool must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    if args.rrf_k <= 0:
        print("Error: --rrf_k must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    # 2. Validate non-empty question
    raw_query = " ".join(args.question).strip()
    if not raw_query:
        print("Error: No medical question provided.", file=sys.stderr)
        print("Usage: python chat.py \"<your medical question>\"", file=sys.stderr)
        print("Example: python chat.py \"What is asthma?\"", file=sys.stderr)
        sys.exit(1)

    # 3. Retrieve top matching records
    results = retrieve_evidence(
        query=raw_query,
        retriever_type=args.retriever,
        records_file=args.records_file,
        top_k=args.top_k,
        candidate_pool=args.candidate_pool,
        rrf_k=args.rrf_k,
    )

    if not results:
        abstention_msg = (
            ROMAN_URDU_ABSTENTION
            if is_roman_urdu(raw_query)
            else ENGLISH_ABSTENTION
        )
        display_rag_output(
            raw_query,
            abstention_msg,
            []
        )
        return

    # 4. Format evidence for the prompt
    evidence_text = format_evidence_block(results)

    # 5. Call local llama.cpp server
    print(f"\n[1/2] Retrieved {len(results)} MedQuAD evidence records.")
    print(f"[2/2] Consulting '{args.model}' (CPU inference may take 30-90s)...")
    answer = call_llm(
        query=raw_query,
        evidence_text=evidence_text,
        api_url=args.api_url,
        model=args.model,
        timeout=args.timeout,
    )

    # 6. Display final answer and verified source links
    display_rag_output(raw_query, answer, results)


if __name__ == "__main__":
    main()
