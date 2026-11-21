"""
Hybrid Retrieval Module (BM25 + Semantic via Reciprocal Rank Fusion)
---------------------------------------------------------------------
This module combines lexical BM25 retrieval (search.py) with dense semantic
retrieval (embed_search.py) using Reciprocal Rank Fusion (RRF).

Formula:
    RRF_score(d) = sum_{m in {bm25, semantic}} (1 / (rrf_k + rank_m(d)))

RRF provides a stable, rank-based combination of sparse lexical matching and
dense semantic similarity without requiring fragile score calibration or normalization.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Baseline modules
from search import load_records, build_bm25_index, search as bm25_search
from embed_search import SemanticRetriever, DEFAULT_MODEL_NAME, DEFAULT_RECORDS_FILE, DEFAULT_EMBEDDINGS_FILE, DEFAULT_METADATA_FILE


DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_POOL = 50


class HybridRetriever:
    """
    Coordinates BM25 and Semantic retrieval and fuses candidate rankings via RRF.
    """

    def __init__(
        self,
        records_file: str = DEFAULT_RECORDS_FILE,
        embeddings_file: str = DEFAULT_EMBEDDINGS_FILE,
        metadata_file: str = DEFAULT_METADATA_FILE,
        model_name: str = DEFAULT_MODEL_NAME,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
        device: str = "cpu",
    ):
        self.records_path = Path(records_file)
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool

        # 1. Load records and initialize BM25
        self.records = load_records(str(self.records_path))
        self.bm25, _ = build_bm25_index(self.records)

        # 2. Initialize Semantic Retriever
        self.semantic_retriever = SemanticRetriever(
            records_file=records_file,
            embeddings_file=embeddings_file,
            metadata_file=metadata_file,
            model_name=model_name,
            device=device,
        )

        # 3. Validate alignment between BM25 and Semantic records
        self._validate_corpus_alignment()

    def _validate_corpus_alignment(self) -> None:
        """
        Ensures that both retrievers operate on the exact same corpus in the same order.
        """
        sem_records = self.semantic_retriever.records
        if len(self.records) != len(sem_records):
            raise ValueError(
                f"Corpus size mismatch: BM25 has {len(self.records)} records, "
                f"Semantic retriever has {len(sem_records)} records."
            )

        for idx, (r_bm25, r_sem) in enumerate(zip(self.records, sem_records)):
            id_bm25 = r_bm25.get("id")
            id_sem = r_sem.get("id")
            if id_bm25 != id_sem:
                raise ValueError(
                    f"Corpus ordering mismatch at index {idx}: "
                    f"BM25 ID '{id_bm25}' != Semantic ID '{id_sem}'."
                )

    def search_with_diagnostics(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: Optional[int] = None,
        rrf_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes BM25, Semantic, and Hybrid retrieval, returning all ranked lists
        and per-document fusion diagnostics.

        Args:
            query: User's question or search query.
            top_k: Number of final fused results to return.
            candidate_pool: Number of top candidates to pull from each individual retriever.
            rrf_k: Smoothing constant in RRF formula (default: 60).

        Returns:
            Dict containing:
                - 'bm25_results': List of (rank, score, record)
                - 'semantic_results': List of (rank, score, record)
                - 'hybrid_results': List of (rank, rrf_score, record)
                - 'diagnostics': Dict mapping record_id to rank & score breakdown
        """
        pool_size = candidate_pool or self.candidate_pool
        k_const = rrf_k or self.rrf_k

        # 1. Retrieve candidate lists from both systems
        bm25_candidates = bm25_search(query, self.records, self.bm25, top_k=pool_size)
        sem_candidates = self.semantic_retriever.search(query, top_k=pool_size)

        # 2. Fuse candidate lists using Reciprocal Rank Fusion
        fused_scores: Dict[str, Dict[str, Any]] = {}

        # Process BM25 rankings (1-based rank)
        for rank_1b, score, record in bm25_candidates:
            rec_id = record.get("id")
            if not rec_id:
                continue
            if rec_id not in fused_scores:
                fused_scores[rec_id] = {
                    "record": record,
                    "bm25_rank": rank_1b,
                    "bm25_score": score,
                    "sem_rank": None,
                    "sem_score": None,
                    "rrf_score": 0.0,
                }
            else:
                fused_scores[rec_id]["bm25_rank"] = rank_1b
                fused_scores[rec_id]["bm25_score"] = score
            fused_scores[rec_id]["rrf_score"] += 1.0 / (k_const + rank_1b)

        # Process Semantic rankings (1-based rank)
        for rank_1b, score, record in sem_candidates:
            rec_id = record.get("id")
            if not rec_id:
                continue
            if rec_id not in fused_scores:
                fused_scores[rec_id] = {
                    "record": record,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "sem_rank": rank_1b,
                    "sem_score": score,
                    "rrf_score": 0.0,
                }
            else:
                fused_scores[rec_id]["sem_rank"] = rank_1b
                fused_scores[rec_id]["sem_score"] = score
            fused_scores[rec_id]["rrf_score"] += 1.0 / (k_const + rank_1b)

        # 3. Sort by RRF score descending
        sorted_candidates = sorted(
            fused_scores.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        # 4. Format hybrid results in standard (rank, score, record) tuple format
        hybrid_results: List[Tuple[int, float, Dict[str, Any]]] = []
        for rank_num, item in enumerate(sorted_candidates[:top_k], start=1):
            hybrid_results.append((rank_num, float(item["rrf_score"]), item["record"]))

        return {
            "query": query,
            "bm25_results": bm25_candidates,
            "semantic_results": sem_candidates,
            "hybrid_results": hybrid_results,
            "diagnostics": {item["record"].get("id"): item for item in sorted_candidates[:top_k]},
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: Optional[int] = None,
        rrf_k: Optional[int] = None,
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Standard search interface compatible with search.py and chat.py.
        Returns List[Tuple[rank, rrf_score, record]].
        """
        diag = self.search_with_diagnostics(
            query=query,
            top_k=top_k,
            candidate_pool=candidate_pool,
            rrf_k=rrf_k,
        )
        return diag["hybrid_results"]


def display_comparison(diag: Dict[str, Any], top_display: int = 3) -> None:
    """
    Renders BM25, Semantic, and Hybrid rankings side-by-side for comparison.
    """
    query = diag["query"]
    bm25_res = diag["bm25_results"][:top_display]
    sem_res = diag["semantic_results"][:top_display]
    hyb_res = diag["hybrid_results"][:top_display]
    diagnostics = diag["diagnostics"]

    print("=" * 70)
    print(f" RETRIEVAL COMPARISON FOR QUERY: \"{query}\"")
    print("=" * 70)

    # 1. BM25 Column
    print(f"\n[A] BM25 Lexical Ranking (top {len(bm25_res)}):")
    if not bm25_res:
        print("    (No results returned / 0 matches)")
    for rank, score, rec in bm25_res:
        print(f"    Rank #{rank} | BM25 Score: {score:7.4f} | [{rec.get('id')}] {rec.get('topic')}")

    # 2. Semantic Column
    print(f"\n[B] Dense Semantic Ranking (top {len(sem_res)}):")
    if not sem_res:
        print("    (No results returned)")
    for rank, score, rec in sem_res:
        print(f"    Rank #{rank} | Cosine Sim: {score:7.4f} | [{rec.get('id')}] {rec.get('topic')}")

    # 3. Hybrid RRF Column
    print(f"\n[C] Hybrid RRF Ranking (top {len(hyb_res)}):")
    if not hyb_res:
        print("    (No results returned)")
    for rank, rrf_score, rec in hyb_res:
        rec_id = rec.get("id")
        diag_item = diagnostics.get(rec_id, {})
        b_rank = diag_item.get("bm25_rank")
        s_rank = diag_item.get("sem_rank")
        b_str = f"#{b_rank}" if b_rank else "None"
        s_str = f"#{s_rank}" if s_rank else "None"
        print(
            f"    Rank #{rank} | RRF Score: {rrf_score:.5f} | [{rec_id}] {rec.get('topic')}\n"
            f"           |-> Source Ranks: BM25={b_str}, Semantic={s_str}"
        )
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Retrieval (BM25 + Semantic via Reciprocal Rank Fusion)"
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
        help="Number of final results to display (default: 5)"
    )
    parser.add_argument(
        "--rrf_k",
        type=int,
        default=DEFAULT_RRF_K,
        help=f"RRF smoothing constant (default: {DEFAULT_RRF_K})"
    )
    parser.add_argument(
        "--candidate_pool",
        type=int,
        default=DEFAULT_CANDIDATE_POOL,
        help=f"Candidates to retrieve from each system before fusion (default: {DEFAULT_CANDIDATE_POOL})"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show detailed comparison of BM25, Semantic, and Hybrid rankings"
    )

    args = parser.parse_args()
    raw_query = " ".join(args.query).strip()

    if not raw_query:
        print("Error: No search query provided.", file=sys.stderr)
        print("Usage: python hybrid_search.py \"<your question>\" [--compare]", file=sys.stderr)
        sys.exit(1)

    retriever = HybridRetriever(
        rrf_k=args.rrf_k,
        candidate_pool=args.candidate_pool,
    )

    if args.compare:
        diag = retriever.search_with_diagnostics(
            raw_query,
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
            rrf_k=args.rrf_k,
        )
        display_comparison(diag, top_display=args.top_k)
    else:
        results = retriever.search(raw_query, top_k=args.top_k)
        print(f"\n=======================================================")
        print(f" Hybrid Search Results for: \"{raw_query}\"")
        print(f" Total Results Returned: {len(results)}")
        print(f"=======================================================\n")
        for rank, score, rec in results:
            print(f"Rank #{rank} | RRF Score: {score:.5f}")
            print(f"Record ID:   {rec.get('id', 'N/A')}")
            print(f"Topic:       {rec.get('topic', 'N/A')}")
            print(f"Question:    {rec.get('question', 'N/A')}")
            answer = rec.get("answer", "")
            preview = answer[:250].rstrip() + "... [truncated]" if len(answer) > 250 else answer
            print(f"Answer:      {preview}")
            print("-" * 55)


if __name__ == "__main__":
    main()
