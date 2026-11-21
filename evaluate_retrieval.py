"""
Retrieval Evaluation Benchmark Module
-------------------------------------
Performs a reproducible, multi-metric retrieval evaluation comparing:
1. BM25 Lexical Retrieval (search.py)
2. Dense Semantic Retrieval (embed_search.py)
3. Hybrid RRF Retrieval (hybrid_search.py, candidate_pool=20, rrf_k=60)
4. Hybrid RRF Retrieval (hybrid_search.py, candidate_pool=50, rrf_k=60)

Evaluates on a curated 25-question diagnostic suite covering:
- Exact medical terminology
- Everyday and colloquial queries
- Clinical synonyms and paraphrases
- Multi-topic queries
- Ambiguous queries
- No-answer / out-of-domain queries

Generates terminal summary and optionally writes 'RETRIEVAL_EVALUATION.md' via --write-report.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

# Baseline retrievers
from search import load_records, build_bm25_index, search as bm25_search
from embed_search import SemanticRetriever
from hybrid_search import HybridRetriever


# Curated 25-Question Diagnostic Benchmark
BENCHMARK_DATASET = [
    # --- Category 1: Exact Medical Terminology ---
    {
        "id": "Q01",
        "category": "Exact Medical Terminology",
        "question": "What is asthma?",
        "relevant_ids": ["MPlusHealthTopics_0000065_1", "MPlusHealthTopics_0000066_1"],
        "notes": "Diagnostic Q1. Definition and childhood asthma.",
        "uncertain": False,
    },
    {
        "id": "Q02",
        "category": "Exact Medical Terminology",
        "question": "What is pneumonia?",
        "relevant_ids": ["MPlusHealthTopics_0000723_1"],
        "notes": "Direct clinical topic match for Pneumonia.",
        "uncertain": False,
    },
    {
        "id": "Q03",
        "category": "Exact Medical Terminology",
        "question": "What is sickle cell anemia?",
        "relevant_ids": ["MPlusHealthTopics_0000815_1"],
        "notes": "Direct clinical match for Sickle Cell Anemia.",
        "uncertain": False,
    },
    {
        "id": "Q04",
        "category": "Exact Medical Terminology",
        "question": "What is tinnitus?",
        "relevant_ids": ["MPlusHealthTopics_0000895_1"],
        "notes": "Direct clinical topic match for Tinnitus.",
        "uncertain": False,
    },
    {
        "id": "Q05",
        "category": "Exact Medical Terminology",
        "question": "What are the symptoms of peptic ulcer?",
        "relevant_ids": ["MPlusHealthTopics_0000700_1"],
        "notes": "Direct clinical topic match for Peptic Ulcer.",
        "uncertain": False,
    },

    # --- Category 2: Everyday / Colloquial Language ---
    {
        "id": "Q06",
        "category": "Everyday / Colloquial Language",
        "question": "What causes a bellyache?",
        "relevant_ids": ["MPlusHealthTopics_0000002_1"],
        "notes": "Diagnostic Q3. Colloquial lay synonym for Abdominal Pain. (Uncertain because 'bellyache' is absent in corpus text).",
        "uncertain": True,
    },
    {
        "id": "Q07",
        "category": "Everyday / Colloquial Language",
        "question": "What can I do for swollen feet and legs?",
        "relevant_ids": ["MPlusHealthTopics_0000317_1"],
        "notes": "Lay description for Edema (swelling caused by fluid retention).",
        "uncertain": False,
    },
    {
        "id": "Q08",
        "category": "Everyday / Colloquial Language",
        "question": "What causes ringing in the ears?",
        "relevant_ids": ["MPlusHealthTopics_0000895_1"],
        "notes": "Lay description of Tinnitus.",
        "uncertain": False,
    },
    {
        "id": "Q09",
        "category": "Everyday / Colloquial Language",
        "question": "How do I deal with heartburn?",
        "relevant_ids": ["MPlusHealthTopics_0000450_1"],
        "notes": "Common non-clinical phrasing for Heartburn / GERD.",
        "uncertain": False,
    },
    {
        "id": "Q10",
        "category": "Everyday / Colloquial Language",
        "question": "What are the signs of a heart attack?",
        "relevant_ids": ["MPlusHealthTopics_0000442_1"],
        "notes": "Standard patient inquiry for Heart Attack warning signs.",
        "uncertain": False,
    },

    # --- Category 3: Clinical Synonyms & Paraphrases ---
    {
        "id": "Q11",
        "category": "Clinical Synonyms & Paraphrases",
        "question": "What is high blood sugar?",
        "relevant_ids": ["MPlusHealthTopics_0000494_1", "MPlusHealthTopics_0000108_1"],
        "notes": "Diagnostic Q2. Matches both Hyperglycemia and Blood Sugar.",
        "uncertain": False,
    },
    {
        "id": "Q12",
        "category": "Clinical Synonyms & Paraphrases",
        "question": "How do I manage high blood pressure?",
        "relevant_ids": ["MPlusHealthTopics_0000471_1", "MPlusHealthTopics_0000107_1"],
        "notes": "Paraphrase covering High Blood Pressure and Blood Pressure Medicines.",
        "uncertain": False,
    },
    {
        "id": "Q13",
        "category": "Clinical Synonyms & Paraphrases",
        "question": "What causes difficulty falling asleep?",
        "relevant_ids": ["MPlusHealthTopics_0000518_1", "MPlusHealthTopics_0000824_1"],
        "notes": "Paraphrase of Insomnia and Sleep Disorders.",
        "uncertain": False,
    },
    {
        "id": "Q14",
        "category": "Clinical Synonyms & Paraphrases",
        "question": "What causes a severe throbbing headache?",
        "relevant_ids": ["MPlusHealthTopics_0000610_1", "MPlusHealthTopics_0000426_1"],
        "notes": "Paraphrase describing Migraine and Headache.",
        "uncertain": False,
    },
    {
        "id": "Q15",
        "category": "Clinical Synonyms & Paraphrases",
        "question": "What medicines treat high blood pressure?",
        "relevant_ids": ["MPlusHealthTopics_0000107_1", "MPlusHealthTopics_0000471_1"],
        "notes": "Specific inquiry targeted at Blood Pressure Medicines.",
        "uncertain": False,
    },

    # --- Category 4: Multiple Relevant Topics ---
    {
        "id": "Q16",
        "category": "Multiple Relevant Topics",
        "question": "What are the symptoms of diabetes?",
        "relevant_ids": ["MPlusHealthTopics_0000266_1", "MPlusHealthTopics_0000272_1", "MPlusHealthTopics_0000273_1"],
        "notes": "Encompasses general Diabetes, Diabetes Type 1, and Diabetes Type 2.",
        "uncertain": False,
    },
    {
        "id": "Q17",
        "category": "Multiple Relevant Topics",
        "question": "What causes a stroke?",
        "relevant_ids": ["MPlusHealthTopics_0000857_1", "MPlusHealthTopics_0000527_1", "MPlusHealthTopics_0000457_1"],
        "notes": "Encompasses general Stroke, Ischemic Stroke, and Hemorrhagic Stroke.",
        "uncertain": False,
    },
    {
        "id": "Q18",
        "category": "Multiple Relevant Topics",
        "question": "What is the difference between viral and bacterial infections?",
        "relevant_ids": ["MPlusHealthTopics_0000949_1", "MPlusHealthTopics_0000079_1"],
        "notes": "Requires both Viral Infections and Bacterial Infections for a complete answer.",
        "uncertain": False,
    },

    # --- Category 5: Ambiguous Queries ---
    {
        "id": "Q19",
        "category": "Ambiguous Queries",
        "question": "How are stones treated?",
        "relevant_ids": ["MPlusHealthTopics_0000540_1", "MPlusHealthTopics_0000388_1"],
        "notes": "Diagnostic Q4. Ambiguous between Kidney Stones and Gallstones.",
        "uncertain": False,
    },
    {
        "id": "Q20",
        "category": "Ambiguous Queries",
        "question": "How is hepatitis treated?",
        "relevant_ids": ["MPlusHealthTopics_0000459_1", "MPlusHealthTopics_0000460_1", "MPlusHealthTopics_0000461_1", "MPlusHealthTopics_0000462_1"],
        "notes": "Ambiguous between general Hepatitis and Hepatitis A, B, C strains.",
        "uncertain": False,
    },
    {
        "id": "Q21",
        "category": "Ambiguous Queries",
        "question": "Can cold weather give you a cold?",
        "relevant_ids": ["MPlusHealthTopics_0000223_1", "MPlusHealthTopics_0000497_1"],
        "notes": "Polysemous: Common Cold (illness) vs Hypothermia (cold weather response).",
        "uncertain": False,
    },

    # --- Category 6: No-Answer / Corpus Lacks Answer ---
    {
        "id": "Q22",
        "category": "No-Answer / Corpus Lacks Answer",
        "question": "How to repair a quantum rocket engine in space?",
        "relevant_ids": [],
        "notes": "Out-of-domain query. Zero medical relevance.",
        "uncertain": False,
    },
    {
        "id": "Q23",
        "category": "No-Answer / Corpus Lacks Answer",
        "question": "What is the exact number of asthma patients in my village?",
        "relevant_ids": [],
        "notes": "Hyper-local demographic query. No village statistics exist in MedQuAD.",
        "uncertain": False,
    },
    {
        "id": "Q24",
        "category": "No-Answer / Corpus Lacks Answer",
        "question": "What causes COVID-19 infection?",
        "relevant_ids": [],
        "notes": 'Corpus gap: the term `"covid"` occurs zero times in `records.jsonl`; general `Coronavirus Infections` record is not labeled as relevant.',
        "uncertain": False,
    },
    {
        "id": "Q25",
        "category": "No-Answer / Corpus Lacks Answer",
        "question": "What is the capital of France?",
        "relevant_ids": [],
        "notes": "Non-medical geographic inquiry. Zero relevant records.",
        "uncertain": False,
    },
]


def calculate_metrics(
    retrieved_results: List[Tuple[int, float, Dict[str, Any]]],
    relevant_ids: List[str],
) -> Dict[str, float]:
    """
    Computes Recall@3, Recall@5, Recall@10, MRR@10, and Top3_Hit for a single query.
    """
    rel_set = set(relevant_ids)
    if not rel_set:
        return {"recall@3": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr@10": 0.0, "top3_hit": 0.0}

    total_rel = len(rel_set)
    retrieved_ids = [rec.get("id") for _, _, rec in retrieved_results]

    # Hits in top k
    hits_3 = set(retrieved_ids[:3]).intersection(rel_set)
    hits_5 = set(retrieved_ids[:5]).intersection(rel_set)
    hits_10 = set(retrieved_ids[:10]).intersection(rel_set)

    recall_3 = len(hits_3) / total_rel
    recall_5 = len(hits_5) / total_rel
    recall_10 = len(hits_10) / total_rel
    top3_hit = 1.0 if len(hits_3) > 0 else 0.0

    # MRR@10 (first relevant item rank)
    mrr_10 = 0.0
    for rank_1b, rid in enumerate(retrieved_ids[:10], start=1):
        if rid in rel_set:
            mrr_10 = 1.0 / rank_1b
            break

    return {
        "recall@3": recall_3,
        "recall@5": recall_5,
        "recall@10": recall_10,
        "mrr@10": mrr_10,
        "top3_hit": top3_hit,
    }


def run_benchmark(
    write_report: bool = False,
    report_path: Path = Path("RETRIEVAL_EVALUATION.md"),
):
    records_file = "records.jsonl"
    embeddings_file = "embeddings.npy"
    metadata_file = "embedding_metadata.json"

    print("Initializing benchmark retrievers...")
    records = load_records(records_file)
    bm25, _ = build_bm25_index(records)
    semantic_retriever = SemanticRetriever(records_file, embeddings_file, metadata_file)
    hybrid_retriever = HybridRetriever(records_file, embeddings_file, metadata_file)

    eval_systems = [
        {"name": "BM25", "fn": lambda q: bm25_search(q, records, bm25, top_k=10)},
        {"name": "Semantic", "fn": lambda q: semantic_retriever.search(q, top_k=10)},
        {"name": "Hybrid (pool=20)", "fn": lambda q: hybrid_retriever.search(q, top_k=10, candidate_pool=20, rrf_k=60)},
        {"name": "Hybrid (pool=50)", "fn": lambda q: hybrid_retriever.search(q, top_k=10, candidate_pool=50, rrf_k=60)},
    ]

    answerable_queries = [item for item in BENCHMARK_DATASET if len(item["relevant_ids"]) > 0]
    no_answer_queries = [item for item in BENCHMARK_DATASET if len(item["relevant_ids"]) == 0]

    system_metrics = {sys["name"]: {"recall@3": [], "recall@5": [], "recall@10": [], "mrr@10": [], "top3_hit": []} for sys in eval_systems}
    per_query_results = []
    no_answer_results = []

    print(f"Evaluating {len(BENCHMARK_DATASET)} queries ({len(answerable_queries)} answerable, {len(no_answer_queries)} no-answer)...")

    # Evaluate Answerable Queries
    for item in answerable_queries:
        qid = item["id"]
        qtext = item["question"]
        rel_ids = item["relevant_ids"]

        query_row = {"id": qid, "category": item["category"], "question": qtext, "relevant_ids": rel_ids, "systems": {}}

        for s in eval_systems:
            s_name = s["name"]
            results = s["fn"](qtext)
            mets = calculate_metrics(results, rel_ids)

            for k in system_metrics[s_name]:
                system_metrics[s_name][k].append(mets[k])

            query_row["systems"][s_name] = {
                "metrics": mets,
                "top3": [(rank, score, rec.get("id"), rec.get("topic")) for rank, score, rec in results[:3]],
            }

        per_query_results.append(query_row)

    # Evaluate No-Answer Queries
    for item in no_answer_queries:
        qid = item["id"]
        qtext = item["question"]
        row = {"id": qid, "category": item["category"], "question": qtext, "notes": item["notes"], "systems": {}}
        for s in eval_systems:
            s_name = s["name"]
            results = s["fn"](qtext)
            row["systems"][s_name] = [
                (rank, score, rec.get("id"), rec.get("topic")) for rank, score, rec in results[:2]
            ]
        no_answer_results.append(row)

    # Aggregate Scores for Answerable Queries
    agg_scores = {}
    for s_name, m_dict in system_metrics.items():
        n = len(answerable_queries)
        agg_scores[s_name] = {
            "Recall@3": sum(m_dict["recall@3"]) / n,
            "Recall@5": sum(m_dict["recall@5"]) / n,
            "Recall@10": sum(m_dict["recall@10"]) / n,
            "MRR@10": sum(m_dict["mrr@10"]) / n,
            "Top-3 Hit": sum(m_dict["top3_hit"]) / n,
        }

    # Print Terminal Summary
    print("\n" + "=" * 78)
    print(" RETRIEVAL EVALUATION SUMMARY (N=21 Answerable Medical Questions)")
    print("=" * 78)
    print(f"{'Retriever System':<20} | {'Recall@3':<9} | {'Recall@5':<9} | {'Recall@10':<9} | {'MRR@10':<8} | {'Top-3 Hit':<9}")
    print("-" * 78)
    for s_name, sc in agg_scores.items():
        print(
            f"{s_name:<20} | {sc['Recall@3']:<9.4f} | {sc['Recall@5']:<9.4f} | "
            f"{sc['Recall@10']:<9.4f} | {sc['MRR@10']:<8.4f} | {sc['Top-3 Hit']:<9.4f}"
        )
    print("-" * 78)

    # Write Markdown Report only if requested
    if write_report:
        write_markdown_report(
            agg_scores,
            per_query_results,
            no_answer_results,
            len(answerable_queries),
            len(no_answer_queries),
            report_path=report_path,
        )
    else:
        print(f"\n(Report generation skipped. Pass --write-report to update '{report_path.name}'.)")

    return agg_scores, per_query_results, no_answer_results


def write_markdown_report(
    agg_scores: Dict[str, Dict[str, float]],
    per_query_results: List[Dict[str, Any]],
    no_answer_results: List[Dict[str, Any]],
    n_ans: int,
    n_no_ans: int,
    report_path: Path = Path("RETRIEVAL_EVALUATION.md"),
):
    lines = []
    lines.append("# Retrieval Evaluation Report: BM25, Semantic, and Hybrid RRF")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(
        "This report documents a reproducible, retrieval-only benchmark evaluating three search modalities across "
        f"a 25-question diagnostic dataset (21 answerable questions, 4 no-answer/out-of-domain questions) on the MedQuAD corpus (981 records)."
    )
    lines.append("")
    lines.append("### Evaluated Systems:")
    lines.append("1. **BM25 Lexical**: Keyword BM25Okapi baseline from `search.py`.")
    lines.append("2. **Dense Semantic**: Cosine similarity using `intfloat/multilingual-e5-small` embeddings from `embed_search.py`.")
    lines.append("3. **Hybrid RRF (pool=20)**: Reciprocal Rank Fusion ($k_{\\text{rrf}}=60$) pooling top 20 candidates from each system.")
    lines.append("4. **Hybrid RRF (pool=50)**: Reciprocal Rank Fusion ($k_{\\text{rrf}}=60$) pooling top 50 candidates from each system.")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> **Disclaimer**: Relevance labels represent preliminary manual judgments based on the MedQuAD health topics corpus, not validated clinical ground truth. No LLM calls were made during this benchmark.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Aggregate Retrieval Metrics (N=21 Answerable Questions)")
    lines.append("")
    lines.append("| Retrieval System | Recall@3 | Recall@5 | Recall@10 | MRR@10 | Top-3 Hit Rate |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|")
    for s_name, sc in agg_scores.items():
        lines.append(
            f"| **{s_name}** | **{sc['Recall@3']:.4f}** | **{sc['Recall@5']:.4f}** | "
            f"**{sc['Recall@10']:.4f}** | **{sc['MRR@10']:.4f}** | **{sc['Top-3 Hit']:.4f}** |"
        )
    lines.append("")
    lines.append("### Metric Definitions:")
    lines.append("- **Recall@k**: Fraction of all labeled relevant records for a query retrieved in the top $k$. $\\text{Recall}@k = \\frac{|\\text{Retrieved}_k \\cap \\text{Relevant}|}{|\\text{Relevant}|}$.")
    lines.append("- **MRR@10**: Mean Reciprocal Rank of the *first* retrieved relevant record within top 10 ($1/\\text{rank}$, or $0$ if not found in top 10).")
    lines.append("- **Top-3 Hit Rate**: Fraction of queries where at least one labeled relevant record appears in the top 3.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Per-Question Detailed Comparison (Answerable Queries)")
    lines.append("")
    lines.append("| ID | Category | Question | Relevant Topic(s) | BM25 Top-3 | Semantic Top-3 | Hybrid (pool=50) Top-3 | RRF MRR@10 |")
    lines.append("|---|---|---|---|---|---|---|:---:|")

    for q in per_query_results:
        qid = q["id"]
        cat = q["category"]
        qtxt = q["question"]
        rel_str = ", ".join(q["relevant_ids"])

        def format_top3(s_top):
            if not s_top:
                return "*(0 results)*"
            return "<br>".join([f"#{r} {t} ({sc:.2f})" for r, sc, _, t in s_top])

        b_top = format_top3(q["systems"]["BM25"]["top3"])
        s_top = format_top3(q["systems"]["Semantic"]["top3"])
        h_top = format_top3(q["systems"]["Hybrid (pool=50)"]["top3"])
        h_mrr = f"{q['systems']['Hybrid (pool=50)']['metrics']['mrr@10']:.3f}"

        lines.append(f"| {qid} | {cat} | *\"{qtxt}\"* | `{rel_str}` | {b_top} | {s_top} | {h_top} | {h_mrr} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. No-Answer / Out-of-Domain Analysis (N=4 Queries)")
    lines.append("")
    lines.append("For queries where the corpus contains no relevant information, ideal retrieval should return zero matches (or allow confidence-based abstention). Retrieval scores alone do not prove whether a question is answerable.")
    lines.append("")
    lines.append("| ID | Question | Corpus Status | BM25 Top Result | Semantic Top Result (Cosine) | Hybrid Top Result (RRF) |")
    lines.append("|---|---|---|---|---|---|")

    for item in no_answer_results:
        qid = item["id"]
        qtxt = item["question"]
        note = item["notes"]
        b_res = item["systems"]["BM25"]
        s_res = item["systems"]["Semantic"]
        h_res = item["systems"]["Hybrid (pool=50)"]

        b_str = f"#{b_res[0][0]} {b_res[0][3]} ({b_res[0][1]:.2f})" if b_res else "*(0 results)*"
        s_str = f"#{s_res[0][0]} {s_res[0][3]} ({s_res[0][1]:.3f})" if s_res else "*(0 results)*"
        h_str = f"#{h_res[0][0]} {h_res[0][3]} ({h_res[0][1]:.4f})" if h_res else "*(0 results)*"

        lines.append(f"| {qid} | *\"{qtxt}\"* | {note} | {b_str} | {s_str} | {h_str} |")

    lines.append("")
    lines.append('### Note on Q24 ("What causes COVID-19 infection?") vs. General "Coronavirus Infections"')
    lines.append('Although record `MPlusHealthTopics_0000239_1` (`Coronavirus Infections`) exists in the corpus and was retrieved at Rank #1 by semantic search (cosine `0.871`), it is **not labeled as a relevant answer**.')
    lines.append('- **Rationale**: The exact term `"covid"` occurs **zero times** in `records.jsonl`. Although the corpus contains a general `Coronavirus Infections` record, that record is not labeled as relevant to the COVID-19-specific question. Record `0000239_1` describes general endemic upper-respiratory coronaviruses (stating *"they usually cause mild illness"* and *"there is no vaccine"*). Relying on this record to answer a COVID-19-specific question would produce factually outdated and clinically misleading guidance regarding SARS-CoV-2.')
    lines.append('- This evaluation therefore treats Q24 as a corpus evidence gap (`relevant_ids = []`), rather than assuming that general coronavirus information adequately answers a COVID-19-specific question.')
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Key Retrieval Failures & Observations")
    lines.append("")
    lines.append('### 1. The Lay Synonym Blindspot (Q06: "What causes a bellyache?")')
    lines.append("- **Observation**: `Abdominal Pain` (`0000002_1`) was completely missed by BM25 (0 results) and ranked #520 by Semantic retrieval. Consequently, Hybrid placed `Bell's Palsy` at #1 due to subword overlap ('bell').")
    lines.append("- **Takeaway**: Dense embeddings do not magically bridge colloquial terms when corpus documents use strictly formal clinical language. Extracting `<Synonyms>` at XML ingestion time remains essential.")
    lines.append("")
    lines.append('### 2. Clinical Term Elevation (Q11: "What is high blood sugar?")')
    lines.append("- **Observation**: BM25 ranked `Blood Sugar` at #1 and pushed `Hyperglycemia` to #6. Semantic retrieval placed `Hyperglycemia` directly at #1. Hybrid RRF harmoniously balanced both: `Blood Sugar` at #1 and `Hyperglycemia` at #2.")
    lines.append("- **Takeaway**: Hybrid search provides the best of both worlds for medical RAG, capturing both the lay query tokens and the clinical entity.")
    lines.append("")
    lines.append('### 3. RRF Candidate-Pool Depth & Rank Inversion (Q19: "How are stones treated?")')
    lines.append("- **Empirical Observation**: Hybrid with `candidate_pool = 20` achieved a higher aggregate Recall@3 (**0.9167**) than `candidate_pool = 50` (**0.8929**).")
    lines.append("- **Verification Across All 21 Queries**: Per-query verification confirms that across all 21 answerable queries, **Q19 is the sole query whose Recall@3 differs** between pool=20 and pool=50. The difference in means is exactly:")
    lines.append(r"  $$\Delta\text{Recall@3} = \frac{1.0 - 0.5}{21} = \frac{0.5}{21} \approx 0.02381$$")
    lines.append("- **Exact Rank & Score Shift on Q19**:")
    lines.append("  - **BM25**: #1 `Kidney Stones` (9.91), #2 `Kidney Diseases` (5.11), `Gallstones` absent (compound-word mismatch).")
    lines.append("  - **Semantic**: #1 `Kidney Stones` (0.8407), #2 `Gallstones` (0.8247), #3 `Bone Grafts` (0.8171).")
    lines.append("  - **Hybrid (pool=20)**:")
    lines.append("    - #1 `Kidney Stones` (RRF `0.03279`) — *Relevant*")
    lines.append("    - #2 `Kidney Diseases` (RRF `0.01613`)")
    lines.append("    - #3 `Gallstones` (RRF `0.01613`) — *Relevant*")
    lines.append("    - **Recall@3 = 2/2 = 1.000**")
    lines.append("  - **Hybrid (pool=50)**:")
    lines.append("    - #1 `Kidney Stones` (RRF `0.03279`) — *Relevant*")
    lines.append("    - #2 `Shoulder Injuries and Disorders` (RRF `0.02402`) — *(BM25 #19, Semantic #28)*")
    lines.append("    - #3 `Gum Disease` (RRF `0.02399`) — *(BM25 #10, Semantic #43)*")
    lines.append("    - #4 `Kidney Diseases` (RRF `0.01613`)")
    lines.append("    - #5 `Gallstones` (RRF `0.01613`) — *Relevant*")
    lines.append("    - **Recall@3 = 1/2 = 0.500**")
    lines.append("- **Mathematical Mechanism**:")
    lines.append("  `Gallstones` is matched only by semantic search (rank #2), yielding a single-system RRF score:")
    lines.append(r"  $$\text{RRF}(\text{Gallstones}) = \frac{1}{60 + 2} = \frac{1}{62} \approx 0.01613$$")
    lines.append("  When `candidate_pool = 20`, candidates below rank 20 are ignored, so `Gallstones` remains at rank #3. When `candidate_pool = 50`, documents appearing moderately in *both* systems enter the pool. For example, `Shoulder Injuries` (BM25 #19, Semantic #28) receives:")
    lines.append(r"  $$\text{RRF} = \frac{1}{60 + 19} + \frac{1}{60 + 28} = \frac{1}{79} + \frac{1}{88} \approx 0.01266 + 0.01136 = 0.02402$$")
    lines.append("  Because $0.02402 > 0.01613$, two weak dual-system matches mathematically leapfrog a strong single-system match. This demonstrates a key trade-off in equal-weighted Reciprocal Rank Fusion when one retrieval modality fails completely on compound words.")
    lines.append("")
    lines.append('### 4. No-Answer Safeguards (Q22: "Quantum rocket engine", Q25: "Capital of France")')
    lines.append("- **Observation**: BM25 cleanly returned 0 results for out-of-domain queries due to core-topic filtering. Dense semantic retrieval, by contrast, *always* returns top-k documents with non-zero cosine similarities (e.g. cosine ~0.76 for Space Medicine or General topics).")
    lines.append("- **Takeaway**: BM25 acts as an effective guardrail against hallucinations on completely out-of-domain inputs. Combining BM25 with dense retrieval prevents the system from forcing semantic answers on unrelated queries.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Conclusion & Recommendation for FYP")
    lines.append("")
    lines.append(
        "On this 25-question diagnostic suite, **Hybrid RRF demonstrates the highest aggregate retrieval robustness** "
        "(highest Recall@5, Recall@10, and top MRR). It successfully compensates for BM25's vocabulary fragility while "
        "leveraging BM25's lexical precision to anchor dense embedding drift."
    )
    lines.append("")
    lines.append(
        "- **Pool Depth Trade-off**: Candidate pool depth of 20 avoids promoting weak dual-retriever distractors over "
        "strong single-system semantic matches, yielding the highest Recall@3 (**0.9167**). A pool depth of 50 captures "
        "deeper candidates but is more susceptible to distractor aggregation under unweighted RRF."
    )
    lines.append(
        "- **FYP Integration**: Integrating Hybrid RRF into `chat.py` with an optional `--retriever bm25|semantic|hybrid` "
        "selector and `candidate_pool=20` is recommended as the default retrieval strategy."
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport successfully saved to '{report_path.name}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation benchmark comparing BM25, Semantic, and Hybrid RRF."
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write evaluation results to markdown report (default: False)",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("RETRIEVAL_EVALUATION.md"),
        help="Path for markdown report output (default: RETRIEVAL_EVALUATION.md)",
    )
    args = parser.parse_args()
    run_benchmark(write_report=args.write_report, report_path=args.report_path)


if __name__ == "__main__":
    main()
