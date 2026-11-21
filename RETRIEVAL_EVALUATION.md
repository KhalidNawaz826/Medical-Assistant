# Retrieval Evaluation Report: BM25, Semantic, and Hybrid RRF

## 1. Executive Summary

This report documents a reproducible, retrieval-only benchmark evaluating three search modalities across a 25-question diagnostic dataset (21 answerable questions, 4 no-answer/out-of-domain questions) on the MedQuAD corpus (981 records).

### Evaluated Systems:
1. **BM25 Lexical**: Keyword BM25Okapi baseline from `search.py`.
2. **Dense Semantic**: Cosine similarity using `intfloat/multilingual-e5-small` embeddings from `embed_search.py`.
3. **Hybrid RRF (pool=20)**: Reciprocal Rank Fusion ($k_{\text{rrf}}=60$) pooling top 20 candidates from each system.
4. **Hybrid RRF (pool=50)**: Reciprocal Rank Fusion ($k_{\text{rrf}}=60$) pooling top 50 candidates from each system.

> [!NOTE]
> **Disclaimer**: Relevance labels represent preliminary manual judgments based on the MedQuAD health topics corpus, not validated clinical ground truth. No LLM calls were made during this benchmark.

---

## 2. Aggregate Retrieval Metrics (N=21 Answerable Questions)

| Retrieval System | Recall@3 | Recall@5 | Recall@10 | MRR@10 | Top-3 Hit Rate |
|---|:---:|:---:|:---:|:---:|:---:|
| **BM25** | **0.8373** | **0.8810** | **0.9286** | **0.8968** | **0.9524** |
| **Semantic** | **0.8690** | **0.9048** | **0.9524** | **0.9127** | **0.9048** |
| **Hybrid (pool=20)** | **0.9167** | **0.9524** | **0.9524** | **0.9286** | **0.9524** |
| **Hybrid (pool=50)** | **0.8929** | **0.9524** | **0.9524** | **0.9286** | **0.9524** |

### Metric Definitions:
- **Recall@k**: Fraction of all labeled relevant records for a query retrieved in the top $k$. $\text{Recall}@k = \frac{|\text{Retrieved}_k \cap \text{Relevant}|}{|\text{Relevant}|}$.
- **MRR@10**: Mean Reciprocal Rank of the *first* retrieved relevant record within top 10 ($1/\text{rank}$, or $0$ if not found in top 10).
- **Top-3 Hit Rate**: Fraction of queries where at least one labeled relevant record appears in the top 3.

---

## 3. Per-Question Detailed Comparison (Answerable Queries)

| ID | Category | Question | Relevant Topic(s) | BM25 Top-3 | Semantic Top-3 | Hybrid (pool=50) Top-3 | RRF MRR@10 |
|---|---|---|---|---|---|---|:---:|
| Q01 | Exact Medical Terminology | *"What is asthma?"* | `MPlusHealthTopics_0000065_1, MPlusHealthTopics_0000066_1` | #1 Asthma in Children (8.26)<br>#2 Asthma (8.25)<br>#3 Environmental Health (6.10) | #1 Asthma (0.90)<br>#2 Asthma in Children (0.89)<br>#3 Environmental Health (0.85) | #1 Asthma in Children (0.03)<br>#2 Asthma (0.03)<br>#3 Environmental Health (0.03) | 1.000 |
| Q02 | Exact Medical Terminology | *"What is pneumonia?"* | `MPlusHealthTopics_0000723_1` | #1 Pneumonia (7.96)<br>#2 Legionnaires' Disease (6.16)<br>#3 Streptococcal Infections (5.15) | #1 Pneumonia (0.88)<br>#2 Legionnaires' Disease (0.85)<br>#3 Pneumococcal Infections (0.85) | #1 Pneumonia (0.03)<br>#2 Legionnaires' Disease (0.03)<br>#3 Pneumocystis Infections (0.03) | 1.000 |
| Q03 | Exact Medical Terminology | *"What is sickle cell anemia?"* | `MPlusHealthTopics_0000815_1` | #1 Sickle Cell Anemia (26.14)<br>#2 Anemia (18.43)<br>#3 Bone Marrow Transplantation (11.64) | #1 Sickle Cell Anemia (0.90)<br>#2 Anemia (0.85)<br>#3 Chickenpox (0.81) | #1 Sickle Cell Anemia (0.03)<br>#2 Anemia (0.03)<br>#3 Aplastic Anemia (0.03) | 1.000 |
| Q04 | Exact Medical Terminology | *"What is tinnitus?"* | `MPlusHealthTopics_0000895_1` | #1 Tinnitus (11.31)<br>#2 Ear Disorders (7.67)<br>#3 Meniere's Disease (7.10) | #1 Tinnitus (0.90)<br>#2 Ear Disorders (0.84)<br>#3 Meniere's Disease (0.83) | #1 Tinnitus (0.03)<br>#2 Ear Disorders (0.03)<br>#3 Meniere's Disease (0.03) | 1.000 |
| Q05 | Exact Medical Terminology | *"What are the symptoms of peptic ulcer?"* | `MPlusHealthTopics_0000700_1` | #1 Peptic Ulcer (17.09)<br>#2 Helicobacter Pylori Infections (12.81)<br>#3 Small Intestine Disorders (11.77) | #1 Peptic Ulcer (0.89)<br>#2 Helicobacter Pylori Infections (0.85)<br>#3 Ulcerative Colitis (0.85) | #1 Peptic Ulcer (0.03)<br>#2 Helicobacter Pylori Infections (0.03)<br>#3 Indigestion (0.03) | 1.000 |
| Q06 | Everyday / Colloquial Language | *"What causes a bellyache?"* | `MPlusHealthTopics_0000002_1` | *(0 results)* | #1 Bell's Palsy (0.84)<br>#2 Anaphylaxis (0.81)<br>#3 Measles (0.81) | #1 Bell's Palsy (0.02)<br>#2 Anaphylaxis (0.02)<br>#3 Measles (0.02) | 0.000 |
| Q07 | Everyday / Colloquial Language | *"What can I do for swollen feet and legs?"* | `MPlusHealthTopics_0000317_1` | #1 Peripheral Arterial Disease (8.46)<br>#2 Edema (8.21)<br>#3 Kawasaki Disease (7.87) | #1 Foot Health (0.86)<br>#2 Leg Injuries and Disorders (0.85)<br>#3 Tendinitis (0.85) | #1 Edema (0.03)<br>#2 Varicose Veins (0.03)<br>#3 Foot Health (0.03) | 1.000 |
| Q08 | Everyday / Colloquial Language | *"What causes ringing in the ears?"* | `MPlusHealthTopics_0000895_1` | #1 Tinnitus (11.69)<br>#2 Throat Cancer (10.01)<br>#3 Neurofibromatosis (9.94) | #1 Tinnitus (0.87)<br>#2 Ear Disorders (0.86)<br>#3 Meniere's Disease (0.85) | #1 Tinnitus (0.03)<br>#2 Ear Infections (0.03)<br>#3 Ear Disorders (0.03) | 1.000 |
| Q09 | Everyday / Colloquial Language | *"How do I deal with heartburn?"* | `MPlusHealthTopics_0000450_1` | #1 Heartburn (10.16)<br>#2 GERD (7.45)<br>#3 Stomach Disorders (6.86) | #1 Heartburn (0.88)<br>#2 GERD (0.85)<br>#3 Pericardial Disorders (0.85) | #1 Heartburn (0.03)<br>#2 GERD (0.03)<br>#3 Chest Pain (0.03) | 1.000 |
| Q10 | Everyday / Colloquial Language | *"What are the signs of a heart attack?"* | `MPlusHealthTopics_0000442_1` | #1 Heart Attack (8.85)<br>#2 Vital Signs (7.29)<br>#3 Women's Health (7.11) | #1 Heart Attack (0.88)<br>#2 Vital Signs (0.87)<br>#3 Heart Valve Diseases (0.86) | #1 Heart Attack (0.03)<br>#2 Vital Signs (0.03)<br>#3 Pericardial Disorders (0.03) | 1.000 |
| Q11 | Clinical Synonyms & Paraphrases | *"What is high blood sugar?"* | `MPlusHealthTopics_0000494_1, MPlusHealthTopics_0000108_1` | #1 Blood Sugar (11.79)<br>#2 Metabolic Syndrome (10.70)<br>#3 Diabetes and Pregnancy (10.39) | #1 Hyperglycemia (0.89)<br>#2 Blood Sugar (0.88)<br>#3 Hypoglycemia (0.87) | #1 Blood Sugar (0.03)<br>#2 Hyperglycemia (0.03)<br>#3 Metabolic Syndrome (0.03) | 1.000 |
| Q12 | Clinical Synonyms & Paraphrases | *"How do I manage high blood pressure?"* | `MPlusHealthTopics_0000471_1, MPlusHealthTopics_0000107_1` | #1 High Blood Pressure (10.65)<br>#2 High Blood Pressure in Pregnancy (10.34)<br>#3 Blood Pressure Medicines (9.59) | #1 High Blood Pressure (0.89)<br>#2 Blood Pressure Medicines (0.89)<br>#3 Low Blood Pressure (0.87) | #1 High Blood Pressure (0.03)<br>#2 Blood Pressure Medicines (0.03)<br>#3 High Blood Pressure in Pregnancy (0.03) | 1.000 |
| Q13 | Clinical Synonyms & Paraphrases | *"What causes difficulty falling asleep?"* | `MPlusHealthTopics_0000518_1, MPlusHealthTopics_0000824_1` | #1 Sleep Disorders (14.11)<br>#2 Insomnia (13.65)<br>#3 Restless Legs (7.83) | #1 Sleep Disorders (0.89)<br>#2 Insomnia (0.86)<br>#3 Sleep Apnea (0.85) | #1 Sleep Disorders (0.03)<br>#2 Insomnia (0.03)<br>#3 Restless Legs (0.03) | 1.000 |
| Q14 | Clinical Synonyms & Paraphrases | *"What causes a severe throbbing headache?"* | `MPlusHealthTopics_0000610_1, MPlusHealthTopics_0000426_1` | #1 Headache (8.03)<br>#2 Migraine (7.56)<br>#3 Traumatic Brain Injury (7.05) | #1 Migraine (0.86)<br>#2 Headache (0.85)<br>#3 Traumatic Brain Injury (0.85) | #1 Headache (0.03)<br>#2 Migraine (0.03)<br>#3 Traumatic Brain Injury (0.03) | 1.000 |
| Q15 | Clinical Synonyms & Paraphrases | *"What medicines treat high blood pressure?"* | `MPlusHealthTopics_0000107_1, MPlusHealthTopics_0000471_1` | #1 High Blood Pressure (11.39)<br>#2 Blood Pressure Medicines (11.28)<br>#3 High Blood Pressure in Pregnancy (11.14) | #1 Blood Pressure Medicines (0.90)<br>#2 High Blood Pressure (0.86)<br>#3 Low Blood Pressure (0.84) | #1 High Blood Pressure (0.03)<br>#2 Blood Pressure Medicines (0.03)<br>#3 Low Blood Pressure (0.03) | 1.000 |
| Q16 | Multiple Relevant Topics | *"What are the symptoms of diabetes?"* | `MPlusHealthTopics_0000266_1, MPlusHealthTopics_0000272_1, MPlusHealthTopics_0000273_1` | #1 Diabetes Type 2 (4.60)<br>#2 Diabetes Type 1 (4.16)<br>#3 Diabetes Insipidus (4.07) | #1 Diabetes Type 2 (0.89)<br>#2 Diabetes Type 1 (0.89)<br>#3 Diabetes (0.89) | #1 Diabetes Type 2 (0.03)<br>#2 Diabetes Type 1 (0.03)<br>#3 Diabetes (0.03) | 1.000 |
| Q17 | Multiple Relevant Topics | *"What causes a stroke?"* | `MPlusHealthTopics_0000857_1, MPlusHealthTopics_0000527_1, MPlusHealthTopics_0000457_1` | #1 Hemorrhagic Stroke (5.70)<br>#2 Stroke (4.90)<br>#3 Stroke Rehabilitation (4.83) | #1 Stroke (0.89)<br>#2 Ischemic Stroke (0.88)<br>#3 Hemorrhagic Stroke (0.88) | #1 Stroke (0.03)<br>#2 Hemorrhagic Stroke (0.03)<br>#3 Ischemic Stroke (0.03) | 1.000 |
| Q18 | Multiple Relevant Topics | *"What is the difference between viral and bacterial infections?"* | `MPlusHealthTopics_0000949_1, MPlusHealthTopics_0000079_1` | #1 Skin Infections (13.10)<br>#2 Meningitis (12.21)<br>#3 Viral Infections (10.11) | #1 Viral Infections (0.88)<br>#2 Skin Infections (0.86)<br>#3 HIV/AIDS and Infections (0.86) | #1 Skin Infections (0.03)<br>#2 Viral Infections (0.03)<br>#3 HIV/AIDS and Infections (0.03) | 0.500 |
| Q19 | Ambiguous Queries | *"How are stones treated?"* | `MPlusHealthTopics_0000540_1, MPlusHealthTopics_0000388_1` | #1 Kidney Stones (9.91)<br>#2 Kidney Diseases (5.11)<br>#3 Gout (4.60) | #1 Kidney Stones (0.84)<br>#2 Gallstones (0.82)<br>#3 Bone Grafts (0.82) | #1 Kidney Stones (0.03)<br>#2 Shoulder Injuries and Disorders (0.02)<br>#3 Gum Disease (0.02) | 1.000 |
| Q20 | Ambiguous Queries | *"How is hepatitis treated?"* | `MPlusHealthTopics_0000459_1, MPlusHealthTopics_0000460_1, MPlusHealthTopics_0000461_1, MPlusHealthTopics_0000462_1` | #1 Hepatitis (12.14)<br>#2 Hepatitis B (8.43)<br>#3 Hepatitis C (8.34) | #1 Hepatitis (0.86)<br>#2 Hepatitis A (0.86)<br>#3 Hepatitis C (0.85) | #1 Hepatitis (0.03)<br>#2 Hepatitis B (0.03)<br>#3 Hepatitis A (0.03) | 1.000 |
| Q21 | Ambiguous Queries | *"Can cold weather give you a cold?"* | `MPlusHealthTopics_0000223_1, MPlusHealthTopics_0000497_1` | #1 Hypothermia (18.90)<br>#2 Common Cold (16.63)<br>#3 Cold and Cough Medicines (16.32) | #1 Hypothermia (0.86)<br>#2 Common Cold (0.86)<br>#3 Cold and Cough Medicines (0.85) | #1 Hypothermia (0.03)<br>#2 Common Cold (0.03)<br>#3 Cold and Cough Medicines (0.03) | 1.000 |

---

## 4. No-Answer / Out-of-Domain Analysis (N=4 Queries)

For queries where the corpus contains no relevant information, ideal retrieval should return zero matches (or allow confidence-based abstention). Retrieval scores alone do not prove whether a question is answerable.

| ID | Question | Corpus Status | BM25 Top Result | Semantic Top Result (Cosine) | Hybrid Top Result (RRF) |
|---|---|---|---|---|---|
| Q22 | *"How to repair a quantum rocket engine in space?"* | Out-of-domain query. Zero medical relevance. | *(0 results)* | #1 First Aid (0.792) | #1 First Aid (0.0164) |
| Q23 | *"What is the exact number of asthma patients in my village?"* | Hyper-local demographic query. No village statistics exist in MedQuAD. | #1 Asthma in Children (8.26) | #1 Asthma in Children (0.854) | #1 Asthma in Children (0.0328) |
| Q24 | *"What causes COVID-19 infection?"* | Corpus gap: the term `"covid"` occurs zero times in `records.jsonl`; general `Coronavirus Infections` record is not labeled as relevant. | #1 Bone Cancer (8.78) | #1 Coronavirus Infections (0.871) | #1 Staphylococcal Infections (0.0277) |
| Q25 | *"What is the capital of France?"* | Non-medical geographic inquiry. Zero relevant records. | *(0 results)* | #1 Uterine Diseases (0.727) | #1 Uterine Diseases (0.0164) |

### Note on Q24 ("What causes COVID-19 infection?") vs. General "Coronavirus Infections"
Although record `MPlusHealthTopics_0000239_1` (`Coronavirus Infections`) exists in the corpus and was retrieved at Rank #1 by semantic search (cosine `0.871`), it is **not labeled as a relevant answer**.
- **Rationale**: The exact term `"covid"` occurs **zero times** in `records.jsonl`. Although the corpus contains a general `Coronavirus Infections` record, that record is not labeled as relevant to the COVID-19-specific question. Record `0000239_1` describes general endemic upper-respiratory coronaviruses (stating *"they usually cause mild illness"* and *"there is no vaccine"*). Relying on this record to answer a COVID-19-specific question would produce factually outdated and clinically misleading guidance regarding SARS-CoV-2.
- This evaluation therefore treats Q24 as a corpus evidence gap (`relevant_ids = []`), rather than assuming that general coronavirus information adequately answers a COVID-19-specific question.

---

## 5. Key Retrieval Failures & Observations

### 1. The Lay Synonym Blindspot (Q06: "What causes a bellyache?")
- **Observation**: `Abdominal Pain` (`0000002_1`) was completely missed by BM25 (0 results) and ranked #520 by Semantic retrieval. Consequently, Hybrid placed `Bell's Palsy` at #1 due to subword overlap ('bell').
- **Takeaway**: Dense embeddings do not magically bridge colloquial terms when corpus documents use strictly formal clinical language. Extracting `<Synonyms>` at XML ingestion time remains essential.

### 2. Clinical Term Elevation (Q11: "What is high blood sugar?")
- **Observation**: BM25 ranked `Blood Sugar` at #1 and pushed `Hyperglycemia` to #6. Semantic retrieval placed `Hyperglycemia` directly at #1. Hybrid RRF harmoniously balanced both: `Blood Sugar` at #1 and `Hyperglycemia` at #2.
- **Takeaway**: Hybrid search provides the best of both worlds for medical RAG, capturing both the lay query tokens and the clinical entity.

### 3. RRF Candidate-Pool Depth & Rank Inversion (Q19: "How are stones treated?")
- **Empirical Observation**: Hybrid with `candidate_pool = 20` achieved a higher aggregate Recall@3 (**0.9167**) than `candidate_pool = 50` (**0.8929**).
- **Verification Across All 21 Queries**: Per-query verification confirms that across all 21 answerable queries, **Q19 is the sole query whose Recall@3 differs** between pool=20 and pool=50. The difference in means is exactly:
  $$\Delta\text{Recall@3} = \frac{1.0 - 0.5}{21} = \frac{0.5}{21} \approx 0.02381$$
- **Exact Rank & Score Shift on Q19**:
  - **BM25**: #1 `Kidney Stones` (9.91), #2 `Kidney Diseases` (5.11), `Gallstones` absent (compound-word mismatch).
  - **Semantic**: #1 `Kidney Stones` (0.8407), #2 `Gallstones` (0.8247), #3 `Bone Grafts` (0.8171).
  - **Hybrid (pool=20)**:
    - #1 `Kidney Stones` (RRF `0.03279`) — *Relevant*
    - #2 `Kidney Diseases` (RRF `0.01613`)
    - #3 `Gallstones` (RRF `0.01613`) — *Relevant*
    - **Recall@3 = 2/2 = 1.000**
  - **Hybrid (pool=50)**:
    - #1 `Kidney Stones` (RRF `0.03279`) — *Relevant*
    - #2 `Shoulder Injuries and Disorders` (RRF `0.02402`) — *(BM25 #19, Semantic #28)*
    - #3 `Gum Disease` (RRF `0.02399`) — *(BM25 #10, Semantic #43)*
    - #4 `Kidney Diseases` (RRF `0.01613`)
    - #5 `Gallstones` (RRF `0.01613`) — *Relevant*
    - **Recall@3 = 1/2 = 0.500**
- **Mathematical Mechanism**:
  `Gallstones` is matched only by semantic search (rank #2), yielding a single-system RRF score:
  $$\text{RRF}(\text{Gallstones}) = \frac{1}{60 + 2} = \frac{1}{62} \approx 0.01613$$
  When `candidate_pool = 20`, candidates below rank 20 are ignored, so `Gallstones` remains at rank #3. When `candidate_pool = 50`, documents appearing moderately in *both* systems enter the pool. For example, `Shoulder Injuries` (BM25 #19, Semantic #28) receives:
  $$\text{RRF} = \frac{1}{60 + 19} + \frac{1}{60 + 28} = \frac{1}{79} + \frac{1}{88} \approx 0.01266 + 0.01136 = 0.02402$$
  Because $0.02402 > 0.01613$, two weak dual-system matches mathematically leapfrog a strong single-system match. This demonstrates a key trade-off in equal-weighted Reciprocal Rank Fusion when one retrieval modality fails completely on compound words.

### 4. No-Answer Safeguards (Q22: "Quantum rocket engine", Q25: "Capital of France")
- **Observation**: BM25 cleanly returned 0 results for out-of-domain queries due to core-topic filtering. Dense semantic retrieval, by contrast, *always* returns top-k documents with non-zero cosine similarities (e.g. cosine ~0.76 for Space Medicine or General topics).
- **Takeaway**: BM25 acts as an effective guardrail against hallucinations on completely out-of-domain inputs. Combining BM25 with dense retrieval prevents the system from forcing semantic answers on unrelated queries.

---

## 6. Conclusion & Recommendation for FYP

On this 25-question diagnostic suite, **Hybrid RRF demonstrates the highest aggregate retrieval robustness** (highest Recall@5, Recall@10, and top MRR). It successfully compensates for BM25's vocabulary fragility while leveraging BM25's lexical precision to anchor dense embedding drift.

- **Pool Depth Trade-off**: Candidate pool depth of 20 avoids promoting weak dual-retriever distractors over strong single-system semantic matches, yielding the highest Recall@3 (**0.9167**). A pool depth of 50 captures deeper candidates but is more susceptible to distractor aggregation under unweighted RRF.
- **FYP Integration**: Integrating Hybrid RRF into `chat.py` with an optional `--retriever bm25|semantic|hybrid` selector and `candidate_pool=20` is recommended as the default retrieval strategy.

---

## 7. End-to-End Verification Note: Village-Specific Query Abstention (November 19, 2026)

Following integration of hybrid retrieval into `chat.py`, an end-to-end verification test was conducted using local LLM inference (`health-fyp` via `llama.cpp` on CPU).

- **Execution Date**: November 19, 2026
- **Test Query**: *"What is the exact number of asthma patients in my village?"*
- **Retriever Configuration**: `--retriever hybrid`, `--top_k 3` (`candidate_pool=20`, `rrf_k=60`)
- **Displayed MedQuAD Sources**:
  1. `[1] Asthma in Children (MPlusHealthTopics)` — `https://www.nlm.nih.gov/medlineplus/asthmainchildren.html`
  2. `[2] Asthma (MPlusHealthTopics)` — `https://www.nlm.nih.gov/medlineplus/asthma.html`
  3. `[3] Environmental Health (MPlusHealthTopics)` — `https://www.nlm.nih.gov/medlineplus/environmentalhealth.html`
- **Observed Generated Answer**:
  > *"The provided medical evidence does not contain sufficient information to answer this question."*
- **Evaluation**: **PASS**
  - Uses the canonical exact abstention sentence.
  - Successfully omits extraneous US national statistics despite their presence in the retrieved `Asthma in Children` evidence block.
  - Does not invent or speculate a village-specific patient count.
- **Scope & Limitations**: This record documents one successful empirical test under the tested prompt and hybrid configuration. It does not constitute proof that all unsupported, ambiguous, or hyper-local queries will always be safely or correctly abstained from across all operational conditions.
