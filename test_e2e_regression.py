"""
End-to-End Regression Tests for MedQuAD Bilingual Medical Assistant
-------------------------------------------------------------------
Tests end-to-end invariants against the local llama.cpp server:
1. English answering & absence of Roman Urdu leakage.
2. Roman Urdu answering & accurate terminology (chest tightness != abdominal pain).
3. Exact canonical abstention for unsupported abdominal-pain queries.
4. Exact canonical abstention for location/demographic queries.

If the local llama.cpp server is not running on http://127.0.0.1:8765,
these tests are safely skipped without failing the fast unit test suite.

Run with:
    python -m unittest test_e2e_regression.py
"""

import unittest
import urllib.request
import urllib.error
from chat import (
    call_llm,
    retrieve_evidence,
    format_evidence_block,
    is_roman_urdu,
    ENGLISH_ABSTENTION,
    ROMAN_URDU_ABSTENTION,
    DEFAULT_API_URL,
    DEFAULT_MODEL_ALIAS,
)


def is_llm_server_running(url: str = "http://127.0.0.1:8765/health") -> bool:
    """Checks whether the local llama.cpp server is responsive."""
    try:
        req = urllib.request.Request(url=url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


SERVER_RUNNING = is_llm_server_running()


@unittest.skipUnless(SERVER_RUNNING, "Local llama.cpp server (http://127.0.0.1:8765) is not available.")
class TestEndToEndSafetyInvariants(unittest.TestCase):
    """Verifies behavioral and safety invariants during end-to-end RAG execution."""

    def _execute_query(self, query: str, top_k: int = 3) -> str:
        results = retrieve_evidence(query=query, retriever_type="hybrid", top_k=top_k)
        self.assertTrue(len(results) > 0, f"Retrieval failed to return records for query: '{query}'")
        evidence_text = format_evidence_block(results)
        return call_llm(
            query=query,
            evidence_text=evidence_text,
            api_url=DEFAULT_API_URL,
            model=DEFAULT_MODEL_ALIAS,
            timeout=180,
        )

    def test_english_query_returns_grounded_english_without_abstention(self):
        query = "What is asthma?"
        answer = self._execute_query(query)

        # Invariant 1: Must not be classified as Roman Urdu
        self.assertFalse(is_roman_urdu(query))

        # Invariant 2: Must not contain canonical abstention sentences
        self.assertNotIn(ENGLISH_ABSTENTION, answer)
        self.assertNotIn(ROMAN_URDU_ABSTENTION, answer)

        # Invariant 3: Grounded medical content in English
        self.assertIn("airways", answer.lower())
        self.assertIn("asthma", answer.lower())

    def test_roman_urdu_asthma_query_terminology_and_no_abstention(self):
        query = "Dama ki alamat kya hain?"
        answer = self._execute_query(query)

        # Invariant 1: Must be Roman Urdu
        self.assertTrue(is_roman_urdu(query))

        # Invariant 2: Chest tightness must not be translated as abdominal pain
        self.assertNotIn("pait ka dard", answer.lower())
        self.assertNotIn("pait dard", answer.lower())

        # Invariant 3: Must include chest terminology
        self.assertTrue(
            "seene" in answer.lower() or "chest" in answer.lower(),
            "Expected chest-related terminology in asthma symptom translation."
        )

        # Invariant 4: No contradictory abstention
        self.assertNotIn(ROMAN_URDU_ABSTENTION, answer)
        self.assertNotIn(ENGLISH_ABSTENTION, answer)

    def test_insufficient_evidence_abdominal_pain_exact_abstention(self):
        query = "Pait dard ki kya waja hai?"
        answer = self._execute_query(query)

        # Invariant: Must return the EXACT canonical Roman Urdu abstention string
        self.assertEqual(
            answer,
            ROMAN_URDU_ABSTENTION,
            "Expected exact canonical Roman Urdu abstention for unsupported cause query."
        )

    def test_location_specific_exact_abstention(self):
        query = "Meray gaon mein asthma kay kitnay mareez hain?"
        answer = self._execute_query(query)

        # Invariant: Must return the EXACT canonical Roman Urdu abstention string
        self.assertEqual(
            answer,
            ROMAN_URDU_ABSTENTION,
            "Expected exact canonical Roman Urdu abstention for unsupported village query."
        )


if __name__ == "__main__":
    unittest.main()
