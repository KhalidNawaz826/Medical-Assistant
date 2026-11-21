"""
Unit and Regression Tests for MedQuAD Bilingual Medical Assistant
-----------------------------------------------------------------
Automated, deterministic test suite covering:
1. English and Roman Urdu language detection.
2. Safe Roman Urdu medical synonym normalization (whole-word boundaries, idempotence).
3. Exact canonical abstention string invariants.
4. Case-insensitive canonical abstention enforcement.
5. Terminology safety guidelines (chest tightness vs. abdominal pain).

Run with:
    python -m unittest test_chat.py
"""

import unittest
from chat import (
    is_roman_urdu,
    normalize_medical_synonyms,
    enforce_canonical_abstention,
    ENGLISH_ABSTENTION,
    ROMAN_URDU_ABSTENTION,
    ROMAN_URDU_MEDICAL_SYNONYMS,
    ROMAN_URDU_MARKERS,
    SYSTEM_PROMPT,
)


class TestLanguageDetection(unittest.TestCase):
    """Verifies lexical marker-based language detection for English and Roman Urdu."""

    def test_english_queries_detected_as_not_roman_urdu(self):
        english_queries = [
            "What is asthma?",
            "What are the symptoms of asthma?",
            "Tell me what causes asthma.",
            "Is asthma dangerous?",
            "What causes high blood sugar?",
            "How do I treat hypertension?",
            "Can cold weather give you a cold?",
            "What is the difference between asthma and diabetes?",
        ]
        for query in english_queries:
            with self.subTest(query=query):
                self.assertFalse(
                    is_roman_urdu(query),
                    f"English query '{query}' was incorrectly classified as Roman Urdu."
                )

    def test_roman_urdu_queries_detected(self):
        roman_urdu_queries = [
            "Asthma ki kya alamat hoti hain?",
            "Dama ki alamat kya hain?",
            "Pait dard ki kya waja hai?",
            "Pait ka dard kyun hota hai?",
            "Meray gaon mein asthma kay kitnay mareez hain?",
            "Khaansi ka ilaj bataiye",
            "Khansi ki dawa kya hai?",
            "Diabetes ki alamat kya hain?",
        ]
        for query in roman_urdu_queries:
            with self.subTest(query=query):
                self.assertTrue(
                    is_roman_urdu(query),
                    f"Roman Urdu query '{query}' was not detected as Roman Urdu."
                )


class TestSynonymNormalization(unittest.TestCase):
    """Verifies safe medical synonym mapping and boundary preservation."""

    def test_english_text_remains_unchanged(self):
        english_queries = [
            "What is asthma?",
            "What are the symptoms of asthma?",
            "Tell me what causes cough and chest pain.",
            "The damage to the lungs was severe.",
        ]
        for query in english_queries:
            with self.subTest(query=query):
                self.assertEqual(
                    normalize_medical_synonyms(query),
                    query,
                    f"English text '{query}' was unexpectedly modified by synonym normalization."
                )

    def test_dama_to_asthma_mapping(self):
        self.assertEqual(
            normalize_medical_synonyms("Dama ki alamat kya hain?"),
            "asthma ki alamat kya hain?"
        )
        self.assertEqual(
            normalize_medical_synonyms("dama ka mareez"),
            "asthma ka mareez"
        )

    def test_pait_dard_mapping(self):
        # Shorter compound
        self.assertEqual(
            normalize_medical_synonyms("Pait dard ki kya waja hai?"),
            "abdominal pain ki kya waja hai?"
        )
        # Longer compound
        self.assertEqual(
            normalize_medical_synonyms("pait ka dard kyun hota hai?"),
            "abdominal pain kyun hota hai?"
        )

    def test_khaansi_and_khansi_to_cough_mapping(self):
        self.assertEqual(
            normalize_medical_synonyms("Khaansi ki kya alamat hain?"),
            "cough ki kya alamat hain?"
        )
        self.assertEqual(
            normalize_medical_synonyms("khansi ka ilaj"),
            "cough ka ilaj"
        )

    def test_whole_word_boundaries_prevent_subword_corruption(self):
        # 'damage' must not be corrupted to 'asthmage'
        self.assertEqual(
            normalize_medical_synonyms("The damage was investigated."),
            "The damage was investigated."
        )

    def test_idempotence(self):
        queries = [
            "Dama aur pait ka dard aur khaansi",
            "What is asthma?",
            "Pait dard ki kya waja hai?",
        ]
        for query in queries:
            with self.subTest(query=query):
                first_pass = normalize_medical_synonyms(query)
                second_pass = normalize_medical_synonyms(first_pass)
                self.assertEqual(
                    first_pass,
                    second_pass,
                    f"Normalization was not idempotent for query '{query}'."
                )


class TestAbstentionInvariants(unittest.TestCase):
    """Verifies canonical abstention strings and case-insensitive enforcement safeguard."""

    def test_exact_canonical_abstention_strings(self):
        self.assertEqual(
            ENGLISH_ABSTENTION,
            "The provided medical evidence does not contain sufficient information to answer this question."
        )
        self.assertEqual(
            ROMAN_URDU_ABSTENTION,
            "Faraham kardah tibbi saboot is sawal ka jawab denay kay liye kafi maloomat nahi rakhta."
        )

    def test_enforce_canonical_abstention_exact_match(self):
        self.assertEqual(
            enforce_canonical_abstention(ENGLISH_ABSTENTION),
            ENGLISH_ABSTENTION
        )
        self.assertEqual(
            enforce_canonical_abstention(ROMAN_URDU_ABSTENTION),
            ROMAN_URDU_ABSTENTION
        )

    def test_enforce_canonical_abstention_strips_preamble_exact_case(self):
        prefixed_urdu = f"Pait dard ki kai wajoohat ho sakti hain. {ROMAN_URDU_ABSTENTION}"
        self.assertEqual(
            enforce_canonical_abstention(prefixed_urdu),
            ROMAN_URDU_ABSTENTION
        )

        prefixed_english = f"There can be many reasons. {ENGLISH_ABSTENTION}"
        self.assertEqual(
            enforce_canonical_abstention(prefixed_english),
            ENGLISH_ABSTENTION
        )

    def test_enforce_canonical_abstention_case_insensitive_leading_shift(self):
        # LLM emits lowercase 'faraham' when continuing a conversational sentence
        lowercase_shift_urdu = (
            "Pait dard ki waja (causes) faraham kardah tibbi saboot "
            "is sawal ka jawab denay kay liye kafi maloomat nahi rakhta."
        )
        self.assertEqual(
            enforce_canonical_abstention(lowercase_shift_urdu),
            ROMAN_URDU_ABSTENTION
        )

        lowercase_shift_english = (
            "Regarding causes, the provided medical evidence does not contain "
            "sufficient information to answer this question."
        )
        self.assertEqual(
            enforce_canonical_abstention(lowercase_shift_english),
            ENGLISH_ABSTENTION
        )

    def test_enforce_canonical_abstention_preserves_grounded_answers(self):
        # Answers that do not signal abstention must remain completely untouched
        valid_english = "The symptoms of asthma include wheezing and coughing."
        self.assertEqual(
            enforce_canonical_abstention(valid_english),
            valid_english
        )

        valid_urdu = (
            "Dama ki alamat in hain: saans lete waqt seeti jaisi awaaz aana, "
            "khaansi, aur seene mein jakran ya tangi."
        )
        self.assertEqual(
            enforce_canonical_abstention(valid_urdu),
            valid_urdu
        )


class TestTerminologySafetyGuidelines(unittest.TestCase):
    """Verifies that system prompt and mappings enforce medical concept separation."""

    def test_synonym_mappings_separation(self):
        # Abdominal pain mappings must never map to chest concepts
        for term, target in ROMAN_URDU_MEDICAL_SYNONYMS.items():
            if "pait" in term:
                self.assertEqual(target, "abdominal pain")
            if "dama" in term:
                self.assertEqual(target, "asthma")
            if "khansi" in term or "khaansi" in term:
                self.assertEqual(target, "cough")

    def test_system_prompt_terminology_invariants(self):
        # Prompt must explicitly forbid translating chest tightness as abdominal pain / pait dard
        self.assertIn("chest tightness", SYSTEM_PROMPT.lower())
        self.assertIn("pait dard", SYSTEM_PROMPT.lower())
        self.assertIn("seene mein jakran", SYSTEM_PROMPT.lower())
        # Prompt must clearly define wheezing
        self.assertIn("seeti jaisi awaaz aana", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
