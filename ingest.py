"""
MedQuAD Dataset Ingestion Script (Phase 1)
------------------------------------------
This script parses medical question-and-answer XML documents from the
MedQuAD dataset using Python's standard library (no external dependencies required).

It extracts clean question-answer pairs, associated topics, source URLs,
and creates uniquely identified JSON records stored in 'records.jsonl'.
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def ingest_dataset(
    data_dir: str = "data/4_MPlus_Health_Topics_QA",
    output_file: str = "records.jsonl",
) -> None:
    """
    Parses XML files from the specified dataset directory and writes valid
    QA records to a JSON Lines (.jsonl) file.

    Args:
        data_dir: Path to the directory containing XML files.
        output_file: Target JSONL filename.
    """
    dataset_path = Path(data_dir)

    # 1. Validate that the dataset directory exists
    if not dataset_path.exists() or not dataset_path.is_dir():
        print(f"Error: Dataset directory '{dataset_path}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    # 2. Find all XML files recursively (includes any nested subfolders)
    xml_files = sorted(list(dataset_path.rglob("*.xml")))

    if not xml_files:
        print(f"Error: No XML files found inside '{dataset_path}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(xml_files)} XML file(s) in '{dataset_path}'. Starting ingestion...")

    files_processed = 0
    pairs_extracted = 0
    records_written = 0

    output_path = Path(output_file)

    # 3. Process each XML file and write records to JSONL
    with output_path.open("w", encoding="utf-8") as f_out:
        for file_path in xml_files:
            files_processed += 1

            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
            except ET.ParseError as err:
                print(f"Warning: Skipping malformed XML in '{file_path.name}': {err}", file=sys.stderr)
                continue

            # Extract document-level metadata safely
            # Example root: <Document id="0000001" source="MPlusHealthTopics" url="https://...">
            collection_name = root.attrib.get("source", "").strip() or dataset_path.name
            source_url = root.attrib.get("url", "").strip()

            # Extract Topic / Medical Focus
            focus_elem = root.find("Focus")
            topic = focus_elem.text.strip() if focus_elem is not None and focus_elem.text else ""

            # Extract Question-Answer Pairs
            # Look for <QAPair> elements inside <QAPairs> or anywhere in the document
            qa_elements = root.findall(".//QAPair")

            for idx, qa in enumerate(qa_elements, start=1):
                # Retrieve pair ID attribute with fallback to loop index
                pair_id = qa.attrib.get("pid", "").strip() or str(idx)

                # Extract Question
                q_elem = qa.find("Question")
                question_text = q_elem.text.strip() if q_elem is not None and q_elem.text else ""
                qtype = q_elem.attrib.get("qtype", "").strip() if q_elem is not None else ""

                # Extract Answer
                a_elem = qa.find("Answer")
                answer_text = a_elem.text.strip() if a_elem is not None and a_elem.text else ""

                # Only keep pairs with non-empty question AND answer
                if not question_text or not answer_text:
                    continue

                pairs_extracted += 1

                # Generate a stable, unique record ID using collection, filename stem, and pair ID
                record_id = f"{collection_name}_{file_path.stem}_{pair_id}"

                # Structure the ingested record
                record = {
                    "id": record_id,
                    "collection": collection_name,
                    "topic": topic,
                    "question": question_text,
                    "answer": answer_text,
                    "source_url": source_url,
                    "pair_id": pair_id,
                    "question_type": qtype,
                    "file_name": file_path.name,
                }

                # Write record as a single JSON line
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1

    # 4. Final verification and summary output
    if records_written == 0:
        print(f"Error: Processed {files_processed} files, but no valid QA records were found.", file=sys.stderr)
        sys.exit(1)

    print("\n--- Ingestion Summary ---")
    print(f"XML files processed:             {files_processed}")
    print(f"Complete QA pairs extracted:     {pairs_extracted}")
    print(f"Records written to '{output_file}': {records_written}")
    print("-------------------------")


if __name__ == "__main__":
    # Target directory defaults to 4_MPlus_Health_Topics_QA
    target_data_directory = "data/4_MPlus_Health_Topics_QA"
    ingest_dataset(data_dir=target_data_directory, output_file="records.jsonl")
