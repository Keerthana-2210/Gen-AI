"""Run the PDF ingestion and chunking pipeline from the command line."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_pdfs
from src.text_splitter import split_documents


def main() -> None:
    pages = load_pdfs()
    chunks = split_documents(pages)

    print(f"PDF pages loaded: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")

    for chunk_number, chunk in enumerate(chunks[:2], start=1):
        print(f"\n--- Chunk {chunk_number} ---")
        print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
        print(f"Page: {chunk.metadata.get('page', 'Unknown')}")
        print(f"Text: {chunk.page_content}")


if __name__ == "__main__":
    main()
