"""Build a local FAISS index from PDFs in data/documents/."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import VECTORSTORE_DIR
from src.document_loader import find_pdf_files, load_pdfs
from src.embeddings import create_embeddings
from src.text_splitter import split_documents
from src.vector_store import create_faiss_vector_store, save_faiss_vector_store


def main() -> None:
    print("Loading documents...")
    pdf_files = find_pdf_files()
    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found in data/documents/. Add at least one PDF and try again."
        )

    documents = load_pdfs()
    print(f"Documents loaded: {len(documents)}")
    if not documents or not any(document.page_content.strip() for document in documents):
        raise ValueError("The PDF files did not contain any extractable text.")

    print("Creating chunks...")
    chunks = split_documents(documents)
    print(f"Chunks created: {len(chunks)}")
    if not chunks:
        raise ValueError("No text chunks were created from the loaded PDF documents.")

    print("Loading embedding model...")
    embeddings = create_embeddings()

    print("Creating FAISS vector store...")
    vector_store = create_faiss_vector_store(chunks, embeddings)
    save_faiss_vector_store(vector_store)

    expected_files = (VECTORSTORE_DIR / "index.faiss", VECTORSTORE_DIR / "index.pkl")
    missing_files = [path.name for path in expected_files if not path.is_file()]
    if missing_files:
        raise RuntimeError(
            f"Vector store save completed, but expected index files are missing: {', '.join(missing_files)}"
        )

    print(f"Vector store saved successfully: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Build failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
