"""Centralize filesystem paths and text chunking settings."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
