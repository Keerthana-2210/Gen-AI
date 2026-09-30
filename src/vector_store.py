"""Create and save a local FAISS index for document chunks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import VECTORSTORE_DIR
from src.embeddings import create_embeddings

if TYPE_CHECKING:
	from langchain_community.vectorstores import FAISS


def create_faiss_vector_store(
	chunks: Sequence[Document], embeddings: Embeddings
) -> FAISS:
	"""Embed chunks and create a FAISS vector store."""
	if not chunks:
		raise ValueError("Cannot create a FAISS vector store from empty chunks.")

	try:
		from langchain_community.vectorstores import FAISS

		return FAISS.from_documents(list(chunks), embeddings)
	except Exception as error:
		raise RuntimeError(
			"Could not create the FAISS vector store. Check that faiss-cpu and its dependencies are installed."
		) from error


def save_faiss_vector_store(
	vector_store: FAISS, vectorstore_dir: Path = VECTORSTORE_DIR
) -> None:
	"""Save the FAISS index and its document metadata locally."""
	try:
		vectorstore_dir.mkdir(parents=True, exist_ok=True)
		vector_store.save_local(str(vectorstore_dir))
	except Exception as error:
		raise RuntimeError(
			f"Could not save the FAISS vector store to '{vectorstore_dir}': {error}"
		) from error


def load_faiss_vector_store(
	embeddings: Embeddings | None = None,
	vectorstore_dir: Path = VECTORSTORE_DIR,
) -> FAISS:
	"""Load a previously saved local FAISS index and its document metadata."""
	if not vectorstore_dir.is_dir():
		raise FileNotFoundError(
			f"FAISS vector store not found at '{vectorstore_dir}'. Build it first with scripts/build_vectorstore.py."
		)

	missing_files = [
		name for name in ("index.faiss", "index.pkl") if not (vectorstore_dir / name).is_file()
	]
	if missing_files:
		raise FileNotFoundError(
			f"FAISS vector store at '{vectorstore_dir}' is incomplete; missing: {', '.join(missing_files)}"
		)

	if embeddings is None:
		embeddings = create_embeddings()

	try:
		from langchain_community.vectorstores import FAISS

		return FAISS.load_local(
			str(vectorstore_dir),
			embeddings,
			allow_dangerous_deserialization=True,
		)
	except Exception as error:
		raise RuntimeError(
			f"Could not load the FAISS vector store from '{vectorstore_dir}': {error}"
		) from error


def build_vector_store(
	chunks: Sequence[Document],
	embeddings: Embeddings | None = None,
	vectorstore_dir: Path = VECTORSTORE_DIR,
) -> FAISS:
	"""Create embeddings, build the FAISS index, and save it locally."""
	if not chunks:
		raise ValueError("Cannot build a FAISS vector store without document chunks.")

	if embeddings is None:
		embeddings = create_embeddings()

	vector_store = create_faiss_vector_store(chunks, embeddings)
	save_faiss_vector_store(vector_store, vectorstore_dir)
	return vector_store
