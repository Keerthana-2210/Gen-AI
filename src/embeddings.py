"""Load the Hugging Face sentence-transformer used to embed document chunks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import EMBEDDING_MODEL_NAME

if TYPE_CHECKING:
	from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings() -> HuggingFaceEmbeddings:
	"""Load the model, downloading it on first use and caching it locally."""
	try:
		from langchain_huggingface import HuggingFaceEmbeddings

		return HuggingFaceEmbeddings(
			model_name=EMBEDDING_MODEL_NAME,
			model_kwargs={"device": "cpu"},
			encode_kwargs={"normalize_embeddings": True},
		)
	except Exception as error:
		raise RuntimeError(
			f"Could not load embedding model '{EMBEDDING_MODEL_NAME}'. "
			"Check the Hugging Face integration dependencies and internet access on first use. "
			f"Details: {error}"
		) from error
