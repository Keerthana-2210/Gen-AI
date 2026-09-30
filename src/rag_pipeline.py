"""Retrieve document chunks and generate context-grounded Gemini answers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document

from src.config import PROJECT_ROOT
from src.embeddings import create_embeddings
from src.vector_store import load_faiss_vector_store

if TYPE_CHECKING:
	from langchain_community.vectorstores import FAISS


GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"


class SourceReference(TypedDict):
	source: str
	page: int | str


class AnswerResult(TypedDict):
	answer: str
	sources: list[SourceReference]


NO_INFORMATION_ANSWER = "I could not find this information in the provided documents."

RAG_SYSTEM_PROMPT = """You are MedDoc-RAG, an informational document assistant, not a medical diagnosis system.
Answer the user's question using only the retrieved document context provided with the request. Do not use outside knowledge.
If the answer is not explicitly supported by that context, respond exactly:
"I could not find this information in the provided documents."
Do not diagnose diseases, recommend personal medical treatment, prescribe medicines, or make clinical decisions.
Keep the answer informational and grounded in the documents. Do not invent facts, filenames, citations, or page numbers.
Use only the source and page information explicitly present in the retrieved context for citations."""


def retrieve_documents(query: str, k: int = 4) -> list[Document]:
	"""Return the top-k chunks relevant to a non-empty query."""
	if not query or not query.strip():
		return []
	if k < 1:
		raise ValueError("k must be at least 1.")

	embeddings = create_embeddings()
	vector_store = load_faiss_vector_store(embeddings)
	return vector_store.similarity_search(query.strip(), k=k)


def _collect_sources(documents: list[Document]) -> list[SourceReference]:
	"""Collect unique source/page pairs from retrieved document metadata."""
	sources: list[SourceReference] = []
	seen: set[tuple[str, int | str]] = set()
	for document in documents:
		source = document.metadata.get("source")
		page = document.metadata.get("page")
		if not isinstance(source, str) or not source or not isinstance(page, (int, str)):
			continue
		citation = (source, page)
		if citation not in seen:
			seen.add(citation)
			sources.append({"source": source, "page": page})
	return sources


def answer_question(
	query: str, k: int = 4, vector_store: FAISS | None = None
) -> AnswerResult:
	"""Retrieve evidence, ask Gemini for a grounded answer, and return its sources."""
	if not isinstance(query, str) or not query.strip():
		raise ValueError("Please enter a non-empty question.")
	if k < 1:
		raise ValueError("k must be at least 1.")

	load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
	api_key = os.getenv("GEMINI_API_KEY", "").strip()
	if not api_key or api_key == "YOUR_REAL_KEY" or api_key == "your_gemini_api_key_here":
		raise RuntimeError(
			"GEMINI_API_KEY is missing. Add your key to the project .env file; "
			"see .env.example for the required variable name."
		)

	if vector_store is None:
		documents = retrieve_documents(query, k=k)
	else:
		documents = vector_store.similarity_search(query.strip(), k=k)
	if not documents:
		return {"answer": NO_INFORMATION_ANSWER, "sources": []}

	context = "\n\n".join(
		f"[Source: {document.metadata.get('source', 'Unknown')} | "
		f"Page: {document.metadata.get('page', 'Unknown')}]\n{document.page_content}"
		for document in documents
	)
	try:
		from google import genai
		from google.genai import types

		client = genai.Client(api_key=api_key)
		response = client.models.generate_content(
			model=GEMINI_MODEL_NAME,
			contents=f"Retrieved document context:\n{context}\n\nQuestion: {query.strip()}",
			config=types.GenerateContentConfig(
				system_instruction=RAG_SYSTEM_PROMPT,
				temperature=0,
				automatic_function_calling=types.AutomaticFunctionCallingConfig(
					disable=True,
				),
			),
		)
	except Exception as error:
		safe_message = str(error).replace(api_key, "[REDACTED]")
		raise RuntimeError(
			f"Gemini answer generation failed ({type(error).__name__}): {safe_message}"
		) from None

	answer = (response.text or "").strip()
	if not answer:
		answer = NO_INFORMATION_ANSWER

	return {"answer": answer, "sources": _collect_sources(documents)}
