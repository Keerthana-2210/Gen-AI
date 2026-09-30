"""Split loaded document pages into retrieval-sized text chunks."""

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def split_documents(documents: Sequence[Document]) -> list[Document]:
	"""Split documents using the configured chunk size and overlap."""
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=CHUNK_SIZE,
		chunk_overlap=CHUNK_OVERLAP,
	)
	return splitter.split_documents(list(documents))
