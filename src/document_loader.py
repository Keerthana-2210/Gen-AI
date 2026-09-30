"""Load PDF pages from the local documents directory."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.config import DOCUMENTS_DIR


def find_pdf_files(documents_dir: Path = DOCUMENTS_DIR) -> list[Path]:
	"""Return PDF files directly inside the documents directory."""
	if not documents_dir.is_dir():
		return []

	return sorted(
		(path for path in documents_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
		key=lambda path: path.name.lower(),
	)


def load_pdfs(documents_dir: Path = DOCUMENTS_DIR) -> list[Document]:
	"""Load all PDFs, preserving each filename and one-based page number."""
	documents: list[Document] = []

	for pdf_path in find_pdf_files(documents_dir):
		try:
			pages = PyPDFLoader(str(pdf_path)).load()
		except Exception as error:
			raise RuntimeError(f"Could not load PDF '{pdf_path.name}': {error}") from error

		for page_number, page in enumerate(pages, start=1):
			page.page_content = " ".join(page.page_content.split())
			page.metadata["source"] = pdf_path.name
			page.metadata["page"] = page_number
			documents.append(page)

	return documents
