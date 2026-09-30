"""Interactively search the saved FAISS document index."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import retrieve_documents


def main() -> int:
	query = input("Enter your question: ").strip()
	if not query:
		print("Please enter a non-empty question.")
		return 0

	try:
		documents = retrieve_documents(query, k=4)
	except FileNotFoundError as error:
		print(f"Vector database unavailable: {error}", file=sys.stderr)
		return 1
	except Exception as error:
		print(f"Could not retrieve documents: {error}", file=sys.stderr)
		return 1

	if not documents:
		print("No matching document chunks were found.")
		return 0

	for result_number, document in enumerate(documents, start=1):
		print(f"\n--- Retrieved Document {result_number} ---")
		print(f"Source: {document.metadata.get('source', 'Unknown')}")
		print(f"Page: {document.metadata.get('page', 'Unknown')}")
		print(f"Text:\n{document.page_content}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())