import unittest

from src.config import DOCUMENTS_DIR
from src.document_loader import find_pdf_files, load_pdfs
from src.text_splitter import split_documents


class DocumentPipelineTests(unittest.TestCase):
    def test_pdf_documents_load_and_split_with_metadata(self):
        pdf_files = find_pdf_files()
        self.assertTrue(pdf_files, f"No PDF files found in {DOCUMENTS_DIR}")

        documents = load_pdfs()
        self.assertTrue(documents)
        self.assertTrue(all(document.page_content.strip() for document in documents))
        self.assertTrue(
            all(document.page_content == " ".join(document.page_content.split()) for document in documents)
        )

        chunks = split_documents(documents)
        self.assertTrue(chunks)
        self.assertTrue(all(chunk.page_content.strip() for chunk in chunks))
        self.assertTrue(all(chunk.metadata.get("source") for chunk in chunks))
        self.assertTrue(all(isinstance(chunk.metadata.get("page"), int) for chunk in chunks))

        source_filenames = {pdf_file.name for pdf_file in pdf_files}
        self.assertTrue(all(chunk.metadata["source"] in source_filenames for chunk in chunks))
        self.assertTrue(all(chunk.metadata["page"] >= 1 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
