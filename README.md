# MedDoc-RAG

MedDoc-RAG is a planned document-based healthcare information retrieval assistant. It will help users find and understand information contained in documents they upload.

**Safety limitation:** MedDoc-RAG is an informational document assistant, not a medical diagnosis system. It will not diagnose diseases, prescribe medicines, recommend treatments, or make clinical decisions. Responses will be limited to explaining retrieved document content.

## Planned architecture and technologies

1. **Interface:** Streamlit will provide document upload, question entry, retrieved-source display, and the informational-assistant notice.
2. **Loading and splitting:** PyPDF and LangChain will extract text from PDFs and split it into manageable chunks.
3. **Embeddings and index:** Hugging Face embedding models will convert chunks into vectors; FAISS will store and search those vectors.
4. **Retrieval and explanation:** LangChain will retrieve relevant chunks and pass them to an LLM API to produce an answer grounded in the uploaded documents.
5. **Configuration:** `python-dotenv` will load local settings from `.env`; real credentials must never be committed.

## Planned workflow

1. Upload a supported document.
2. Extract and split its text.
3. Create embeddings and index the chunks in FAISS.
4. Submit a question about the uploaded material.
5. Retrieve relevant passages and generate an informational explanation with source references.

The application, PDF processing, vector indexing, and RAG pipeline are not implemented yet. `data/documents/` is intended for local documents and is excluded from Git to reduce the risk of committing sensitive material.

## Project structure

- `app.py`: Planned Streamlit entry point.
- `src/`: Planned document processing, retrieval, configuration, and pipeline modules.
- `data/documents/`: Local uploaded documents; not tracked by Git.
- `tests/`: Future automated tests.
- `requirements.txt`: Python dependencies for the planned stack.
- `.env.example`: Names and placeholders for future local configuration; contains no credentials.
