"""Streamlit interface for the MedDoc-RAG document information assistant."""

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from src.rag_pipeline import NO_INFORMATION_ANSWER, answer_question


st.set_page_config(
	page_title="MedDoc-RAG",
	layout="wide",
	initial_sidebar_state="expanded",
)


def clear_uploaded_document() -> None:
	"""Discard the session-only index and reset its file uploader."""
	for key in (
		"uploaded_fingerprint",
		"uploaded_vector_store",
		"uploaded_filename",
		"uploaded_page_count",
		"uploaded_chunk_count",
		"upload_error",
		"submitted_question",
		"answer_result",
		"request_error",
	):
		st.session_state.pop(key, None)
	st.session_state["question_input"] = ""
	st.session_state["uploader_version"] = st.session_state.get("uploader_version", 0) + 1


def clear_active_upload_state() -> None:
	"""Remove upload-specific state when the uploader is empty or a file changes."""
	for key in (
		"uploaded_fingerprint",
		"uploaded_vector_store",
		"uploaded_filename",
		"uploaded_page_count",
		"uploaded_chunk_count",
		"upload_error",
	):
		st.session_state.pop(key, None)


def process_uploaded_pdf(uploaded_file) -> None:
	"""Build an in-memory FAISS index once for the uploaded PDF in this session."""
	from src.document_loader import load_pdfs
	from src.embeddings import create_embeddings
	from src.text_splitter import split_documents
	from src.vector_store import create_faiss_vector_store

	file_name = Path(uploaded_file.name).name
	file_bytes = uploaded_file.getvalue()
	fingerprint = hashlib.sha256(file_bytes).hexdigest()
	if fingerprint == st.session_state.get("uploaded_fingerprint"):
		return

	clear_active_upload_state()
	for key in ("submitted_question", "answer_result", "request_error"):
		st.session_state.pop(key, None)
	st.session_state["uploaded_fingerprint"] = fingerprint
	if Path(file_name).suffix.lower() != ".pdf" or uploaded_file.type not in (
		None,
		"",
		"application/pdf",
		"application/x-pdf",
	):
		st.session_state["upload_error"] = "Please upload a valid PDF file."
		return
	if not file_bytes:
		st.session_state["upload_error"] = "The uploaded PDF is empty."
		return

	try:
		with st.spinner("Processing document..."):
			with tempfile.TemporaryDirectory(prefix="meddoc-rag-") as temp_dir:
				temp_pdf = Path(temp_dir) / file_name
				temp_pdf.write_bytes(file_bytes)
				try:
					documents = load_pdfs(Path(temp_dir))
				except Exception as error:
					raise RuntimeError(f"PDF text extraction failed: {error}") from error

		if not documents or not any(document.page_content.strip() for document in documents):
			st.session_state["upload_error"] = "No readable text was found in the uploaded PDF."
			return

		chunks = split_documents(documents)
		if not chunks:
			st.session_state["upload_error"] = "No text chunks could be created from the uploaded PDF."
			return

		try:
			embeddings = create_embeddings()
		except Exception as error:
			raise RuntimeError(f"Could not create embeddings for the uploaded PDF: {error}") from error

		try:
			vector_store = create_faiss_vector_store(chunks, embeddings)
		except Exception as error:
			raise RuntimeError(f"Could not create a vector index for the uploaded PDF: {error}") from error

		st.session_state["uploaded_vector_store"] = vector_store
		st.session_state["uploaded_filename"] = file_name
		st.session_state["uploaded_page_count"] = len(documents)
		st.session_state["uploaded_chunk_count"] = len(chunks)
	except Exception as error:
		st.session_state["upload_error"] = str(error)

st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

	:root {
		--blue: #2563EB;
		--dark-blue: #1E3A8A;
		--light-blue: #EFF6FF;
		--canvas: #F8FBFF;
		--white: #FFFFFF;
		--ink: #0F172A;
		--secondary: #475569;
		--border: #D6E4F0;
	}

	html, body, [class*="css"] {
		font-family: 'DM Sans', 'Segoe UI', sans-serif;
		color: var(--ink);
	}
	p, li, label { color: var(--ink); }
	.stApp, [data-testid="stAppViewContainer"] { background-color: var(--canvas); }
	[data-testid="stHeader"] { background: transparent; box-shadow: none; }
	[data-testid="stSidebar"] {
		background: var(--light-blue);
		border-right: 1px solid var(--border);
	}
	[data-testid="stSidebar"] h1 { color: var(--dark-blue); font-size: 1.45rem; }
	[data-testid="stSidebar"] h3 { color: var(--dark-blue); font-size: 1rem; }
	[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: var(--secondary); }
	[data-testid="stSidebar"] [data-testid="stExpander"] {
		border: 1px solid var(--border);
		border-radius: 10px;
		background: rgba(255, 255, 255, 0.7);
	}
	.block-container { max-width: 900px; padding-top: 0.35rem; padding-bottom: 2rem; }
	h1, h2, h3, h4 { color: var(--ink); letter-spacing: 0; }
	.page-header {
		position: relative;
		text-align: center;
		margin: 0 auto 1rem;
		padding: 0.8rem 0 1rem;
		border-bottom: 1px solid var(--border);
	}
	.brand-mark {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 1.65rem;
		height: 1.65rem;
		margin-right: 0.45rem;
		border-radius: 7px;
		background: var(--light-blue);
		color: var(--blue);
		font-size: 1.1rem;
		vertical-align: 0.12rem;
	}
	.page-header h1 { margin: 0; color: var(--dark-blue); font-size: 2rem; font-weight: 700; }
	.page-header p { color: var(--secondary); font-size: 1rem; margin: 0.3rem 0 0; }
	.header-tags { color: var(--blue); font-size: 0.78rem; font-weight: 600; margin-top: 0.45rem; }
	[data-testid="stMainBlockContainer"] > div { gap: 0.65rem; }
	[data-testid="stAlert"] {
		border: 1px solid #bfdbfe;
		border-left: 4px solid var(--blue);
		border-radius: 10px;
		background: var(--light-blue);
		color: var(--dark-blue);
		padding: 0.4rem 0.75rem;
	}
	[data-testid="stAlert"] p { color: var(--dark-blue) !important; font-size: 0.9rem; }
	[data-testid="stVerticalBlockBorderWrapper"] {
		border-color: var(--border);
		border-radius: 12px;
		background: var(--white);
		box-shadow: 0 4px 15px rgba(37, 99, 235, 0.06);
	}
	[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 0.8rem 1rem; }
	[data-testid="stForm"] { background: transparent; border: 0; padding: 0; }
	[data-testid="stTextInput"] label { color: var(--ink) !important; font-weight: 600; }
	[data-testid="stTextInput"] input {
		background: var(--white);
		border-color: #B7C9DB;
		color: var(--ink);
		border-radius: 8px;
		min-height: 2.8rem;
	}
	[data-testid="stTextInput"] input::placeholder { color: var(--secondary); opacity: 1; }
	[data-testid="stTextInput"] input:focus {
		border-color: var(--blue);
		box-shadow: 0 0 0 1px var(--blue);
	}
	.stButton > button, [data-testid="stFormSubmitButton"] > button {
		border-radius: 8px;
		font-weight: 600;
		min-height: 2.55rem;
	}
	.stButton > button { color: var(--dark-blue); border-color: #B7C9DB; background: var(--white); }
	[data-testid="stFormSubmitButton"] > button[kind="primary"] {
		background: var(--blue);
		border-color: var(--blue);
		color: var(--white) !important;
		box-shadow: 0 3px 8px rgba(37, 99, 235, 0.18);
	}
	[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
		background: var(--dark-blue);
		border-color: var(--dark-blue);
	}
	[data-testid="stFileUploaderDropzone"] {
		border: 1px dashed #93B4D5;
		border-radius: 10px;
		background: var(--canvas);
	}
	.pipeline-flow {
		color: var(--secondary);
		font-family: 'IBM Plex Mono', monospace;
		font-size: 0.78rem;
		line-height: 1.75;
	}
	[data-testid="stCaptionContainer"] { color: var(--secondary); }
	#MainMenu, footer { visibility: hidden; }
	[data-testid="stAppDeployButton"] { display: none; }
	.stSpinner > div { border-top-color: var(--blue) !important; }
	@media (max-width: 700px) {
		.block-container { padding: 0.25rem 0.75rem 1.25rem; }
		.page-header h1 { font-size: 1.75rem; }
		[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 0.7rem 0.8rem; }
	}
	</style>
	""",
	unsafe_allow_html=True,
)


def clear_current_result() -> None:
	"""Clear the question and answer before the next Streamlit rerun."""
	st.session_state["question_input"] = ""
	for key in ("submitted_question", "answer_result", "request_error"):
		st.session_state.pop(key, None)


with st.sidebar:
	st.title("MedDoc-RAG")
	st.markdown("### Technology Stack")
	st.markdown(
		"- Python\n"
		"- Streamlit\n"
		"- LangChain\n"
		"- Hugging Face Embeddings\n"
		"- FAISS\n"
		"- Google Gemini"
	)
	with st.expander("How it works", expanded=False):
		st.markdown(
			"<div class='pipeline-flow'>Upload PDF<br>↓ Extract text<br>↓ Create chunks"
			"<br>↓ Generate embeddings<br>↓ FAISS retrieval<br>↓ Gemini<br>↓ Answer</div>",
			unsafe_allow_html=True,
		)

st.markdown(
	"<div class='page-header'><h1><span class='brand-mark'>✚</span>MedDoc-RAG</h1>"
	"<p>Document-Based Healthcare Information Assistant</p>"
	"<div class='header-tags'>Secure&nbsp;&nbsp;•&nbsp;&nbsp;Document-Based&nbsp;&nbsp;•&nbsp;&nbsp;AI Powered</div></div>",
	unsafe_allow_html=True,
)
with st.container(border=True):
	st.markdown("**⚠ Important Information**")
	st.markdown(
		"This tool provides information retrieved from the provided documents. "
		"It is not a medical diagnosis or treatment system."
)

with st.container(border=True):
	st.markdown("### Upload Document")
	st.caption("Upload a healthcare PDF document")
	st.session_state.setdefault("uploader_version", 0)
	uploaded_file = st.file_uploader(
		"Upload a healthcare document (PDF)",
		type=["pdf"],
		key=f"healthcare_pdf_{st.session_state['uploader_version']}",
	)

	if uploaded_file is None:
		clear_active_upload_state()
	else:
		process_uploaded_pdf(uploaded_file)
		if st.session_state.get("upload_error"):
			st.error(st.session_state["upload_error"])
		elif st.session_state.get("uploaded_vector_store") is not None:
			st.success("Document processed successfully.")
			st.markdown(f"**File:** {st.session_state['uploaded_filename']}")
			st.caption(
				f"{st.session_state['uploaded_page_count']} pages • "
				f"{st.session_state['uploaded_chunk_count']} chunks"
			)

	st.button(
		"Clear uploaded document",
		on_click=clear_uploaded_document,
		disabled=uploaded_file is None,
		key="clear_uploaded_document",
		)

with st.container(border=True):
	st.markdown("### Ask a Question")
	with st.form("question_form", clear_on_submit=False):
		question = st.text_input(
			"Ask a question about the documents",
			key="question_input",
			placeholder="What would you like to know from the document?",
		)
		submitted = st.form_submit_button("Get Answer", type="primary", use_container_width=True)

clear_column, _ = st.columns([1, 5])
with clear_column:
	st.button("Clear", on_click=clear_current_result, use_container_width=True)

if submitted:
	if uploaded_file is not None and st.session_state.get("upload_error"):
		st.session_state["request_error"] = "Fix the uploaded document error before asking a question."
		st.session_state.pop("answer_result", None)
	elif not question.strip():
		st.session_state["request_error"] = "Enter a question about the available documents."
		st.session_state.pop("answer_result", None)
	else:
		st.session_state["submitted_question"] = question.strip()
		st.session_state.pop("request_error", None)
		st.session_state.pop("answer_result", None)
		try:
			with st.spinner("Retrieving document passages and preparing an answer..."):
				st.session_state["answer_result"] = answer_question(
					question.strip(),
					vector_store=st.session_state.get("uploaded_vector_store"),
				)
		except FileNotFoundError:
			st.session_state["request_error"] = (
				"The document index is unavailable. Please build the FAISS index before asking a question."
			)
		except RuntimeError as error:
			st.session_state["request_error"] = str(error)
		except ValueError as error:
			st.session_state["request_error"] = str(error)
		except Exception:
			st.session_state["request_error"] = (
				"The question could not be processed. Please try again."
			)

if st.session_state.get("request_error"):
	st.error(st.session_state["request_error"])

result = st.session_state.get("answer_result")
if result:
	with st.container(border=True):
		st.markdown("#### Your question")
		st.write(st.session_state.get("submitted_question", ""))

	with st.container(border=True):
		st.markdown("### AI Answer")
		st.markdown(result["answer"])

	with st.container(border=True):
		st.markdown("### Sources")
		source_lines: list[str] = []
		seen_sources: set[tuple[str, int | str]] = set()
		for reference in result.get("sources", []):
			source_name = reference.get("source", "")
			page_number = reference.get("page")
			if not source_name or page_number is None:
				continue
			source_key = (source_name, page_number)
			if source_key in seen_sources:
				continue
			seen_sources.add(source_key)
			source_lines.append(f"- {source_name} — Page {page_number}")

		if source_lines:
			st.markdown("\n".join(source_lines))
		elif result["answer"] == NO_INFORMATION_ANSWER:
			st.info("No relevant document passages were retrieved for this question.")
		else:
			st.caption("No source references were returned.")
