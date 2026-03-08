"""
rag/create_embeddings.py
Loads math docs, chunks them, creates embeddings and persists to Chroma.
Run once (or whenever docs are updated):
    python rag/create_embeddings.py
"""
from __future__ import annotations
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import MATH_DOCS_DIR, CHROMA_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, GEMINI_API_KEY


def build_vector_store(force_rebuild: bool = False):
    """
    Build (or rebuild) the Chroma vector store from math docs.

    Returns the Chroma vectorstore object.
    """
    import chromadb
    from langchain_community.document_loaders import TextLoader, DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # Choose embedding backend
    embeddings = _get_embeddings()

    # If already built and not forced, just load
    if os.path.exists(CHROMA_DB_DIR) and not force_rebuild:
        print(f"[RAG] Vector store already exists at {CHROMA_DB_DIR}. Loading...")
        return _load_existing_store(embeddings)

    print(f"[RAG] Loading documents from {MATH_DOCS_DIR} ...")
    loader = DirectoryLoader(
        MATH_DOCS_DIR,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"[RAG] Loaded {len(docs)} document(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[RAG] Split into {len(chunks)} chunks.")

    # Build Chroma store
    from langchain_community.vectorstores import Chroma
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    print(f"[RAG] Vector store persisted to {CHROMA_DB_DIR}")
    return vectorstore


def _load_existing_store(embeddings):
    from langchain_community.vectorstores import Chroma
    return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)


def _get_embeddings():
    """Return Gemini embeddings or fall back to sentence-transformers."""
    if GEMINI_API_KEY:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            print("[RAG] Using Gemini embeddings.")
            return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        except Exception as e:
            print(f"[RAG] Gemini embeddings failed ({e}), falling back to local.")

    # Free local fallback
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("[RAG] Using HuggingFace sentence-transformers embeddings (free).")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        raise RuntimeError(
            f"No embedding backend available. "
            f"Set GEMINI_API_KEY or install sentence-transformers: "
            f"pip install sentence-transformers\nError: {e}"
        )


if __name__ == "__main__":
    build_vector_store(force_rebuild=True)
    print("[RAG] Done.")
