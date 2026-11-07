"""Persistent ChromaDB store with document fingerprint change detection."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from utils.document_loader import list_document_files, load_documents
from utils.text_chunking import chunk_documents

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "hospital_docs"
DEFAULT_TOP_K = 3
FINGERPRINT_FILENAME = ".docs_fingerprint.json"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_documents_dir() -> Path:
    return project_root() / "documents"


def default_persist_dir() -> Path:
    return project_root() / "vector_db" / "chroma_db"


def compute_documents_fingerprint(documents_dir: str | Path) -> str:
    """Hash of supported files' names, sizes, and mtimes."""
    files = list_document_files(documents_dir)
    hasher = hashlib.sha256()
    for path in files:
        stat = path.stat()
        hasher.update(path.name.encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
    return hasher.hexdigest()


def _fingerprint_path(persist_dir: Path) -> Path:
    return persist_dir / FINGERPRINT_FILENAME


def _read_saved_fingerprint(persist_dir: Path) -> str | None:
    path = _fingerprint_path(persist_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("fingerprint")
    except (json.JSONDecodeError, OSError):
        return None


def _write_fingerprint(persist_dir: Path, fingerprint: str) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    payload = {"fingerprint": fingerprint}
    _fingerprint_path(persist_dir).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _clear_persist_dir(persist_dir: Path) -> None:
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)


def build_or_load_vector_store(
    embeddings: Embeddings,
    documents_dir: str | Path | None = None,
    persist_dir: str | Path | None = None,
    force_rebuild: bool = False,
    collection_name: str = DEFAULT_COLLECTION,
) -> Chroma:
    """
    Create or load a persistent Chroma collection.

    Skips re-indexing when the documents fingerprint is unchanged, unless
    force_rebuild is True.
    """
    docs_dir = Path(documents_dir) if documents_dir else default_documents_dir()
    store_dir = Path(persist_dir) if persist_dir else default_persist_dir()
    current_fp = compute_documents_fingerprint(docs_dir)
    saved_fp = _read_saved_fingerprint(store_dir)
    chroma_ready = (store_dir / "chroma.sqlite3").exists() or any(
        store_dir.glob("**/chroma.sqlite3")
    )

    if (
        not force_rebuild
        and saved_fp == current_fp
        and chroma_ready
        and list(store_dir.iterdir())
    ):
        logger.info("Document fingerprint unchanged; loading existing Chroma store.")
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(store_dir),
        )

    logger.info("Building Chroma index from %s", docs_dir)
    documents = load_documents(docs_dir)
    chunks = chunk_documents(documents)

    _clear_persist_dir(store_dir)

    if not chunks:
        logger.warning(
            "No document chunks to index. Creating an empty Chroma collection."
        )
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(store_dir),
        )
    else:
        store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=str(store_dir),
        )

    _write_fingerprint(store_dir, current_fp)
    logger.info("Indexed %d chunk(s).", len(chunks))
    return store


def similarity_search_with_scores(
    store: Chroma,
    query: str,
    k: int = DEFAULT_TOP_K,
) -> list[tuple[Document, float]]:
    """Retrieve top-k chunks with distance scores (lower = more similar)."""
    try:
        count = store._collection.count()  # noqa: SLF001 — LangChain Chroma API
    except Exception:  # noqa: BLE001
        count = 0

    if count == 0:
        return []

    return store.similarity_search_with_score(query, k=min(k, count))
