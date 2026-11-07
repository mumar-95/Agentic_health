"""Load hospital documents (PDF, DOCX, TXT) from the documents folder."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

_LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}


def list_document_files(documents_dir: str | Path) -> list[Path]:
    """Return supported document files under documents_dir (non-recursive)."""
    root = Path(documents_dir)
    if not root.exists():
        logger.warning("Documents directory does not exist: %s", root)
        return []

    files = sorted(
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return files


def load_documents(documents_dir: str | Path) -> list[Document]:
    """
    Load and extract content from PDF, DOCX, and TXT files.

    Skips unreadable or empty files with a warning.
    """
    documents: list[Document] = []
    files = list_document_files(documents_dir)

    if not files:
        logger.info("No PDF/DOCX/TXT files found in %s", documents_dir)
        return documents

    for path in files:
        loader_cls = _LOADER_MAP.get(path.suffix.lower())
        if loader_cls is None:
            continue

        try:
            if path.suffix.lower() == ".txt":
                loader = loader_cls(str(path), encoding="utf-8")
            else:
                loader = loader_cls(str(path))

            loaded = loader.load()
            usable = []
            for doc in loaded:
                text = (doc.page_content or "").strip()
                if not text:
                    continue
                doc.page_content = text
                doc.metadata = {
                    **(doc.metadata or {}),
                    "source": str(path.name),
                    "file_path": str(path.resolve()),
                }
                usable.append(doc)

            if not usable:
                logger.warning("Skipping empty document: %s", path.name)
                continue

            documents.extend(usable)
            logger.info("Loaded %s (%d page(s)/section(s))", path.name, len(usable))
        except Exception as exc:  # noqa: BLE001 — skip bad files gracefully
            logger.warning("Could not read %s: %s", path.name, exc)

    return documents
