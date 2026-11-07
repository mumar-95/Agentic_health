"""Supabase client, chat persistence, and hospital document sync."""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client

from utils.document_loader import list_document_files, load_documents
from utils.text_chunking import chunk_documents
from utils.vector_store import compute_documents_fingerprint, default_documents_dir

logger = logging.getLogger(__name__)


def get_supabase_config() -> dict[str, str]:
    load_dotenv()
    url = (os.getenv("SUPABASE_URL") or "").strip()
    # Prefer secret key for server-side writes (bypasses RLS).
    key = (
        os.getenv("SUPABASE_SECRET_KEY")
        or os.getenv("SUPABASE_PUBLISHABLE_KEY")
        or ""
    ).strip()
    jwks_url = (os.getenv("SUPABASE_JWKS_URL") or "").strip()
    return {
        "url": url,
        "key": key,
        "jwks_url": jwks_url,
        "publishable_key": (os.getenv("SUPABASE_PUBLISHABLE_KEY") or "").strip(),
        "secret_key": (os.getenv("SUPABASE_SECRET_KEY") or "").strip(),
    }


def get_supabase_client() -> Client:
    config = get_supabase_config()
    if not config["url"] or not config["key"]:
        raise ValueError(
            "Supabase is not configured. Set SUPABASE_URL and "
            "SUPABASE_SECRET_KEY (or SUPABASE_PUBLISHABLE_KEY) in .env"
        )
    return create_client(config["url"], config["key"])


def create_chat_session(client: Client) -> str:
    response = client.table("chat_sessions").insert({}).execute()
    if not response.data:
        raise RuntimeError("Failed to create chat session in Supabase")
    return str(response.data[0]["id"])


def save_chat_message(
    client: Client,
    session_id: str,
    role: str,
    content: str,
    source: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": session_id,
        "role": role,
        "content": content,
    }
    if source:
        payload["source"] = source

    response = client.table("chat_messages").insert(payload).execute()
    if not response.data:
        raise RuntimeError("Failed to save chat message in Supabase")
    return response.data[0]


def fetch_session_messages(client: Client, session_id: str) -> list[dict[str, Any]]:
    response = (
        client.table("chat_messages")
        .select("id, role, content, source, created_at")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )
    return list(response.data or [])


def ping_supabase(client: Client) -> bool:
    """Lightweight connectivity check against chat_sessions."""
    client.table("chat_sessions").select("id").limit(1).execute()
    return True


def list_hospital_documents(client: Client) -> list[dict[str, Any]]:
    response = (
        client.table("hospital_documents")
        .select(
            "id, file_name, file_type, file_size_bytes, section_count, "
            "chunk_count, indexed_at, updated_at"
        )
        .order("file_name")
        .execute()
    )
    return list(response.data or [])


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_document_sync_payload(
    documents_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Build upsert payloads for each local hospital document.

    Includes file metadata, extracted text, and chunk texts.
    """
    docs_dir = Path(documents_dir) if documents_dir else default_documents_dir()
    folder_fp = compute_documents_fingerprint(docs_dir)
    files = {p.name: p for p in list_document_files(docs_dir)}
    loaded = load_documents(docs_dir)

    by_source: dict[str, list] = {}
    for doc in loaded:
        source = str(doc.metadata.get("source") or "unknown")
        by_source.setdefault(source, []).append(doc)

    chunks = chunk_documents(loaded)
    chunks_by_source: dict[str, list[str]] = {}
    for chunk in chunks:
        source = str(chunk.metadata.get("source") or "unknown")
        chunks_by_source.setdefault(source, []).append(chunk.page_content)

    now = datetime.now(timezone.utc).isoformat()
    payloads: list[dict[str, Any]] = []

    for file_name, path in files.items():
        sections = by_source.get(file_name, [])
        content_text = "\n\n".join(doc.page_content for doc in sections).strip()
        chunk_texts = chunks_by_source.get(file_name, [])
        stat = path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

        payloads.append(
            {
                "file_name": file_name,
                "file_type": path.suffix.lower().lstrip(".") or "unknown",
                "file_size_bytes": int(stat.st_size),
                "file_mtime": mtime,
                "content_text": content_text,
                "section_count": len(sections),
                "chunk_count": len(chunk_texts),
                "content_sha256": _sha256_text(content_text),
                "folder_fingerprint": folder_fp,
                "indexed_at": now,
                "updated_at": now,
                "_chunk_texts": chunk_texts,
            }
        )

    return payloads


def sync_hospital_documents(
    client: Client,
    documents_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Upsert local hospital documents into Supabase and replace their chunks.

    Removes DB rows for files that no longer exist locally.
    """
    payloads = build_document_sync_payload(documents_dir)
    current_names = {item["file_name"] for item in payloads}

    existing = (
        client.table("hospital_documents")
        .select("id, file_name")
        .execute()
        .data
        or []
    )
    existing_by_name = {row["file_name"]: row["id"] for row in existing}

    stale_ids = [
        row["id"]
        for row in existing
        if row["file_name"] not in current_names
    ]
    if stale_ids:
        client.table("hospital_documents").delete().in_("id", stale_ids).execute()

    synced = 0
    for item in payloads:
        chunk_texts: list[str] = item.pop("_chunk_texts", [])
        response = (
            client.table("hospital_documents")
            .upsert(item, on_conflict="file_name")
            .execute()
        )
        if not response.data:
            raise RuntimeError(f"Failed to upsert document {item['file_name']}")

        document_id = response.data[0]["id"]
        # Replace chunks for this document
        client.table("hospital_document_chunks").delete().eq(
            "document_id", document_id
        ).execute()

        if chunk_texts:
            chunk_rows = [
                {
                    "document_id": document_id,
                    "chunk_index": idx,
                    "content": text,
                }
                for idx, text in enumerate(chunk_texts)
            ]
            # Insert in batches to stay within payload limits
            batch_size = 50
            for start in range(0, len(chunk_rows), batch_size):
                batch = chunk_rows[start : start + batch_size]
                client.table("hospital_document_chunks").insert(batch).execute()

        synced += 1
        logger.info(
            "Synced document to Supabase: %s (%d chunks)",
            item["file_name"],
            len(chunk_texts),
        )

    return {
        "synced": synced,
        "removed": len(stale_ids),
        "file_names": sorted(current_names),
        "previous_ids": existing_by_name,
    }
