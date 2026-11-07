"""Streamlit chatbot for the Cancer Healthcare Assistant."""

from __future__ import annotations

import logging

import streamlit as st
from groq import Groq

from utils.document_loader import list_document_files
from utils.embeddings import get_embeddings
from utils.rag_pipeline import answer_question, get_groq_client
from utils.supabase_client import (
    create_chat_session,
    get_supabase_client,
    list_hospital_documents,
    ping_supabase,
    save_chat_message,
    sync_hospital_documents,
)
from utils.vector_store import (
    build_or_load_vector_store,
    compute_documents_fingerprint,
    default_documents_dir,
    default_persist_dir,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SOURCE_LABELS = {
    "hospital_docs": "Hospital documents",
    "general_knowledge": "General educational knowledge",
    "hospital_unverified": "Not verified in hospital documents",
    "out_of_scope": "Out of scope",
}


@st.cache_resource(show_spinner="Loading embedding model…")
def load_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner="Connecting to Groq…")
def load_groq_client() -> Groq:
    return get_groq_client()


@st.cache_resource(show_spinner="Connecting to Supabase…")
def load_supabase():
    return get_supabase_client()


@st.cache_resource(show_spinner="Preparing document index…")
def load_vector_store(fingerprint: str, force_rebuild: bool = False):
    """Cache keyed by document fingerprint so updates trigger a rebuild."""
    embeddings = load_embeddings()
    return build_or_load_vector_store(
        embeddings=embeddings,
        force_rebuild=force_rebuild,
    )


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "force_rebuild" not in st.session_state:
        st.session_state.force_rebuild = False
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = None
    if "supabase_ok" not in st.session_state:
        st.session_state.supabase_ok = False
    if "docs_synced_fingerprint" not in st.session_state:
        st.session_state.docs_synced_fingerprint = None
    if "docs_sync_result" not in st.session_state:
        st.session_state.docs_sync_result = None


def ensure_chat_session(supabase) -> str | None:
    """Create a Supabase chat session once per browser session."""
    if st.session_state.chat_session_id:
        return st.session_state.chat_session_id
    try:
        session_id = create_chat_session(supabase)
        st.session_state.chat_session_id = session_id
        return session_id
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not create Supabase chat session")
        st.session_state.supabase_ok = False
        st.warning(f"Could not create database session: {exc}")
        return None


def persist_message(
    supabase,
    role: str,
    content: str,
    source: str | None = None,
) -> None:
    session_id = ensure_chat_session(supabase)
    if not session_id:
        return
    try:
        save_chat_message(
            client=supabase,
            session_id=session_id,
            role=role,
            content=content,
            source=source,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not save message to Supabase")
        st.session_state.supabase_ok = False
        st.warning(f"Message was not saved to the database: {exc}")


def ensure_documents_synced(supabase, fingerprint: str, force: bool = False) -> None:
    """Upsert hospital document details into Supabase when files change."""
    if (
        not force
        and st.session_state.docs_synced_fingerprint == fingerprint
    ):
        return
    try:
        with st.spinner("Syncing documents to Supabase…"):
            result = sync_hospital_documents(supabase)
        st.session_state.docs_synced_fingerprint = fingerprint
        st.session_state.docs_sync_result = result
        logger.info("Document sync result: %s", result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not sync documents to Supabase")
        st.warning(
            f"Document details were not saved to Supabase: {exc}. "
            "If tables are missing, run `supabase/add_documents_tables.sql` "
            "(or the full `supabase/schema.sql`) in the SQL Editor."
        )


def main() -> None:
    st.set_page_config(
        page_title="Cancer Healthcare Assistant",
        page_icon="🩺",
        layout="centered",
    )
    init_session_state()

    st.title("Cancer Healthcare Assistant")
    st.caption("General cancer awareness and hospital-document Q&A")

    st.info(
        "**Educational tool only.** This assistant does not diagnose cancer, "
        "prescribe medication, or replace professional medical advice. "
        "If you have concerning symptoms, please consult a qualified healthcare professional."
    )

    docs_dir = default_documents_dir()
    doc_files = list_document_files(docs_dir)
    fingerprint = compute_documents_fingerprint(docs_dir)

    supabase = None
    try:
        supabase = load_supabase()
        ping_supabase(supabase)
        st.session_state.supabase_ok = True
    except Exception as exc:  # noqa: BLE001
        st.session_state.supabase_ok = False
        logger.exception("Supabase unavailable")
        st.warning(
            f"Supabase is not ready: {exc}. "
            "Chat still works locally. Run `supabase/schema.sql` in the "
            "Supabase SQL Editor if tables are missing."
        )

    force = bool(st.session_state.force_rebuild)

    with st.sidebar:
        st.header("Documents")
        st.write(f"Folder: `{docs_dir}`")
        if doc_files:
            st.success(f"{len(doc_files)} document(s) ready to index")
            for path in doc_files:
                st.text(f"• {path.name}")
        else:
            st.warning(
                "No PDF/DOCX/TXT files found. Add hospital documents to the "
                "`documents/` folder, then rebuild the index."
            )

        st.divider()
        st.header("Database")
        if st.session_state.supabase_ok:
            st.success("Supabase connected")
            if st.session_state.chat_session_id:
                st.caption(f"Session: `{st.session_state.chat_session_id}`")
            sync_result = st.session_state.docs_sync_result
            if sync_result is not None:
                st.caption(
                    f"Documents synced: {sync_result.get('synced', 0)} "
                    f"(removed {sync_result.get('removed', 0)})"
                )
            if st.button("Sync documents now", use_container_width=True):
                if supabase is not None:
                    ensure_documents_synced(supabase, fingerprint, force=True)
                    st.rerun()
            try:
                if supabase is not None:
                    db_docs = list_hospital_documents(supabase)
                    if db_docs:
                        with st.expander("Documents in Supabase"):
                            for row in db_docs:
                                st.text(
                                    f"• {row['file_name']} "
                                    f"({row.get('chunk_count', 0)} chunks)"
                                )
            except Exception:  # noqa: BLE001
                st.caption("hospital_documents table not available yet")
        else:
            st.error("Supabase not connected")

        st.divider()
        if st.button("Rebuild index", use_container_width=True):
            st.session_state.force_rebuild = True
            st.session_state.docs_synced_fingerprint = None
            load_vector_store.clear()
            st.rerun()

        st.caption(
            "The index rebuilds automatically when files in `documents/` change. "
            "Use **Rebuild index** after replacing files with the same name/size."
        )
        st.caption(f"Persist path: `{default_persist_dir()}`")

    try:
        vector_store = load_vector_store(fingerprint, force_rebuild=force)
        st.session_state.force_rebuild = False
        client = load_groq_client()
    except ValueError as exc:
        st.error(str(exc))
        st.markdown(
            "Create a `.env` file from `.env.example` and set your `GROQ_API_KEY`, "
            "then restart the app."
        )
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize resources")
        st.error(f"Could not start the assistant: {exc}")
        return

    if supabase is not None and st.session_state.supabase_ok:
        ensure_documents_synced(supabase, fingerprint, force=force)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("source"):
                st.caption(
                    f"Source: {SOURCE_LABELS.get(message['source'], message['source'])}"
                )

    prompt = st.chat_input("Ask about cancer awareness or hospital information…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if supabase is not None and st.session_state.supabase_ok:
        persist_message(supabase, role="user", content=prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = answer_question(
                question=prompt,
                vector_store=vector_store,
                client=client,
            )
        st.markdown(result.answer)
        st.caption(f"Source: {SOURCE_LABELS.get(result.source, result.source)}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "source": result.source,
        }
    )

    if supabase is not None and st.session_state.supabase_ok:
        persist_message(
            supabase,
            role="assistant",
            content=result.answer,
            source=result.source,
        )


if __name__ == "__main__":
    main()
