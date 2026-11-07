"""Retrieval, relevance check, and Groq LLM fallback for the assistant."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.prompt import (
    GENERAL_USER_TEMPLATE,
    GROUNDED_USER_TEMPLATE,
    HOSPITAL_UNVERIFIED_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    SYSTEM_PROMPT,
)
from utils.vector_store import similarity_search_with_scores

logger = logging.getLogger(__name__)

# Chroma with normalized embeddings typically returns cosine distance in [0, 2].
# Lower distance = more similar. Tune with representative questions.
DEFAULT_RELEVANCE_DISTANCE_THRESHOLD = 0.55
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

AnswerSource = Literal[
    "hospital_docs",
    "general_knowledge",
    "hospital_unverified",
    "out_of_scope",
]

HOSPITAL_SPECIFIC_PATTERNS = [
    r"\bthis hospital\b",
    r"\bour hospital\b",
    r"\byour hospital\b",
    r"\bhospital (provide|offers|has|available)\b",
    r"\b(oncology|cancer) (department|ward|clinic|center|centre)\b",
    r"\b(head|director|chief|doctor|dr\.?)\b.+\b(oncolog|department)\b",
    r"\b(who is|contact|phone|address|appointment|booking)\b",
    r"\b(facility|facilities|service|services)\b.+\b(hospital|oncolog|cancer)\b",
    r"\bscreening (facilities|services|available)\b",
]

OUT_OF_SCOPE_PATTERNS = [
    r"\bweather\b",
    r"\btemperature\b",
    r"\bfootball\b",
    r"\bcricket\b",
    r"\bstock market\b",
    r"\bjoke\b",
    r"\brecipe\b",
    r"\bmovie\b",
]


@dataclass
class RAGResult:
    answer: str
    source: AnswerSource
    relevant_chunks: list[Document]
    distances: list[float]


def get_groq_client() -> Groq:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY was not found in the .env file")
    return Groq(api_key=api_key)


def get_groq_model() -> str:
    load_dotenv()
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def is_hospital_specific(question: str) -> bool:
    return _matches_any(question, HOSPITAL_SPECIFIC_PATTERNS)


def is_out_of_scope(question: str) -> bool:
    return _matches_any(question, OUT_OF_SCOPE_PATTERNS)


def filter_relevant_chunks(
    results: list[tuple[Document, float]],
    threshold: float = DEFAULT_RELEVANCE_DISTANCE_THRESHOLD,
) -> tuple[list[Document], list[float]]:
    """Keep chunks whose distance is at or below the relevance threshold."""
    relevant_docs: list[Document] = []
    distances: list[float] = []
    for doc, distance in results:
        if distance <= threshold:
            relevant_docs.append(doc)
            distances.append(float(distance))
    return relevant_docs, distances


def _format_context(chunks: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(chunks, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(parts)


def call_groq(
    client: Groq,
    user_content: str,
    model: str | None = None,
) -> str:
    response = client.chat.completions.create(
        model=model or get_groq_model(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()


def answer_question(
    question: str,
    vector_store: Chroma,
    client: Groq | None = None,
    relevance_threshold: float = DEFAULT_RELEVANCE_DISTANCE_THRESHOLD,
    top_k: int = 3,
) -> RAGResult:
    """
    Search hospital documents, assess relevance, then choose an answer path.

    Paths:
    - Relevant hospital context -> Groq grounded on context
    - Not relevant + general educational -> Groq general-knowledge fallback
    - Hospital-specific without support -> cannot verify message
    - Out of scope -> brief refusal
    """
    q = (question or "").strip()
    if not q:
        return RAGResult(
            answer="Please enter a question about cancer awareness or hospital information.",
            source="out_of_scope",
            relevant_chunks=[],
            distances=[],
        )

    if is_out_of_scope(q):
        return RAGResult(
            answer=OUT_OF_SCOPE_MESSAGE,
            source="out_of_scope",
            relevant_chunks=[],
            distances=[],
        )

    results = similarity_search_with_scores(vector_store, q, k=top_k)
    relevant_chunks, distances = filter_relevant_chunks(results, relevance_threshold)

    groq_client = client or get_groq_client()

    if relevant_chunks:
        user_prompt = GROUNDED_USER_TEMPLATE.format(
            context=_format_context(relevant_chunks),
            question=q,
        )
        answer = call_groq(groq_client, user_prompt)
        return RAGResult(
            answer=answer,
            source="hospital_docs",
            relevant_chunks=relevant_chunks,
            distances=distances,
        )

    if is_hospital_specific(q):
        return RAGResult(
            answer=HOSPITAL_UNVERIFIED_MESSAGE,
            source="hospital_unverified",
            relevant_chunks=[],
            distances=[float(d) for _, d in results],
        )

    user_prompt = GENERAL_USER_TEMPLATE.format(question=q)
    answer = call_groq(groq_client, user_prompt)
    return RAGResult(
        answer=answer,
        source="general_knowledge",
        relevant_chunks=[],
        distances=[float(d) for _, d in results],
    )
