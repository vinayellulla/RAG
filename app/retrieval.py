"""Retrieval: search the Chroma vector store for chunks relevant to a query.

Uses the same embedding model as ingestion (get_embeddings), which is
required: query vectors and chunk vectors must come from the same model.
"""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import (
    BALANCED_RETRIEVAL,
    CHROMA_DIR,
    COLLECTION_NAME,
    PER_SOURCE_K,
    RELEVANCE_THRESHOLD,
    SOURCE_MARGIN,
    TOP_K,
)
from app.ingestion import get_embeddings


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Open the persisted Chroma store once and reuse it (loading the
    embedding model is the slow part, so we only do it a single time)."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"No vector store at {CHROMA_DIR}. Run 'python3 -m app.ingestion' first."
        )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


@lru_cache(maxsize=1)
def list_sources() -> tuple[str, ...]:
    """Distinct filenames stored in the vector store (one per ingested paper)."""
    data = get_vector_store().get(include=["metadatas"])
    names = {m["filename"] for m in data["metadatas"] if m and "filename" in m}
    return tuple(sorted(names))


def _clean(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")
    return query


def search_basic(query: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    """Plain search over all papers: the k closest chunks overall.

    Distance: LOWER means MORE similar (Chroma's default metric is L2 distance).
    """
    return get_vector_store().similarity_search_with_score(_clean(query), k=k)


def search_balanced(query: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    """Search each paper separately so comparison questions reach every paper.

    A question that names two papers becomes one blended vector, and a plain
    search then favours whichever paper matches more strongly. Here we:
      1. take the PER_SOURCE_K closest chunks from EACH paper (metadata filter),
      2. keep a chunk only if it is within SOURCE_MARGIN of the best chunk
         overall and under RELEVANCE_THRESHOLD,
      3. if two or more papers survive, return them merged (best first, at most
         2*k chunks); otherwise it is a single-topic question, so fall back to
         the plain search and behave exactly as before.
    """
    query = _clean(query)
    store = get_vector_store()

    per_source = []
    for source in list_sources():
        hits = store.similarity_search_with_score(
            query, k=PER_SOURCE_K, filter={"filename": source}
        )
        per_source.extend(hits)
    if not per_source:
        return []

    best = min(dist for _, dist in per_source)
    cutoff = min(best + SOURCE_MARGIN, RELEVANCE_THRESHOLD)
    kept = [(doc, dist) for doc, dist in per_source if dist <= cutoff]

    papers_present = {doc.metadata.get("filename") for doc, _ in kept}
    if len(papers_present) < 2:
        return search_basic(query, k=k)

    kept.sort(key=lambda pair: pair[1])
    return kept[: 2 * k]


def search(query: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    """Main entry point used by the RAG chain and the evaluation script.

    Set BALANCED_RETRIEVAL=0 in the environment to use the plain search, which
    lets you compare both methods on the same questions.
    """
    if BALANCED_RETRIEVAL:
        return search_balanced(query, k=k)
    return search_basic(query, k=k)
