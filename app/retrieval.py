"""Retrieval: search the Chroma vector store for chunks relevant to a query.
 
Uses the same embedding model as ingestion (get_embeddings), which is
required: query vectors and chunk vectors must come from the same model.
"""

from functools import lru_cache
from langchain_chroma import Chroma 
from langchain_core.documents import Document
from app.config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from app.ingestion import get_embeddings


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Open the persisted Chroma store once and reuse it (loading the
    embedding model is the slow part, so we only do it a single time)."""

    if not CHROMA_DIR.exists():
      raise FileExistsError(
        "No vector store at {CHROMA_DIR}. Run 'python3 -m app.ingestion' first"
    )

    return Chroma(
       collection_name = COLLECTION_NAME,
       embedding_function = get_embeddings(),
       persist_directory=str(CHROMA_DIR)
    )


def search(query: str, k: int=TOP_K) -> list[tuple[Document,float]]:

   """Return the top-k chunks for a query as (chunk, distance) pairs.
 
    Distance: LOWER means MORE similar (Chroma's default metric is L2 distance).
    """
   
   query = query.strip()
   if not query:
    raise ValueError("Query must not be empty.")
   
   return get_vector_store().similarity_search_with_score(query, k=k)


    # python3 -m scripts.query_retrieval "What is multi-head attention?"
    # python3 -m scripts.query_retrieval "How does BERT do masking?" -k 3