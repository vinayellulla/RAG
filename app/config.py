import os
from pathlib import Path 

try:
    from dotenv import load_dotenv


    load_dotenv
except ImportError:
    pass


BASE_DIR = Path(__file__).resolve().parent.parent
 
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"  # your NGINX + Kubernetes PDFs
CHROMA_DIR = DATA_DIR / "chroma_db"  # persisted vector store (gitignored)


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))  # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))  # shared characters between chunks


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "incident_docs")


TOP_K = int(os.getenv("TOP_K", 5))  # chunks returned per query



RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", 1.2))  # max distance to count as relevant
BALANCED_RETRIEVAL = os.getenv("BALANCED_RETRIEVAL", "1") == "1"  # 1 = per-paper search, 0 = old plain search
PER_SOURCE_K = int(os.getenv("PER_SOURCE_K", 3))  # chunks pulled from each paper
SOURCE_MARGIN = float(os.getenv("SOURCE_MARGIN", 0.3))  # how close to the best match a chunk must be


# BALANCED_RETRIEVAL = os.getenv("BALANCED_RETRIEVAL", "1") == "1"  # 1 = per-paper search, 0 = old plain search
# PER_SOURCE_K = int(os.getenv("PER_SOURCE_K", 3))        # chunks pulled from each paper
# SOURCE_MARGIN = float(os.getenv("SOURCE_MARGIN", 0.3))  # how close to the best match a chunk must be

# RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", 1.2))  # max distance to count as relevant


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))  # 0 = most factual
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 60))

if CHUNK_OVERLAP>CHUNK_SIZE:
    raise ValueError(
        f"chunk overlap {CHUNK_OVERLAP} is greater than {CHUNK_SIZE} it should be less than that"
    )



# print("Hello Error in the east ")
