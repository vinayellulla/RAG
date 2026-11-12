import re
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


from app.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RAW_DATA_DIR,
)

def clean_text(text: str) -> str:
    """Light cleanup that keeps commands and code blocks readable.
 
    We only remove things that are pure noise. Aggressive cleaning can destroy
    the exact commands (kubectl ..., nginx -t) your assistant needs to cite.
    """
    text = text.replace("\x00", "")  # null bytes break some vector stores
    text = re.sub(r"[ \t]+", " ", text)  # collapse runs of spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse 3+ blank lines into 1
    return text.strip()



def load_documents() -> list[Document]:
    """Load every PDF in data/raw/, one Document per page."""
    pdf_paths = sorted(RAW_DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {RAW_DATA_DIR}. Put your NGINX and "
            "Kubernetes PDFs there and run again."
        )
 
    documents: list[Document] = []
    for pdf_path in pdf_paths:
        pages = PyPDFLoader(str(pdf_path)).load()
        kept = 0
        for page in pages:
            content = clean_text(page.page_content)
            if not content:  # skip blank or image-only pages
                continue
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "filename": pdf_path.name,
                        "page": page.metadata.get("page", 0) + 1,  # 1-based for humans
                    },
                )
            )
            kept += 1
        print(f"  {pdf_path.name}: {len(pages)} pages, {kept} with text")
 
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """This function will split pages into overlapping chunks with metadata is copied to every chunk"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""], # prefer paragraph , the line for breaks

    )

    return splitter.split_documents(documents)



def get_embeddings() -> HuggingFaceEmbeddings:
    """Embedding model, shared with retrieval so both use the exact same one."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    

def build_vector_store(chunks: list[Document]) -> Chroma:
    """Embed the chunks and persist them in Chroma (rebuilding from scratch)."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
 
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )


def main() -> None:
    print(f"Loading data form the {RAW_DATA_DIR}")
    documents = load_documents()
    print(f"loaded {len(documents)} pages with text. \n")

    print(f"Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    chunks=split_documents(documents)
    print(f"Created {len(chunks)} chunks.\n")

    # Manual inspection: check that commands and code blocks survived extraction.

    sample = chunks[len(chunks) // 2 ]
    print("-------Sample Chunk---------------")

    print(f"Source : {sample.metadata['filename']}  (page {sample.metadata['page']})")

    print(sample.page_content)

    print("----------------\n")

    print(f"Embeddings with '{EMBEDDING_MODEL}' and storing in Chroma.....")

    build_vector_store(chunks)

    print(f"Done. Vector Store saved to  {CHROMA_DIR}")



if __name__ == "__main__":
    main()

