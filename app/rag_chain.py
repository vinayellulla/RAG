"""RAG chain: retrieve chunks -> build a grounded prompt -> ask the LLM -> return
the answer together with the sources it was based on.

Run a quick check from the project root:
    python3 -m app.rag_chain "What is multi-head attention?"
"""

import sys
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    RELEVANCE_THRESHOLD,
    TOP_K,
)
from app.retrieval import search

load_dotenv()  # makes GROQ_API_KEY available

NO_INFO_MESSAGE = (
    "I couldn't find enough information in the research papers to answer this reliably."
)

SYSTEM_PROMPT = f"""You are a research paper assistant. You help students understand \
deep learning papers (such as "Attention Is All You Need", BERT, Generative Adversarial \
Nets, and "An Image is Worth 16x16 Words") using ONLY the numbered context passages \
provided.

Rules:
1. Use only the context. Do not use outside knowledge, and do not invent facts, \
numbers, equations, results or hyperparameters that are not in the context.
2. Explain clearly and simply, as if teaching a motivated beginner, but keep the \
paper's own terminology. Use short paragraphs or bullet points.
3. After each claim, cite the passage number in square brackets, like [1] or [2][3].
4. Always say which paper a point comes from (the source name is shown with each \
passage). When the question compares papers, keep each paper's points separate.
5. If the passages do not contain enough information to answer, reply with exactly: \
"{NO_INFO_MESSAGE}" and nothing else.
6. If the passages only partly cover the question, answer the covered part and say \
what is missing.
7. Equations and tables in the passages may be garbled by PDF extraction. If a passage \
looks corrupted, describe the idea in words and say the formula is unclear instead of \
guessing.
"""

USER_PROMPT = """Context passages:
{context}

Question: {question}"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
)


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Create the LLM client once and reuse it."""
    return ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=2,
    )


def format_context(results) -> str:
    """Number each chunk so the model can cite it as [1], [2], ..."""
    blocks = []
    for number, (doc, _distance) in enumerate(results, start=1):
        source = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[{number}] (source: {source}, page {page})\n{doc.page_content}")
    return "\n\n".join(blocks)


def answer_question(question: str, k: int = TOP_K) -> dict:
    """Return {"answer": str, "sources": [...], "grounded": bool}.

    grounded is False when the knowledge base could not support an answer.
    """
    results = search(question, k=k)

    # Lower distance = more similar. Drop chunks that are too far away.
    relevant = [(doc, dist) for doc, dist in results if dist <= RELEVANCE_THRESHOLD]
    if not relevant:
        return {"answer": NO_INFO_MESSAGE, "sources": [], "grounded": False}

    messages = PROMPT.format_messages(
        context=format_context(relevant), question=question
    )
    response = get_llm().invoke(messages)
    text = (response.content or "").strip()

    # The model itself can decide the context is not enough.
    if text.startswith(NO_INFO_MESSAGE):
        return {"answer": NO_INFO_MESSAGE, "sources": [], "grounded": False}

    sources = [
        {
            "id": number,
            "filename": doc.metadata.get("filename", "unknown"),
            "page": doc.metadata.get("page"),
            "distance": round(dist, 3),
        }
        for number, (doc, dist) in enumerate(relevant, start=1)
    ]
    return {"answer": text, "sources": sources, "grounded": True}


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python3 -m app.rag_chain "your question"')
    question = " ".join(sys.argv[1:])
    result = answer_question(question)

    print(f"\nQuestion: {question}\n")
    print(result["answer"])
    if result["sources"]:
        print("\nSources:")
        for s in result["sources"]:
            print(f"  [{s['id']}] {s['filename']} (page {s['page']})  distance={s['distance']}")
    print(f"\ngrounded: {result['grounded']}")


if __name__ == "__main__":
    main()