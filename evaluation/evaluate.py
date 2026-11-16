import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, RELEVANCE_THRESHOLD, TOP_K
from app.retrieval import search

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "questions.json"
RESULTS_FILE = EVAL_DIR / "results.json"


def matches(filename: str, keywords: list[str]) -> bool:
    """True if the filename contains any keyword (case-insensitive)."""
    name = filename.lower()
    return any(keyword.lower() in name for keyword in keywords)


def evaluate_retrieval(item: dict, k : int)->dict:
    results = search(item["question"], k=k)
    filenames = [doc.metadata.get("filename","") for doc, _ in results]
    distances = [round(dist,3) for _, dist in results]
    best = min(distances) if distances else None


    row = {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "top_files": filenames,
        "distances": distances,
        "best_distance": best,
    }

    if item["type"] == "off_topic":
         # Correct = nothing passes the relevance gate, so the LLM is never called.

         row["correct"] = not any(d <= RELEVANCE_THRESHOLD for d in distances)

         return row 

    keywords = item["expected_sources"]
    row["rank"] = next(
        (i for i, name in enumerate(filenames, start=1) if matches(name, keywords)),
        None,
    )
    row["gate_ok"] = best is not None and best <= RELEVANCE_THRESHOLD


    if item["type"] == "comparison":
        found = [kw for kw in keywords if any(matches(n, [kw]) for n in filenames)]
        row["sources_found"] = f"{len(found)}/{len(keywords)}"
        row["correct"] = len(found) == len(keywords)
    else:
        row["correct"] = row["rank"] is not None

    return row


def add_llm_result(row: dict, item: dict, k: int) -> dict:
    """Run the full RAG chain and record whether the grounded flag is right."""
    from app.rag_chain import answer_question  # imported here so retrieval-only runs need no API key

    start = time.perf_counter()
    try:
        result = answer_question(item["question"], k=k)
    except Exception as error:  # e.g. rate limit (429) or network problem
        row["llm_error"] = f"{type(error).__name__}: {error}"
        return
    row["latency_s"] = round(time.perf_counter() - start, 2)
    row["grounded"] = result["grounded"]
    row["llm_correct"] = result["grounded"] == (item["type"] != "off_topic")
    row["answer"] = result["answer"]

    return row


def summarize(rows: list[dict]) -> dict:
    single = [r for r in rows if r["type"] == "single"]
    comparison = [r for r in rows if r["type"] == "comparison"]
    off_topic = [r for r in rows if r["type"] == "off_topic"]
    answerable = single + comparison

    def pct(count: int, total: int) -> str:
        return f"{count}/{total} ({100 * count / total:.0f}%)" if total else "n/a"

    summary = {
        "hit@k (single)": pct(sum(r["correct"] for r in single), len(single)),
        "top-1 (single)": pct(sum(r["rank"] == 1 for r in single), len(single)),
        "MRR (single)": round(
            sum(1 / r["rank"] for r in single if r["rank"]) / len(single), 3
        )
        if single
        else "n/a",
        "coverage (comparison)": pct(sum(r["correct"] for r in comparison), len(comparison)),
        "rejection (off-topic)": pct(sum(r["correct"] for r in off_topic), len(off_topic)),
        "wrongly refused (answerable)": pct(
            sum(not r["gate_ok"] for r in answerable), len(answerable)
        ),
    }

    llm_rows = [r for r in rows if "llm_correct" in r]
    if llm_rows:
        summary["llm grounded flag correct"] = pct(
            sum(r["llm_correct"] for r in llm_rows), len(llm_rows)
        )
        summary["avg latency (s)"] = round(
            sum(r["latency_s"] for r in llm_rows) / len(llm_rows), 2
        )
    errors = [r for r in rows if "llm_error" in r]
    if errors:
        summary["llm errors"] = len(errors)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--llm", action="store_true", help="also call the LLM for each question")
    parser.add_argument("-k", type=int, default=TOP_K, help="chunks retrieved per question")
    parser.add_argument("--delay", type=float, default=4.0,
                        help="seconds to wait between LLM calls (protects free-tier limits)")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    rows = []

    print(f"Evaluating {len(questions)} questions "
          f"(k={args.k}, chunk={CHUNK_SIZE}/{CHUNK_OVERLAP}, threshold={RELEVANCE_THRESHOLD})\n")

    for item in questions:
        row = evaluate_retrieval(item, args.k)
        if args.llm:
            # Off-topic questions rejected by the gate never reach the LLM, so
            # they cost no API quota; only pause after real LLM calls.
            add_llm_result(row, item, args.k)
            if row.get("best_distance") is not None and row["best_distance"] <= RELEVANCE_THRESHOLD:
                time.sleep(args.delay)
        rows.append(row)

        mark = "PASS" if row["correct"] else "FAIL"
        detail = (
            f"rank={row['rank']}" if item["type"] == "single"
            else f"found={row['sources_found']}" if item["type"] == "comparison"
            else "rejected" if row["correct"] else "WRONGLY ACCEPTED"
        )
        print(f"{mark}  [{row['id']:<7}] {item['type']:<10} {detail:<18} best_dist={row['best_distance']}")

    summary = summarize(rows)
    print("\n=== Summary ===")
    for name, value in summary.items():
        print(f"{name:<30} {value}")

    RESULTS_FILE.write_text(
        json.dumps(
            {
                "run_at": datetime.now().isoformat(timespec="seconds"),
                "config": {
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "top_k": args.k,
                    "relevance_threshold": RELEVANCE_THRESHOLD,
                },
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved details to {RESULTS_FILE}")


if __name__ == "__main__":
    main()


