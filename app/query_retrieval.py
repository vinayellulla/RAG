"""Command-line retrieval check: type a question, see the top matching chunks"""

import argparse

from app.config import TOP_K
from app.retrieval import search

PREVIEW_CHARS = 500

def main() -> None:
    parser = argparse.ArgumentParser(description="Search the knowledge base")
    parser.add_argument("query", nargs="+", help="Your question")
    parser.add_argument("-k", type=int, default=TOP_K, help="Number of chunks to return")

    args=parser.parse_args()
    query=" ".join(args.query)
    results= search(query,k=args.k)

    print(f"\nQuery: {query}\n")

    for rank, (doc, distance) in enumerate(results,start=1):
        source = doc.metadata.get("filename","unkown")
        page = doc.metadata.get("page","?")
        text=doc.page_content.replace("\n"," ")
        if len(text)>PREVIEW_CHARS:
            text=text[:PREVIEW_CHARS] + "..."

        print(f"[{rank}] {source} (page{page}) distance={distance:.3f}")
        print(f" {text}\n")


if __name__ == "__main__":
    main()
