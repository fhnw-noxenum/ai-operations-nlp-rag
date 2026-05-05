import argparse

from langchain_core.documents import Document

from vector_store import get_vector_store


def retrieve(query: str, k: int = 4) -> list[tuple[Document, float]]:
    return get_vector_store().similarity_search_with_score(query, k=k)


def format_result(document: Document, score: float) -> str:
    source = document.metadata.get("source", "unknown")
    chunk = document.metadata.get("chunk", "?")
    content = " ".join(document.page_content.split())
    return f"[{score:.4f}] {source}#{chunk}\n{content}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    for document, score in retrieve(args.query, k=args.k):
        print(format_result(document, score))
        print()
