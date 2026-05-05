import argparse
from urllib.parse import urlsplit, urlunsplit

from langchain_core.documents import Document
from psycopg import connect
from psycopg.rows import dict_row

from settings import COLLECTION_NAME, database_url
from vector_store import get_vector_store


SEARCH_TYPES = ("hybrid", "vector", "keyword")
VECTOR_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3


def psycopg_database_url() -> str:
    parsed = urlsplit(database_url())
    scheme = parsed.scheme.split("+", 1)[0]
    return urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment))


def retrieval_key(document: Document) -> tuple[str, int | str, str]:
    return (
        document.metadata.get("source", ""),
        document.metadata.get("chunk", ""),
        document.page_content,
    )


def folder_filter(source_folder: str | None) -> dict | None:
    if not source_folder:
        return None

    # PGVector stores metadata as JSONB; this exact metadata filter powers folder selection.
    return {"source_folder": source_folder}


def relevance_from_vector_distance(distance: float) -> float:
    return 1 / (1 + max(distance, 0))


def with_retrieval_metadata(
    document: Document,
    score: float,
    method: str,
    vector_distance: float | None = None,
    keyword_score: float | None = None,
) -> Document:
    metadata = {
        **document.metadata,
        "retrieved_score": score,
        "retrieval_method": method,
    }
    if vector_distance is not None:
        metadata["vector_distance"] = vector_distance
    if keyword_score is not None:
        metadata["keyword_score"] = keyword_score

    return Document(page_content=document.page_content, metadata=metadata)


def vector_retrieve(
    query: str,
    k: int,
    source_folder: str | None = None,
) -> list[tuple[Document, float]]:
    results = get_vector_store().similarity_search_with_score(
        query,
        k=k,
        filter=folder_filter(source_folder),
    )

    scored_results = []
    for document, distance in results:
        relevance = relevance_from_vector_distance(distance)
        scored_results.append(
            (
                with_retrieval_metadata(
                    document,
                    relevance,
                    "vector",
                    vector_distance=distance,
                ),
                relevance,
            )
        )

    return scored_results


def keyword_retrieve(
    query: str,
    k: int,
    source_folder: str | None = None,
) -> list[tuple[Document, float]]:
    params = {
        "collection_name": COLLECTION_NAME,
        "query": query,
        "source_folder": source_folder or "",
        "limit": k,
    }
    folder_clause = ""
    if source_folder:
        folder_clause = "AND e.cmetadata->>'source_folder' = %(source_folder)s"

    # Full-text keyword search complements embeddings for exact names and commands.
    sql = f"""
        SELECT
            e.document,
            e.cmetadata,
            ts_rank_cd(
                to_tsvector('simple', coalesce(e.document, '')),
                plainto_tsquery('simple', %(query)s)
            ) AS keyword_score
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %(collection_name)s
          AND to_tsvector('simple', coalesce(e.document, ''))
              @@ plainto_tsquery('simple', %(query)s)
          {folder_clause}
        ORDER BY keyword_score DESC
        LIMIT %(limit)s
    """

    with connect(psycopg_database_url(), row_factory=dict_row) as connection:
        rows = connection.execute(sql, params).fetchall()

    if not rows:
        return []

    max_score = max(float(row["keyword_score"]) for row in rows) or 1
    results = []
    for row in rows:
        keyword_score = float(row["keyword_score"])
        score = keyword_score / max_score
        document = Document(
            page_content=row["document"],
            metadata=row["cmetadata"] or {},
        )
        results.append(
            (
                with_retrieval_metadata(
                    document,
                    score,
                    "keyword",
                    keyword_score=keyword_score,
                ),
                score,
            )
        )

    return results


def hybrid_retrieve(
    query: str,
    k: int,
    source_folder: str | None = None,
) -> list[tuple[Document, float]]:
    vector_results = vector_retrieve(query, max(k * 4, 20), source_folder=source_folder)
    keyword_results = keyword_retrieve(query, max(k * 4, 20), source_folder=source_folder)

    combined = {}
    for document, score in vector_results:
        key = retrieval_key(document)
        combined.setdefault(key, {"document": document, "vector": 0.0, "keyword": 0.0})
        combined[key]["vector"] = max(combined[key]["vector"], score)

    for document, score in keyword_results:
        key = retrieval_key(document)
        combined.setdefault(key, {"document": document, "vector": 0.0, "keyword": 0.0})
        combined[key]["keyword"] = max(combined[key]["keyword"], score)

    results = []
    for item in combined.values():
        score = (VECTOR_WEIGHT * item["vector"]) + (KEYWORD_WEIGHT * item["keyword"])
        document = with_retrieval_metadata(
            item["document"],
            score,
            "hybrid",
            keyword_score=item["keyword"] or None,
        )
        results.append((document, score))

    return sorted(results, key=lambda result: result[1], reverse=True)[:k]


def retrieve(
    query: str,
    k: int = 4,
    source_folder: str | None = None,
    search_type: str = "hybrid",
) -> list[tuple[Document, float]]:
    if search_type not in SEARCH_TYPES:
        raise ValueError(
            f"Unknown search type '{search_type}'. Choose one of {SEARCH_TYPES}."
        )

    if search_type == "vector":
        return vector_retrieve(query, k, source_folder=source_folder)
    if search_type == "keyword":
        return keyword_retrieve(query, k, source_folder=source_folder)

    return hybrid_retrieve(query, k, source_folder=source_folder)


def format_result(document: Document, score: float) -> str:
    filename = document.metadata.get("filename") or document.metadata.get(
        "source", "unknown"
    )
    chunk = document.metadata.get("chunk", "?")
    method = document.metadata.get("retrieval_method", "unknown")
    content = " ".join(document.page_content.split())
    return f"[{method} {score:.4f}] {filename}#{chunk}\n{content}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--source-folder")
    parser.add_argument("--search-type", choices=SEARCH_TYPES, default="hybrid")
    args = parser.parse_args()

    for document, score in retrieve(
        args.query,
        k=args.k,
        source_folder=args.source_folder,
        search_type=args.search_type,
    ):
        print(format_result(document, score))
        print()
