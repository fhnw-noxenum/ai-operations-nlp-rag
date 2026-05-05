from hashlib import sha1
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import DATA_DIR
from vector_store import get_vector_store

SUPPORTED_SUFFIXES = {".txt", ".md"}


def data_files() -> list[Path]:
    return sorted(
        path
        for path in DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_documents() -> list[Document]:
    documents = []

    for path in data_files():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path.relative_to(DATA_DIR))},
                )
            )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    for chunk_number, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk"] = chunk_number

    return chunks


def chunk_id(chunk: Document) -> str:
    source = chunk.metadata.get("source", "")
    chunk_number = chunk.metadata.get("chunk", "")
    raw_id = f"{source}:{chunk_number}:{chunk.page_content}"
    return sha1(raw_id.encode("utf-8")).hexdigest()


def ingest() -> tuple[int, int]:
    documents = load_documents()
    chunks = split_documents(documents)
    store = get_vector_store(pre_delete_collection=True)

    if not chunks:
        return len(documents), 0

    store.add_documents(chunks, ids=[chunk_id(chunk) for chunk in chunks])
    return len(documents), len(chunks)


if __name__ == "__main__":
    document_count, chunk_count = ingest()
    print(f"Ingested {document_count} documents into {chunk_count} chunks.")
