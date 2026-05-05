from hashlib import sha1
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from settings import DATA_DIR
from vector_store import get_vector_store

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def data_files() -> list[Path]:
    return sorted(
        path
        for path in DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def source_folders() -> list[str]:
    folders = {
        source_metadata(path)["source_folder"]
        for path in data_files()
    }
    return sorted(folders)


def source_metadata(path: Path) -> dict:
    relative_path = path.relative_to(DATA_DIR).as_posix()
    return {
        "source": relative_path,
        "filename": path.name,
        "source_folder": Path(relative_path).parent.as_posix()
        if Path(relative_path).parent.as_posix() != "."
        else "",
    }


def load_text_document(path: Path) -> Document | None:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None

    return Document(page_content=text, metadata=source_metadata(path))


def load_pdf_documents(path: Path) -> list[Document]:
    reader = PdfReader(path)
    documents = []

    # PDFs are loaded page-by-page so source metadata can point back to the page.
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            metadata = source_metadata(path)
            metadata["page"] = page_number
            documents.append(Document(page_content=text, metadata=metadata))

    return documents


def load_documents(paths: list[Path] | None = None) -> list[Document]:
    documents = []

    for path in paths or data_files():
        if path.suffix.lower() == ".pdf":
            documents.extend(load_pdf_documents(path))
            continue

        document = load_text_document(path)
        if document:
            documents.append(document)

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    chunk_counts_by_source = {}
    for chunk in chunks:
        # Chunk numbers are source-local so the UI shows stable "file chunk N" labels.
        source = chunk.metadata.get("source", "")
        chunk_counts_by_source[source] = chunk_counts_by_source.get(source, 0) + 1
        chunk_number = chunk_counts_by_source[source]
        chunk.metadata["chunk"] = chunk_number

    return chunks


def chunk_id(chunk: Document) -> str:
    source = chunk.metadata.get("source", "")
    chunk_number = chunk.metadata.get("chunk", "")
    raw_id = f"{source}:{chunk_number}:{chunk.page_content}"
    return sha1(raw_id.encode("utf-8")).hexdigest()


def ingest() -> tuple[int, int]:
    files = data_files()
    documents = load_documents(files)
    chunks = split_documents(documents)
    store = get_vector_store(pre_delete_collection=True)

    if not chunks:
        return len(files), 0

    store.add_documents(chunks, ids=[chunk_id(chunk) for chunk in chunks])
    return len(files), len(chunks)


if __name__ == "__main__":
    document_count, chunk_count = ingest()
    print(f"Ingested {document_count} documents into {chunk_count} chunks.")
