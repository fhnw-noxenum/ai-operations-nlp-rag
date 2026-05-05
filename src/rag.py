import argparse
from collections.abc import Iterator

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from retrieve import SEARCH_TYPES, retrieve
from settings import CHAT_MODEL

SYSTEM_PROMPT = """You answer questions using only the provided context.
If the context does not contain the answer, say that you do not know.
Cite the source filenames you used."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)


def source_label(document: Document) -> str:
    source = document.metadata.get("filename") or document.metadata.get(
        "source", "unknown"
    )
    chunk = document.metadata.get("chunk")
    return f"{source}#{chunk}" if chunk else source


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"Source: {source_label(document)}\n{document.page_content}"
        for document in documents
    )


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    return str(content)


def answer(
    question: str,
    k: int = 4,
    source_folder: str | None = None,
    search_type: str = "hybrid",
) -> dict:
    results = retrieve(question, k=k, source_folder=source_folder, search_type=search_type)
    documents = [document for document, _ in results]
    chain = PROMPT | ChatOpenAI(model=CHAT_MODEL)
    response = chain.invoke({"question": question, "context": format_context(documents)})

    return {
        "answer": content_text(response.content),
        "sources": list(dict.fromkeys(source_label(document) for document in documents)),
        "documents": documents,
    }


def stream_answer(
    question: str,
    k: int = 4,
    source_folder: str | None = None,
    search_type: str = "hybrid",
) -> tuple[Iterator[str], list[Document]]:
    results = retrieve(question, k=k, source_folder=source_folder, search_type=search_type)
    documents = [document for document, _ in results]
    chain = PROMPT | ChatOpenAI(model=CHAT_MODEL)

    # Stream model chunks to Streamlit while keeping retrieved documents for source display.
    def stream() -> Iterator[str]:
        for chunk in chain.stream(
            {"question": question, "context": format_context(documents)}
        ):
            text = content_text(chunk.content)
            if text:
                yield text

    return stream(), documents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--source-folder")
    parser.add_argument("--search-type", choices=SEARCH_TYPES, default="hybrid")
    args = parser.parse_args()

    result = answer(
        args.question,
        k=args.k,
        source_folder=args.source_folder,
        search_type=args.search_type,
    )
    print(result["answer"])
    print()
    print("Sources:")
    for source in result["sources"]:
        print(f"- {source}")
