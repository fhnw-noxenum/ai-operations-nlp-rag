import argparse

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from retrieve import retrieve
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
    source = document.metadata.get("source", "unknown")
    chunk = document.metadata.get("chunk")
    return f"{source}#{chunk}" if chunk else source


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"Source: {source_label(document)}\n{document.page_content}"
        for document in documents
    )


def answer(question: str, k: int = 4) -> dict:
    results = retrieve(question, k=k)
    documents = [document for document, _ in results]
    chain = PROMPT | ChatOpenAI(model=CHAT_MODEL)
    response = chain.invoke({"question": question, "context": format_context(documents)})

    return {
        "answer": response.content,
        "sources": list(dict.fromkeys(source_label(document) for document in documents)),
        "documents": documents,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    result = answer(args.question, k=args.k)
    print(result["answer"])
    print()
    print("Sources:")
    for source in result["sources"]:
        print(f"- {source}")
