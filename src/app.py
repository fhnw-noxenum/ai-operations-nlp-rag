import streamlit as st

from ingest import ingest, source_folders
from rag import stream_answer
from retrieve import SEARCH_TYPES

st.set_page_config(page_title="RAG Lab", layout="centered")


def folder_label(folder: str | None) -> str:
    if folder is None:
        return "All sources"
    return folder or "."


def score_label(document) -> str:
    score = document.metadata.get("retrieved_score")
    if score is None:
        return "n/a"
    return f"{score:.4f}"


def show_sources(documents):
    if not documents:
        return

    with st.expander("Sources"):
        for document in documents:
            metadata = document.metadata
            filename = metadata.get("filename") or metadata.get("source", "unknown")
            chunk = metadata.get("chunk", "?")
            method = metadata.get("retrieval_method", "unknown")
            page = metadata.get("page")
            page_text = f" · page {page}" if page else ""

            # The source panel now exposes retrieval metadata for each returned chunk.
            st.markdown(
                f"**{filename} · chunk {chunk}{page_text} · "
                f"score {score_label(document)} · {method}**"
            )
            st.write(document.page_content)


with st.sidebar:
    st.title("RAG Lab")
    k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)
    search_type = st.radio(
        "Search mode",
        options=SEARCH_TYPES,
        index=SEARCH_TYPES.index("hybrid"),
        format_func=str.title,
    )
    selected_folder = st.selectbox(
        "Source folder",
        options=[None, *source_folders()],
        format_func=folder_label,
    )

    if st.button("Ingest data", use_container_width=True):
        with st.spinner("Indexing data..."):
            try:
                document_count, chunk_count = ingest()
                st.success(f"Ingested {document_count} documents into {chunk_count} chunks.")
            except Exception as error:
                st.error(str(error))

st.title("RAG Lab")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("documents", []))

question = st.chat_input("Ask a question about your data")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            stream, documents = stream_answer(
                question,
                k=k,
                source_folder=selected_folder,
                search_type=search_type,
            )
            # Stream the response into the chat instead of waiting for the full answer.
            response = st.write_stream(stream)
            show_sources(documents)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "documents": documents,
                }
            )
        except Exception as error:
            st.error(str(error))
