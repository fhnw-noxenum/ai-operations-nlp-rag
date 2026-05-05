import streamlit as st

from ingest import ingest
from rag import answer, source_label

st.set_page_config(page_title="RAG Lab", layout="centered")


def show_sources(documents):
    if not documents:
        return

    with st.expander("Sources"):
        for document in documents:
            st.markdown(f"**{source_label(document)}**")
            st.write(document.page_content)


with st.sidebar:
    st.title("RAG Lab")
    k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)

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
        with st.spinner("Thinking..."):
            try:
                result = answer(question, k=k)
                st.markdown(result["answer"])
                show_sources(result["documents"])
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "documents": result["documents"],
                    }
                )
            except Exception as error:
                st.error(str(error))
