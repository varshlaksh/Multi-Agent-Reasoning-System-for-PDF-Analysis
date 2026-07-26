import streamlit as st
import tempfile
import os
from ingestion import ingest_pdf
from agents.planner import run_planner
from utils.chroma_client import delete_collection

st.set_page_config(
    page_title="PDF Multi-Agent System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Multi-Agent PDF Analysis System")
st.caption("Upload PDFs and ask questions. The system auto-routes to the right agent.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "collection_ready" not in st.session_state:
    st.session_state.collection_ready = False

COLLECTION_NAME = "pdf_store"

with st.sidebar:
    st.header("📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 Index Documents", type="primary"):
            delete_collection(COLLECTION_NAME)
            st.session_state.uploaded_docs = []
            st.session_state.collection_ready = False
            st.session_state.messages = []

            progress = st.progress(0, text="Starting ingestion...")

            for i, file in enumerate(uploaded_files):
                progress.progress(
                    int((i / len(uploaded_files)) * 100),
                    text=f"Indexing {file.name}..."
                )
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name

                result = ingest_pdf(
                    file_path=tmp_path,
                    doc_name=file.name,
                    collection_name=COLLECTION_NAME
                )
                os.unlink(tmp_path)
                st.session_state.uploaded_docs.append(result)

            progress.progress(100, text="Done!")
            st.session_state.collection_ready = True
            st.rerun()

    if st.session_state.uploaded_docs:
        st.divider()
        st.subheader("📚 Indexed Documents")
        for doc in st.session_state.uploaded_docs:
            st.markdown(f"**{doc['doc_name']}**")
            st.caption(f"{doc['pages']} pages · {doc['chunks']} chunks")

    st.divider()
    st.subheader("🤖 How it works")
    st.markdown("""
    The **Planner** reads your query and auto-routes to:
    - 🔍 **RAG Agent** — specific questions
    - 📝 **Summary Agent** — document summaries
    - ⚖️ **Comparator Agent** — cross-doc comparison
    """)

if not st.session_state.collection_ready:
    st.info("👈 Upload and index your PDF documents to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("evidence"):
                with st.expander(f"📎 Evidence ({len(msg['evidence'])} sources)", expanded=False):
                    for j, ev in enumerate(msg["evidence"]):
                        st.markdown(f"**[{j+1}]** `{ev['doc_name']}` · Page {ev['page_number']} · Score `{ev['score']}`")
                        st.caption(ev["text"][:300] + "...")
                        st.divider()
            if msg.get("agent_chain"):
                st.caption(f"🔗 Agent chain: {' → '.join(msg['agent_chain'])}")

    if prompt := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_planner(prompt, collection_name=COLLECTION_NAME)

            st.markdown(result["answer"])

            if result.get("evidence"):
                with st.expander(f"📎 Evidence ({len(result['evidence'])} sources)", expanded=False):
                    for j, ev in enumerate(result["evidence"]):
                        st.markdown(f"**[{j+1}]** `{ev['doc_name']}` · Page {ev['page_number']} · Score `{ev['score']}`")
                        st.caption(ev["text"][:300] + "...")
                        st.divider()

            st.caption(f"🔗 Agent chain: {' → '.join(result['agent_chain'])}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "evidence": result.get("evidence", []),
            "agent_chain": result.get("agent_chain", [])
        })
