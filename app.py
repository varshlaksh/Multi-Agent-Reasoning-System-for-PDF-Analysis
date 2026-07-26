import streamlit as st
import tempfile
import os
import fitz
from ingestion import ingest_pdf
from my_agents.planner import run_planner
from utils.chroma_client import delete_collection

st.set_page_config(
    page_title="PDF Multi-Agent System",
    page_icon="📄",
    layout="wide"
)

# ── Always visible header ─────────────────────────────────────────────────────
st.title("📄 Multi-Agent PDF Analysis System")
st.caption("Upload PDFs and ask questions. The planner auto-routes to the right agent.")
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "collection_ready" not in st.session_state:
    st.session_state.collection_ready = False
if "doc_paths" not in st.session_state:
    st.session_state.doc_paths = {}
if "nav_doc" not in st.session_state:
    st.session_state.nav_doc = None
if "nav_page" not in st.session_state:
    st.session_state.nav_page = 1

COLLECTION_NAME = "pdf_store"

# ── Sidebar ───────────────────────────────────────────────────────────────────
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
            st.session_state.doc_paths = {}
            st.session_state.nav_doc = None
            st.session_state.nav_page = 1

            progress = st.progress(0, text="Starting ingestion...")
            for i, file in enumerate(uploaded_files):
                progress.progress(
                    int((i / len(uploaded_files)) * 100),
                    text=f"Indexing {file.name}..."
                )
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(file.read())
                tmp.flush()
                tmp_path = tmp.name
                tmp.close()

                result = ingest_pdf(
                    file_path=tmp_path,
                    doc_name=file.name,
                    collection_name=COLLECTION_NAME
                )
                st.session_state.doc_paths[file.name] = tmp_path
                st.session_state.uploaded_docs.append(result)

            progress.progress(100, text="Done!")
            st.session_state.collection_ready = True
            if st.session_state.uploaded_docs:
                st.session_state.nav_doc = st.session_state.uploaded_docs[0]["doc_name"]
            st.rerun()

    if st.session_state.uploaded_docs:
        st.divider()
        st.subheader("📚 Indexed Documents")
        for doc in st.session_state.uploaded_docs:
            st.markdown(f"**{doc['doc_name']}**")
            st.caption(f"{doc['pages']} pages · {doc['chunks']} chunks")

    st.divider()
    st.subheader("🤖 Agent Routing")
    st.markdown("""
- 🔍 **RAG Agent** — specific questions
- 📝 **Summary Agent** — summaries
- ⚖️ **Comparator Agent** — comparisons
    """)

# ── Not ready ─────────────────────────────────────────────────────────────────
if not st.session_state.collection_ready:
    st.info("👈 Upload and index your PDF documents to get started.")
    st.stop()

# ── Main layout ───────────────────────────────────────────────────────────────
chat_col, nav_col = st.columns([6, 4])

# ── RIGHT: Navigator ──────────────────────────────────────────────────────────
with nav_col:
    st.subheader("🗺️ Document Navigator")
    doc_names = list(st.session_state.doc_paths.keys())

    selected_doc = st.selectbox(
        "Document",
        doc_names,
        index=doc_names.index(st.session_state.nav_doc) if st.session_state.nav_doc in doc_names else 0,
        key="nav_doc_select"
    )

    if selected_doc != st.session_state.nav_doc:
        st.session_state.nav_page = 1
        st.session_state.nav_doc = selected_doc

    pdf_path = st.session_state.doc_paths[selected_doc]

    try:
        pdf = fitz.open(pdf_path)
        total_pages = len(pdf)

        current_page = st.number_input(
            f"Page (1 to {total_pages})",
            min_value=1,
            max_value=total_pages,
            value=st.session_state.nav_page,
            step=1,
            key="nav_page_input"
        )
        st.session_state.nav_page = current_page

        page = pdf[current_page - 1]
        pix = page.get_pixmap(dpi=130)
        img_bytes = pix.tobytes("png")
        st.image(img_bytes, caption=f"Page {current_page} of {total_pages}", use_container_width=True)

        cited_here = []
        for msg in st.session_state.messages:
            if msg.get("evidence"):
                for ev in msg["evidence"]:
                    if ev["doc_name"] == selected_doc and ev["page_number"] == current_page:
                        cited_here.append(ev)

        if cited_here:
            st.success(f"📌 {len(cited_here)} citation(s) on this page")
            for ev in cited_here:
                with st.expander(f"Chunk `{ev['chunk_id']}` · Score `{ev['score']}`"):
                    st.markdown(ev["text"][:400] + "...")

    except Exception as e:
        st.error(f"Could not render PDF: {e}")

# ── LEFT: Chat ────────────────────────────────────────────────────────────────
with chat_col:
    st.subheader("💬 Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("agent_chain"):
                st.caption(f"🔗 {' → '.join(msg['agent_chain'])}")
            if msg.get("evidence"):
                with st.expander(f"📎 {len(msg['evidence'])} source(s)"):
                    for j, ev in enumerate(msg["evidence"]):
                        st.markdown(
                            f"**[{j+1}]** `{ev['doc_name']}` · "
                            f"Page **{ev['page_number']}** · "
                            f"Chunk `{ev['chunk_id']}` · "
                            f"Score `{ev['score']}`"
                        )
                        st.caption(ev["text"][:250] + "...")
                        if st.button(
                            f"📍 Jump to Page {ev['page_number']}",
                            key=f"jump_{id(msg)}_{j}"
                        ):
                            st.session_state.nav_doc = ev["doc_name"]
                            st.session_state.nav_page = ev["page_number"]
                            st.rerun()
                        st.divider()

    if prompt := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_planner(prompt, collection_name=COLLECTION_NAME)
            st.markdown(result["answer"])
            if result.get("agent_chain"):
                st.caption(f"🔗 {' → '.join(result['agent_chain'])}")
            if result.get("evidence"):
                with st.expander(f"📎 {len(result['evidence'])} source(s)"):
                    for j, ev in enumerate(result["evidence"]):
                        st.markdown(
                            f"**[{j+1}]** `{ev['doc_name']}` · "
                            f"Page **{ev['page_number']}** · "
                            f"Chunk `{ev['chunk_id']}` · "
                            f"Score `{ev['score']}`"
                        )
                        st.caption(ev["text"][:250] + "...")
                        st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "evidence": result.get("evidence", []),
            "agent_chain": result.get("agent_chain", [])
        })
