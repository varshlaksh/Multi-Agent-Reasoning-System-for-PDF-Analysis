# AGENTS.md — System Design & Decision Log

This document explains every architectural decision made in this system. It exists so that any engineer picking up this codebase understands not just what was built, but why each choice was made and what the tradeoffs are.

---

## 1. Ingestion Pipeline

### Text Extraction — pdfplumber

pdfplumber was chosen over PyMuPDF for text extraction because it handles tables and structured layouts more cleanly. PyMuPDF is faster but returns raw character streams that require more post-processing for columnar PDFs. For this system where document quality matters for retrieval accuracy, pdfplumber is the safer default.

**What to change if needed:** Swap to `PyMuPDF (fitz)` if speed becomes a bottleneck on large PDFs (100+ pages). PyMuPDF is approximately 3x faster for plain text extraction.

### Chunking Strategy — Fixed token window with overlap

- Chunk size: 400 tokens
- Overlap: 50 tokens
- Tokenizer: `cl100k_base` (same tokenizer used by GPT-4 and OpenAI embedding models)

**Why 400 tokens:** Large enough to contain a complete idea or paragraph, small enough that retrieved chunks stay focused. Chunks above 600 tokens tend to dilute similarity scores because they contain multiple topics.

**Why 50 token overlap:** Prevents context loss at chunk boundaries. If a key sentence falls at the end of chunk N and the beginning of chunk N+1, overlap ensures it appears in at least one retrieved chunk.

**What to change if needed:** For highly structured documents (legal, medical), switch to semantic chunking using sentence boundaries instead of fixed token windows. Libraries like `langchain.text_splitter.RecursiveCharacterTextSplitter` handle this well.

### Embeddings — sentence-transformers (all-MiniLM-L6-v2)

Runs fully locally. No API key, no cost, no rate limits. Produces 384-dimensional vectors.

**Why this model:** Best balance of speed and quality for retrieval tasks at this scale. Benchmarks well on MTEB retrieval tasks. Downloads once (~90MB) and runs on CPU without GPU.

**Tradeoff:** OpenAI's `text-embedding-3-small` produces 1536-dimensional vectors with higher accuracy, especially for domain-specific content. If retrieval quality is insufficient, switching to OpenAI embeddings is the first lever to pull.

**What to change if needed:** For production, replace with `text-embedding-3-small` via OpenAI API or `BAAI/bge-large-en` for higher accuracy locally.

### Vector Store — ChromaDB

Persistent client storing embeddings to disk under `./chroma_store`. Cosine similarity used for distance metric.

**Why cosine over Euclidean:** Cosine similarity measures the angle between vectors, not their magnitude. Two sentences with the same meaning but different lengths will have similar angles but different magnitudes. Cosine is the correct metric for semantic similarity.

**Why ChromaDB over FAISS:** ChromaDB supports metadata filtering natively (filter by doc_name, page_number). FAISS requires a separate metadata store and custom filtering logic. For multi-document retrieval with citation requirements, ChromaDB reduces complexity significantly.

**What to change if needed:** FAISS is faster at scale (millions of vectors). If the system needs to handle 100+ documents simultaneously, FAISS with a metadata sidecar (SQLite or Postgres) is the better choice.

---

## 2. Agent Architecture

### Planner

The Planner is a zero-temperature LLM call that classifies the query into one of three intents: `rag`, `summary`, `compare`. It then routes to the appropriate agent chain.

**Why LLM-based intent detection over keyword matching:** Keyword matching breaks on paraphrasing. "Give me an overview" should route to summarization but contains no keyword like "summarize". LLM classification handles semantic intent correctly.

**Why zero temperature:** Intent classification is a deterministic task. There is exactly one correct routing decision per query. Temperature=0 removes randomness and ensures consistent routing.

**Fallback:** If the LLM returns anything outside `[rag, summary, compare]`, the planner defaults to `rag`. This prevents silent failures.

**What to change if needed:** Add more intent classes as new agents are added. For example, adding a Timeline Agent requires adding `timeline` as a valid intent and a corresponding routing branch.

### RAG Agent

Retrieves top-5 chunks by cosine similarity, builds a numbered context block, and prompts the LLM to answer only from that context.

**Why top-5:** Balances context coverage vs prompt length. Top-3 misses relevant chunks for complex queries. Top-10 bloats the prompt and reduces answer precision.

**Why grounded prompting:** The system prompt explicitly instructs the LLM to answer only from retrieved context and say "not available" if the answer is absent. This is the primary hallucination prevention mechanism.

**Context management:** Each chunk is labeled [1], [2] etc. in the prompt. The LLM is instructed to cite these numbers inline. The evidence list returned to the UI is ordered to match these numbers so citations are traceable.

**What to change if needed:** Add reranking (e.g. Cohere Rerank or cross-encoder) between retrieval and generation for higher precision. Reranking re-scores the top-k chunks using a more expensive model before passing to the LLM.

### Summarization Agent

Uses map-reduce strategy:
1. Map: each chunk is summarized individually in a separate LLM call
2. Reduce: all chunk summaries are combined into one final coherent summary

**Why map-reduce over stuffing:** Stuffing all chunks into one prompt fails for documents exceeding the context window. Map-reduce scales to any document length.

**Tradeoff:** Map-reduce makes N+1 LLM calls (N chunks + 1 final). For a 50-page document this could be 30-40 API calls. For Groq's free tier this is fine; for paid APIs this increases cost.

**What to change if needed:** For very long documents, use chunk-refinement instead — summarize chunk 1, then refine the summary with chunk 2, then refine again with chunk 3. This produces smoother summaries but is sequential and slower.

### Comparator Agent

Takes the RAG Agent's retrieved evidence as input, groups chunks by document name, and prompts the LLM to compare across groups.

**Why chain after RAG instead of direct retrieval:** The Comparator needs the same evidence chunks the RAG Agent found. Re-retrieving independently could return different chunks, making the comparison inconsistent with the RAG answer. Chaining ensures the Comparator reasons over the same evidence.

**Output structure:** SIMILARITIES / DIFFERENCES / CONCLUSION is enforced in the prompt. This prevents the LLM from returning a free-form paragraph that the UI cannot parse or present cleanly.

**Limitation:** Requires at least 2 documents. Returns a clear error message if only one document is in the vector store. This is checked before the LLM call to avoid wasting tokens.

**What to change if needed:** Add a Timeline Agent that sorts events by date across documents. The same chaining pattern applies — RAG retrieves evidence, Timeline Agent sorts and structures it chronologically.

---

## 3. AgentSDK Layer

`my_agents/sdk_planner.py` implements the full OpenAI AgentSDK pattern:

- `@function_tool` decorators wrap retrieval functions as callable tools
- `Agent` objects are instantiated with names, instructions, and tool lists
- `handoffs` list on the Planner agent enables delegation to sub-agents
- `Runner.run()` executes the agent loop asynchronously
- `asyncio.run()` bridges the async SDK into Streamlit's synchronous context

**Why a separate file:** The primary planner (`planner.py`) uses Groq directly for reliability without requiring an OpenAI key. The SDK planner (`sdk_planner.py`) is the AgentSDK-compliant version ready to activate with a valid `OPENAI_API_KEY`. Both implement identical logic.

**To activate AgentSDK:** Add `OPENAI_API_KEY=sk-...` to `.env` and change the import in `app.py` from `my_agents.planner` to `my_agents.sdk_planner`.

---

## 4. Frontend Decisions

### Streamlit layout — two columns

Chat on the left (wider), Document Navigator on the right. This mirrors how a human would use the system — read the answer on the left, verify the source on the right.

### Document Navigator

Renders PDF pages using PyMuPDF at 130 DPI. Lower DPI (72) is too blurry to read. Higher DPI (200) slows rendering noticeably on large pages.

Citation jump: clicking "📍 Jump to Page N" writes to `st.session_state.nav_page` and calls `st.rerun()`. This forces Streamlit to re-render the navigator with the new page number.

**Why session state for navigation:** Streamlit rerenders the entire page on every interaction. Without session state, page number resets to 1 on every rerender. Session state persists values across rerenders.

### PDF temp file storage

Uploaded files are saved to `tempfile.NamedTemporaryFile` with `delete=False`. The path is stored in `st.session_state.doc_paths`. This is required because:
1. pdfplumber needs a file path, not a bytes buffer
2. PyMuPDF needs a file path to render pages
3. The file must persist for the lifetime of the session

**What to change if needed:** In production, store uploaded files in S3 or GCS instead of temp files. Temp files are cleared on server restart.

---

## 5. What Can Be Extended

| Feature | How to add |
|---|---|
| Timeline Agent | New agent file, add `timeline` intent to planner, chain after RAG |
| Aggregator Agent | Deduplicates overlapping evidence from RAG before final answer |
| Reranking | Add Cohere Rerank between ChromaDB retrieval and LLM generation |
| Streaming responses | Replace `generate_content` with streaming API calls, use `st.write_stream` |
| Multi-user support | Replace local ChromaDB with a hosted instance, scope collections by user ID |
| Better chunking | Swap fixed token chunks for semantic sentence-boundary chunking |
| Auth | Add Streamlit authentication before file upload |

---

## 6. Known Limitations

- Single-page PDFs show "Page 1 to 1" in navigator — correct behavior, not a bug
- Summarization makes N+1 LLM calls — slow for large documents on rate-limited APIs
- AgentSDK requires paid OpenAI key — Groq backend used as equivalent free alternative
- Temp files cleared on Streamlit server restart — re-upload required after restart
- No persistence of chat history across sessions — stored only in session state