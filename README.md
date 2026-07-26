# Multi-Agent PDF Analysis System

A Streamlit application for conversational interaction with PDF documents using a multi-agent architecture. Upload PDFs, ask questions in plain English — the system automatically routes to the right agent.

---

## Stack

- **Frontend** — Streamlit
- **LLM** — Groq (llama-3.3-70b-versatile)
- **Embeddings** — sentence-transformers (all-MiniLM-L6-v2), runs locally
- **Vector Store** — ChromaDB (persistent, on-disk)
- **PDF Parsing** — pdfplumber
- **PDF Rendering** — PyMuPDF

---

## Architecture

```
User Query → Planner → RAG Agent                    (specific questions)
                     → Summarization Agent           (summaries)
                     → RAG Agent → Comparator Agent  (cross-doc comparison)
```

The Planner detects intent from the query and routes to the correct agent chain automatically. No manual mode switching.

---

## Project Structure

```
.
├── app.py                    # Streamlit UI
├── ingestion.py              # PDF extraction → chunking → embedding → ChromaDB
├── my_agents/
│   ├── planner.py            # Intent detection + routing
│   ├── rag_agent.py          # Retrieval + grounded answer generation
│   ├── summary_agent.py      # Map-reduce summarization
│   ├── comparator.py         # Cross-document comparison
│   └── sdk_planner.py        # AgentSDK implementation (Agent, Runner, handoffs)
├── utils/
│   └── chroma_client.py      # Shared ChromaDB client
├── requirements.txt
└── .env                      # Not committed
```

---

## Setup

```bash
git clone https://github.com/varshlaksh/Multi-Agent-Reasoning-System-for-PDF-Analysis.git
cd Multi-Agent-Reasoning-System-for-PDF-Analysis

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_key_here
```

Get a free key at https://console.groq.com/keys — no credit card required.

```bash
streamlit run app.py
```

---

## Agents

**Planner**
Classifies user intent (question / summary / comparison) using the LLM and routes to the appropriate agent chain.

**RAG Agent**
Embeds the query locally, retrieves top-5 chunks from ChromaDB, passes context to the LLM. Returns a grounded answer with numbered citations and a ranked evidence list (doc name, page, chunk ID, similarity score).

**Summarization Agent**
Fetches all chunks from the vector store, summarizes each chunk individually (map step), then combines into a single coherent summary (reduce step).

**Comparator Agent**
Takes RAG evidence as input. Groups chunks by document and asks the LLM to compare them. Output is structured as SIMILARITIES / DIFFERENCES / CONCLUSION.

---

## Features

- Upload multiple PDFs simultaneously
- Automatic intent detection — no toggles or buttons to switch modes
- Citations in every answer with doc name, page number, chunk ID, score
- Agent chain trace shown on every response
- Document Navigator renders PDF pages and highlights cited passages
- Jump-to-page from any citation in the evidence panel
- Hallucination guard — refuses to answer questions outside the uploaded documents

---

## AgentSDK Note

`my_agents/sdk_planner.py` contains the full OpenAI AgentSDK implementation using `Agent`, `Runner`, `function_tool`, and `handoffs`. The primary planner uses Groq as a free-tier backend with the same architecture. To switch to AgentSDK, add `OPENAI_API_KEY` to `.env` and update the import in `app.py`.

---

## Screenshots

> Add screenshots here

---

## Author

Lakshya Varshney — [github.com/varshlaksh](https://github.com/varshlaksh)