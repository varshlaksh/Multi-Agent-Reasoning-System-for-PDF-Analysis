import pdfplumber
import tiktoken
from sentence_transformers import SentenceTransformer
from utils.chroma_client import get_or_create_collection
from dotenv import load_dotenv
import os

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

TOKENIZER = tiktoken.get_encoding("cl100k_base")
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def extract_pages(file_path: str, doc_name: str) -> list[dict]:
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "text": text.strip(),
                    "page_number": i + 1,
                    "doc_name": doc_name
                })
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    chunks = []
    chunk_id = 0
    for page in pages:
        tokens = TOKENIZER.encode(page["text"])
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for start in range(0, len(tokens), step):
            end = start + CHUNK_SIZE
            chunk_tokens = tokens[start:end]
            chunk_text = TOKENIZER.decode(chunk_tokens)
            chunks.append({
                "chunk_id": f"{page['doc_name']}_chunk_{chunk_id}",
                "text": chunk_text,
                "page_number": page["page_number"],
                "doc_name": page["doc_name"]
            })
            chunk_id += 1
            if end >= len(tokens):
                break
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    return embedder.encode(texts, show_progress_bar=True).tolist()


def index_chunks(chunks: list[dict], collection_name: str = "pdf_store"):
    collection = get_or_create_collection(collection_name)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "page_number": c["page_number"],
                "doc_name": c["doc_name"],
                "chunk_id": c["chunk_id"]
            }
            for c in chunks
        ]
    )
    return len(chunks)


def ingest_pdf(file_path: str, doc_name: str, collection_name: str = "pdf_store") -> dict:
    print(f"[ingestion] Starting: {doc_name}")
    pages = extract_pages(file_path, doc_name)
    print(f"[ingestion] Extracted {len(pages)} pages")
    chunks = chunk_pages(pages)
    print(f"[ingestion] Created {len(chunks)} chunks")
    count = index_chunks(chunks, collection_name)
    print(f"[ingestion] Indexed {count} chunks into ChromaDB")
    return {
        "doc_name": doc_name,
        "pages": len(pages),
        "chunks": count
    }


def embed_query(query: str) -> list[float]:
    return embedder.encode([query]).tolist()[0]
