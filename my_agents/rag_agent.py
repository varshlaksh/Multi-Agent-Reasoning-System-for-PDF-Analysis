from ingestion import embed_query
from utils.chroma_client import get_or_create_collection
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def retrieve_chunks(query: str, collection_name: str = "pdf_store", top_k: int = 5) -> list[dict]:
    collection = get_or_create_collection(collection_name)
    query_embedding = embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "doc_name": results["metadatas"][0][i]["doc_name"],
            "page_number": results["metadatas"][0][i]["page_number"],
            "chunk_id": results["metadatas"][0][i]["chunk_id"],
            "score": round(results["distances"][0][i], 4)
        })
    return chunks

def run_rag_agent(query: str, collection_name: str = "pdf_store") -> dict:
    chunks = retrieve_chunks(query, collection_name)
    if not chunks:
        return {"answer": "No relevant information found in uploaded documents.", "evidence": []}

    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[{i+1}] (Doc: {chunk['doc_name']}, Page: {chunk['page_number']})\n{chunk['text']}\n\n"

    prompt = f"""You are a helpful assistant that answers questions strictly based on the provided document context.

Context from documents:
{context}

Question: {query}

Instructions:
- Answer using ONLY the context above
- Cite sources using [1], [2] etc.
- If answer not in context, say "This information is not available in the uploaded documents"

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return {
        "answer": response.choices[0].message.content,
        "evidence": chunks
    }
