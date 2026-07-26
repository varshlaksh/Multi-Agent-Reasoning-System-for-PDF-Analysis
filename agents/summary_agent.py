from utils.chroma_client import get_or_create_collection
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_summary_agent(collection_name: str = "pdf_store", doc_name: str = None) -> dict:
    collection = get_or_create_collection(collection_name)
    all_data = collection.get(include=["documents", "metadatas"])

    if not all_data["documents"]:
        return {"answer": "No documents found. Please upload a PDF first.", "evidence": []}

    docs, metas = [], []
    for text, meta in zip(all_data["documents"], all_data["metadatas"]):
        if doc_name is None or meta["doc_name"] == doc_name:
            docs.append(text)
            metas.append(meta)

    if not docs:
        return {"answer": f"No content found for document: {doc_name}", "evidence": []}

    chunk_summaries = []
    for text in docs:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"Summarize this passage in 2-3 sentences:\n\n{text}"}],
            temperature=0.3
        )
        chunk_summaries.append(response.choices[0].message.content.strip())

    combined = "\n\n".join([f"Part {i+1}: {s}" for i, s in enumerate(chunk_summaries)])
    final_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"Combine these section summaries into one coherent, non-redundant summary:\n\n{combined}"}],
        temperature=0.3
    )

    evidence = [{"text": docs[i], "doc_name": metas[i]["doc_name"], "page_number": metas[i]["page_number"], "chunk_id": metas[i]["chunk_id"], "score": 1.0} for i in range(len(docs))]
    return {"answer": final_response.choices[0].message.content, "evidence": evidence}
