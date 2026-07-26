from ingestion import ingest_pdf, embed_query
from utils.chroma_client import get_or_create_collection

result = ingest_pdf(
    file_path="Market Research_ Smart Electricity Bill Analysis Tool for Indian Households.pdf",
    doc_name="sample",
    collection_name="pdf_store"
)
print(result)

col = get_or_create_collection("pdf_store")
print(f"\nTotal chunks in DB: {col.count()}")

query_embedding = embed_query("what is this document about")

results = col.query(query_embeddings=[query_embedding], n_results=3)

for i, doc in enumerate(results["documents"][0]):
    meta = results["metadatas"][0][i]
    print(f"\n--- Result {i+1} ---")
    print(f"Doc: {meta['doc_name']} | Page: {meta['page_number']}")
    print(doc[:200])
