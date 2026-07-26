from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_comparator_agent(query: str, rag_result: dict) -> dict:
    evidence = rag_result.get("evidence", [])
    if not evidence:
        return {"answer": "No evidence available to compare.", "evidence": []}

    docs_map = {}
    for chunk in evidence:
        name = chunk["doc_name"]
        if name not in docs_map:
            docs_map[name] = []
        docs_map[name].append(chunk["text"])

    if len(docs_map) < 2:
        return {"answer": "Comparison requires at least 2 documents. Only one document found in evidence.", "evidence": evidence}

    context_blocks = ""
    for doc_name, texts in docs_map.items():
        context_blocks += f"\n=== {doc_name} ===\n" + "\n".join(texts) + "\n"

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""Compare these documents for the question: {query}

{context_blocks}

Structure response as:
SIMILARITIES:
DIFFERENCES:
CONCLUSION:"""}],
        temperature=0.3
    )
    return {"answer": response.choices[0].message.content, "evidence": evidence}
