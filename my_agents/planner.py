from my_agents.rag_agent import run_rag_agent
from my_agents.summary_agent import run_summary_agent
from my_agents.comparator import run_comparator_agent
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def detect_intent(query: str) -> str:
    """
    Asks Groq to classify the user query into one of three intents.
    Returns: "rag", "summary", or "compare"
    """
    prompt = f"""Classify the following user query into exactly one of these three categories:
- "summary": user wants a summary or overview of the document(s)
- "compare": user wants to compare, contrast, or find differences between documents
- "rag": user wants to ask a specific question or find specific information

Reply with ONLY one word: summary, compare, or rag. Nothing else.

Query: {query}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0  # zero temperature = deterministic, no creativity
    )
    intent = response.choices[0].message.content.strip().lower()

    # Safety fallback — if model returns something unexpected, default to rag
    if intent not in ["summary", "compare", "rag"]:
        intent = "rag"

    return intent


def run_planner(query: str, collection_name: str = "pdf_store") -> dict:
    """
    Main entry point. Detects intent, routes to correct agent(s), returns result.
    Result always has: answer, evidence, intent, agent_chain
    """
    print(f"[planner] Query: {query}")

    intent = detect_intent(query)
    print(f"[planner] Detected intent: {intent}")

    if intent == "summary":
        print("[planner] Route: Summarization Agent")
        result = run_summary_agent(collection_name=collection_name)
        result["intent"] = intent
        result["agent_chain"] = ["Planner", "Summarization Agent"]

    elif intent == "compare":
        print("[planner] Route: RAG Agent → Comparator Agent")
        rag_result = run_rag_agent(query, collection_name)
        result = run_comparator_agent(query, rag_result)
        result["intent"] = intent
        result["agent_chain"] = ["Planner", "RAG Agent", "Comparator Agent"]

    else:  # rag
        print("[planner] Route: RAG Agent")
        result = run_rag_agent(query, collection_name)
        result["intent"] = intent
        result["agent_chain"] = ["Planner", "RAG Agent"]

    print(f"[planner] Done. Chain: {result['agent_chain']}")
    return result
