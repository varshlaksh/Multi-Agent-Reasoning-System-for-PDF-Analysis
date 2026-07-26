import asyncio
import os
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
from ingestion import embed_query
from utils.chroma_client import get_or_create_collection
from groq import Groq

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

COLLECTION_NAME = "pdf_store"


@function_tool
def retrieve_documents(query: str) -> str:
    """Retrieves relevant chunks from uploaded PDFs based on the query."""
    collection = get_or_create_collection(COLLECTION_NAME)
    query_embedding = embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=5)

    if not results["documents"][0]:
        return "No relevant documents found."

    output = ""
    for i in range(len(results["documents"][0])):
        output += (
            f"[{i+1}] Doc: {results['metadatas'][0][i]['doc_name']} | "
            f"Page: {results['metadatas'][0][i]['page_number']} | "
            f"Score: {round(results['distances'][0][i], 4)}\n"
            f"{results['documents'][0][i]}\n\n"
        )
    return output


@function_tool
def get_all_chunks() -> str:
    """Retrieves all document chunks from the vector store for summarization."""
    collection = get_or_create_collection(COLLECTION_NAME)
    all_data = collection.get(include=["documents", "metadatas"])

    if not all_data["documents"]:
        return "No documents found in the store."

    output = ""
    for text, meta in zip(all_data["documents"], all_data["metadatas"]):
        output += (
            f"Doc: {meta['doc_name']} | Page: {meta['page_number']}\n"
            f"{text}\n\n"
        )
    return output


def llm_call(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content


rag_agent = Agent(
    name="RAG Agent",
    instructions="""You are a precise question-answering agent.
Use the retrieve_documents tool to find relevant content, then answer the question.
Always cite sources using [1], [2] format matching the retrieved chunk numbers.
Only answer from retrieved context. If not found, say this information is not in the documents.""",
    tools=[retrieve_documents],
    model="o3-mini"
)

summary_agent = Agent(
    name="Summarization Agent",
    instructions="""You are a document summarization expert.
Use the get_all_chunks tool to retrieve all document content.
Then produce a coherent, non-redundant summary covering key points from all documents.
Preserve factual accuracy and avoid inventing information.""",
    tools=[get_all_chunks],
    model="o3-mini"
)

comparator_agent = Agent(
    name="Comparator Agent",
    instructions="""You are a document comparison expert.
You receive retrieved evidence from multiple documents.
Structure your response as:
SIMILARITIES: (what the documents agree on)
DIFFERENCES: (how the documents differ)
CONCLUSION: (direct answer to the user question)
Be specific and cite document names.""",
    tools=[retrieve_documents],
    model="o3-mini"
)

planner_agent = Agent(
    name="Planner",
    instructions="""You are an orchestration planner that routes user queries to the right agent.

Rules:
- If the user wants a SUMMARY or overview → hand off to Summarization Agent
- If the user wants to COMPARE or contrast documents → hand off to Comparator Agent
- For any specific QUESTION or information retrieval → hand off to RAG Agent

Always hand off. Never answer directly yourself.""",
    handoffs=[rag_agent, summary_agent, comparator_agent],
    model="o3-mini"
)


def run_sdk_planner(query: str) -> dict:
    print(f"[sdk_planner] Query: {query}")

    async def _run():
        result = await Runner.run(planner_agent, input=query)
        return result

    result = asyncio.run(_run())

    answer = result.final_output if hasattr(result, "final_output") else str(result)

    agent_chain = ["Planner"]
    if hasattr(result, "raw_responses"):
        for resp in result.raw_responses:
            if hasattr(resp, "agent") and resp.agent.name not in agent_chain:
                agent_chain.append(resp.agent.name)

    print(f"[sdk_planner] Chain: {agent_chain}")
    return {
        "answer": answer,
        "agent_chain": agent_chain,
        "evidence": []
    }
