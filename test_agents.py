from agents.rag_agent import run_rag_agent
from agents.summary_agent import run_summary_agent
from agents.comparator import run_comparator_agent

print("=" * 50)
print("TEST 1: RAG Agent")
print("=" * 50)
result = run_rag_agent("what are the main requirements of this assignment?")
print(result["answer"])
print(f"\nEvidence count: {len(result['evidence'])}")
for e in result["evidence"]:
    print(f"  - {e['doc_name']} | Page {e['page_number']} | Score {e['score']}")

print("\n" + "=" * 50)
print("TEST 2: Summary Agent")
print("=" * 50)
result = run_summary_agent()
print(result["answer"])

print("\n" + "=" * 50)
print("TEST 3: Comparator Agent (single doc - should warn)")
print("=" * 50)
rag_result = run_rag_agent("what agents are described?")
comp_result = run_comparator_agent("compare the agents", rag_result)
print(comp_result["answer"])
