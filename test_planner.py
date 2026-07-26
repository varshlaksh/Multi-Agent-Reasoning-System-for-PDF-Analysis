from agents.planner import run_planner

queries = [
    "What are the main requirements of this system?",
    "Give me a summary of the document",
    "Compare the different agents described"
]

for q in queries:
    print("\n" + "="*60)
    print(f"QUERY: {q}")
    print("="*60)
    result = run_planner(q)
    print(f"Intent:      {result['intent']}")
    print(f"Agent chain: {result['agent_chain']}")
    print(f"Answer:\n{result['answer'][:300]}...")
