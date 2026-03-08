def test_full_pipeline_combinatorics():
    print("\n--- Testing Full Pipeline Combinatorics ---")
    from agents.solver_agent import run_solver_agent
    
    parsed = {
        "problem_text": "What is the number of ways to arrange the letters in the word 'MATHEMATICS'?",
        "topic": "permutations_combinations",
        "variables": [],
        "constraints": []
    }
    route = {
        "detected_topic": "permutations_combinations",
        "solver_mode": "analytical"
    }
    retrieved_docs = [
        {"source": "probability_statistics.txt", "content": "PERMUTATIONS === P(n, r) = nPr = n! / (n-r)!"}
    ]
    
    result = run_solver_agent(parsed, route, retrieved_docs)
    print(f"Method Used: {result.get('method_used')}")
    print(f"Final Answer: {result.get('final_answer')}")
    print("Solution Snippet:")
    print(result.get('solution')[:500])

if __name__ == "__main__":
    # test_sympy_parsing()
    # test_gemini_api()
    # test_combinatorics()
    test_full_pipeline_combinatorics()
