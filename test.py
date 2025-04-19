from rag_engine import rag_pipeline

if __name__ == "__main__":
    print("Welcome to Emotion-aware RAG CLI")
    print("-" * 50)
    while True:
        user_query = input("🧑 Your query (or 'exit'): ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break

        response = rag_pipeline(user_query)
        print("\nLLaMA's response:")
        print(response)
        print("-" * 50)
