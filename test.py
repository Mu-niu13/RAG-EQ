from rag_engine import rag_pipeline

if __name__ == "__main__":
    print("Start single turn RAG testing")
    print("-" * 50)
    while True:
        user_query = input("Query (or 'exit'): ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break

        response = rag_pipeline(user_query)
        print("\nResponse:")
        print(response)
        print("-" * 50)
