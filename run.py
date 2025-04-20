import csv
from rag_engine import rag_pipeline, baseline_pipeline

if __name__ == "__main__":
    output_file = "comparison_results.csv"

    with open("query.txt", "r", encoding="utf-8") as f, \
            open(output_file, "w", encoding="utf-8", newline='') as out_csv:

        writer = csv.writer(out_csv)
        writer.writerow(["Query", "Baseline Response", "RAG Response"])  # header

        for query in f:
            user_query = query.strip()
            if not user_query:
                continue

            try:
                baseline_response = baseline_pipeline(user_query)
            except Exception as e:
                baseline_response = f"[Error] {e}"

            try:
                rag_response = rag_pipeline(user_query)
                print("===============================================================")
                print(rag_response)
            except Exception as e:
                rag_response = f"[Error] {e}"

            writer.writerow([user_query, baseline_response, rag_response])

    print(f"\nAll results saved to {output_file}")
