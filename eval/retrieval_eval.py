import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from emotion_classifier import EmotionClassifier
import faiss

INDEX_FILE = "../faiss.index"
DATA_FILE = "data/knowledge.json"
EVAL_QUERIES_FILE = "data/eval_queries.json"
TOP_K = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
emotion_model = EmotionClassifier()
faiss_index = faiss.read_index(INDEX_FILE)

with open(DATA_FILE, "r", encoding="utf-8") as f:
    corpus = json.load(f)
with open(EVAL_QUERIES_FILE, "r", encoding="utf-8") as f:
    eval_queries = json.load(f)


def evaluate_baseline():
    hit = 0
    total_sim = 0
    results = []

    for item in eval_queries:
        query = item["query"]
        label_idx = item["conv_id"]

        query_vec = embed_model.encode([query]).astype("float32")
        D, I = faiss_index.search(query_vec, TOP_K)
        retrieved = [corpus[i] for i in I[0]]

        q_emotion, q_dist = emotion_model.predict(query, return_distribution=True)
        q_vec = emotion_model.get_emotion_vector_eval(q_dist)

        matched = False
        sims = []
        for entry in retrieved:
            e_dist = emotion_model.predict(entry["query"], return_distribution=True)[1]
            e_vec = emotion_model.get_emotion_vector_eval(e_dist)
            sim = cosine_similarity([q_vec], [e_vec])[0][0]
            sims.append(sim)

            if entry.get("conv_id") == label_idx:
                matched = True

        results.append({
            "query": query,
            "emotion": q_emotion,
            "hit@5": matched,
            "avg_emotion_sim": round(np.mean(sims), 3)
        })

        if matched:
            hit += 1
        total_sim += np.mean(sims)

    print("\n=== Baseline RAG ===")
    print(f"Hit@5: {hit / len(eval_queries):.4f}")
    print(f"Avg Emotion Similarity: {total_sim / len(eval_queries):.4f}")
    return results


def evaluate_with_rerank():
    hit_count = 0
    total_emotion_sim = 0.0
    results = []

    for query_item in eval_queries:
        query_text = query_item["query"]
        gt_conv_id = query_item["conv_id"]

        query_emotion, query_dist = emotion_model.predict(query_text, return_distribution=True)
        query_vec = embed_model.encode([query_text]).astype("float32")
        query_emotion_vec = emotion_model.get_emotion_vector_eval(query_dist)

        D, I = faiss_index.search(query_vec, TOP_K)
        retrieved = [corpus[i] for i in I[0]]

        reranked = []
        for entry in retrieved:
            entry_dist = emotion_model.predict(entry["query"], return_distribution=True)[1]
            entry_vec = emotion_model.get_emotion_vector_eval(entry_dist)
            sim = cosine_similarity([query_emotion_vec], [entry_vec])[0][0]
            reranked.append((sim, entry))

        reranked.sort(reverse=True, key=lambda x: x[0])
        top_docs = [x[1] for x in reranked[:TOP_K]]

        matched = any(doc.get("conv_id") == gt_conv_id for doc in top_docs)
        avg_sim = np.mean([x[0] for x in reranked[:TOP_K]])
        total_emotion_sim += avg_sim
        if matched:
            hit_count += 1

        results.append({
            "query": query_text,
            "emotion": query_emotion,
            "hit@5": matched,
            "avg_emotion_sim": round(avg_sim, 3)
        })

    print(f"\n[RERANK] Hit@{TOP_K}: {hit_count / len(eval_queries):.4f}")
    print(f"[RERANK] Avg Emotion Similarity: {total_emotion_sim / len(eval_queries):.4f}")
    return results


def summarize(results, name="Model"):
    hit = sum(1 for r in results if r["hit@5"])
    avg_sim = np.mean([r["avg_emotion_sim"] for r in results])
    print(f"\n======== {name} Evaluation ========")
    print(f"Hit@5: {hit / len(results):.4f}")
    print(f"Avg Emotion Similarity: {avg_sim:.4f}")


if __name__ == "__main__":
    print("\n[BASELINE] Evaluating without rerank...")
    baseline_results = evaluate_baseline()

    print("\n[IMPROVED] Evaluating with emotion rerank...")
    rerank_results = evaluate_with_rerank()

    # summarize(baseline_results, "Baseline")
    summarize(rerank_results, "Emotion Rerank")

    with open("retrieval_eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            # "baseline": baseline_results,
            "emotion_rerank": rerank_results
        }, f, indent=2, ensure_ascii=False)