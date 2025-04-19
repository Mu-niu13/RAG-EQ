# eval acc for retrieval prediction
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from emotion_classifier import EmotionClassifier
import faiss


# Settings
INDEX_FILE = "../faiss.index"
DATA_FILE = "data/empathetic_dialogues.json"
TOP_K = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
emotion_model = EmotionClassifier()
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
index = faiss.read_index(INDEX_FILE)

# Load corpus
with open(DATA_FILE, "r", encoding="utf-8") as f:
    corpus = json.load(f)

# Load evaluation set
# You can change this to another dataset with (query, label_idx) structure
eval_set = load_dataset("empathetic_dialogues", split="test[:500]")

hit_count = 0
emotion_sims = []
total = 0

print("[Eval] Start retrieval evaluation...")

for sample in tqdm(eval_set):
    query = sample["utterance"]
    label_idx = sample["conv_id"] if "conv_id" in sample else None  # optional

    # Emotion vector of query
    query_emotion, dist = emotion_model.predict(query, return_distribution=True)
    query_vec = embed_model.encode([query]).astype("float32")
    query_emotion_vec = np.array([dist.get(label, 0.0) for label in emotion_model.emotion_labels])

    # Retrieval
    D, I = index.search(query_vec, TOP_K)
    retrieved = [corpus[i] for i in I[0]]

    # Emotion similarity
    sims = []
    matched = False
    for entry in retrieved:
        entry_dist = emotion_model.predict(entry["utterance"], return_distribution=True)[1]
        entry_vec = emotion_model.get_emotion_vector_eval(entry_dist)

        sim = cosine_similarity([query_emotion_vec], [entry_vec])[0][0]
        sims.append(sim)

        # Ground-truth hit
        if label_idx is not None and entry.get("conv_id") == label_idx:
            matched = True

    if matched:
        hit_count += 1
    emotion_sims.append(np.mean(sims))
    total += 1

# Results
print("\n========= Retrieval Evaluation =========")
if label_idx is not None:
    print(f"Precision@{TOP_K}: {hit_count / total:.4f}")
print(f"Average Emotion Similarity: {np.mean(emotion_sims):.4f}")
