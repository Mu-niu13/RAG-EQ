import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from emotion_classifier import EmotionClassifier
from plutchik_engine import related_emotions
from llama_cpp import Llama
import json
import torch
from sklearn.metrics.pairwise import cosine_similarity

# setting
INDEX_FILE = 'faiss.index'
DATA_FILE = 'data/empathetic_dialogues.json'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# init
print(f"[Embedding] SentenceTransformer loaded on {device}")
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)
EMOTION_MODEL = EmotionClassifier()
LLM = Llama(model_path="models/llama-2-7b-chat.Q4_K_M.gguf", n_ctx=2048, n_threads=4)
FAISS_INDEX = faiss.read_index(INDEX_FILE)

# helper 情绪向量化函数（基于 EmotionClassifier 概率输出）
def get_emotion_vector(emotion_dist, labels):
    return np.array([emotion_dist.get(label, 0.0) for label in labels])


# two-stage rag main pipeline
def rag_pipeline(user_query, top_k=5):
    # Stage 1
    # Emotion classification
    primary_emotion, emotion_dist = EMOTION_MODEL.predict(user_query, return_distribution=True)
    related = related_emotions(primary_emotion)
    print(f"[Emotion] Primary: {primary_emotion}, Related: {related}")

    # FAISS dense retrieval
    query_vec = EMBED_MODEL.encode([user_query]).astype('float32')
    D, I = FAISS_INDEX.search(query_vec, top_k)

    # Load original texts
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        dialogues = json.load(f)
    candidates = [dialogues[i] for i in I[0]]

    # Stage 2
    # Emotion rerank
    user_emotion_vec = get_emotion_vector(emotion_dist, EMOTION_MODEL.emotion_labels)
    scored = []
    for entry in candidates:
        entry_dist = EMOTION_MODEL.predict(entry['utterance'], return_distribution=True)[1]
        entry_vec = get_emotion_vector(entry_dist, EMOTION_MODEL.emotion_labels)
        sim = cosine_similarity([user_emotion_vec], [entry_vec])[0][0]
        scored.append((sim, entry))
    scored.sort(reverse=True, key=lambda x: x[0])
    top_docs = [x[1] for x in scored[:3]]

    # prompt
    context = "\n".join([f"- {doc['utterance']}" for doc in top_docs])
    # 情绪陪伴
#     prompt = f"""You are an empathetic emotional support assistant.
# The user currently feels {primary_emotion}.
# Your goal is to guide the user toward feeling {', '.join(related)} by offering a supportive and helpful response.
#
# Relevant past responses:
# {context}
#
# User's concern:
# {user_query}
#
# Assistant:"""
    # 高情商
    prompt = f"""You are an expert in high emotional intelligence and workplace diplomacy.
Your role is not just to comfort the user, but to provide tactful, thoughtful, and strategic suggestions for handling sensitive interpersonal situations.

The user currently feels {primary_emotion}. Your goal is to guide them toward feeling {', '.join(related)} by offering practical advice that balances emotional sensitivity and social strategy.

Here are examples of how others handled similar situations:
{context}

Now, please respond to the user's concern:
{user_query}

Assistant:"""

    print(f"[Prompt Ready] Final prompt:\n{prompt}")

    # LLM response
    output = LLM(prompt, max_tokens=256)
    print("Raw output:", print(output))
    return output['choices'][0]['text'].strip()