# from emotion_classifier import EmotionClassifier
# from llama_cpp import Llama
# from plutchik_engine import related_emotions
# from sentence_transformers import SentenceTransformer
# import faiss
# import torch
# import json
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
#
# # init
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)
# EMOTION_MODEL = EmotionClassifier()
# LLM = Llama(model_path="models/llama-2-7b-chat.Q4_K_M.gguf", n_ctx=2048, n_threads=4)
# INDEX_FILE = 'faiss.index'
# DATA_FILE = 'data/empathetic_dialogues.json'
# FAISS_INDEX = faiss.read_index(INDEX_FILE)
#
# with open(DATA_FILE, 'r', encoding='utf-8') as f:
#     CORPUS = json.load(f)
#
# # helper
# def get_emotion_vector(emotion_dist, labels):
#     return np.array([emotion_dist.get(label, 0.0) for label in labels])
#
# def build_conversational_prompt(history, current_user_query, emotion, related, context):
#     lines = []
#     if context:
#         lines.append("Here are similar responses from others:\n" + context)
#     lines.append(f"User is feeling {emotion}. Try to guide them toward feeling {', '.join(related)}.\n")
#     lines.append("Conversation so far:")
#     for turn in history:
#         lines.append(f"{turn['role'].capitalize()}: {turn['content']}")
#     lines.append(f"User: {current_user_query}")
#     lines.append("Assistant:")
#     return "\n".join(lines)
#
# # ===== 主函数：支持多轮对话 =====
# def multi_turn_rag(user_query, history=None, top_k=5):
#     history = history or []
#
#     # emotion detect
#     primary_emotion, emotion_dist = EMOTION_MODEL.predict(user_query, return_distribution=True)
#     related = related_emotions(primary_emotion)
#
#     # search
#     query_vec = EMBED_MODEL.encode([user_query]).astype('float32')
#     D, I = FAISS_INDEX.search(query_vec, top_k)
#     candidates = [CORPUS[i] for i in I[0]]
#
#     # rerank
#     user_emotion_vec = get_emotion_vector(emotion_dist, EMOTION_MODEL.emotion_labels)
#     scored = []
#     for entry in candidates:
#         entry_dist = EMOTION_MODEL.predict(entry['utterance'], return_distribution=True)[1]
#         entry_vec = get_emotion_vector(entry_dist, EMOTION_MODEL.emotion_labels)
#         sim = cosine_similarity([user_emotion_vec], [entry_vec])[0][0]
#         scored.append((sim, entry))
#     scored.sort(reverse=True, key=lambda x: x[0])
#     top_docs = [x[1] for x in scored[:3]]
#     context = "\n".join([f"- {doc['utterance']}" for doc in top_docs])
#
#     # prompt construct
#     prompt = build_conversational_prompt(history, user_query, primary_emotion, related, context)
#
#     # LLM response
#     output = LLM(prompt, max_tokens=256)
#     response = output['choices'][0]['text'].strip()
#
#     # update history
#     history.append({"role": "user", "content": user_query})
#     history.append({"role": "assistant", "content": response})
#
#     return response, history