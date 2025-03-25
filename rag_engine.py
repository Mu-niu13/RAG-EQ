import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from emotion_classifier import EmotionClassifier
from plutchik_engine import related_emotions
from llama_cpp import Llama
import json

INDEX_FILE = 'faiss.index'
DATA_FILE = 'data/empathetic_dialogues.json'

# 初始化模型
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print(f"[Query Embedding] Model loaded on {device}")
EMOTION_MODEL = EmotionClassifier()
# EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
LLM = Llama(model_path="models/llama-2-7b-chat.Q4_K_M.gguf", n_ctx=2048, n_threads=4)

def rag_pipeline(user_query, top_k=3):
    # 1. 情绪识别
    primary_emotion = EMOTION_MODEL.predict(user_query)
    related = related_emotions(primary_emotion)

    # 2. 检索数据
    index = faiss.read_index(INDEX_FILE)
    query_vec = EMBED_MODEL.encode([user_query]).astype('float32')
    D, I = index.search(query_vec, top_k)

    # 3. 读取检索文档
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        dialogues = json.load(f)

    retrieved_docs = [dialogues[i]['utterance'] for i in I[0]]
    context = "\n".join([f"- {doc}" for doc in retrieved_docs])

    # 4. Prompt 构建
    prompt = f"""
    You are an empathetic emotional support assistant.
    The user feels {primary_emotion}.
    Provide support that encourages {', '.join(related)} emotions.
    
    Context examples from previous dialogues:
    {context}
    
    Now, answer to the user's question:
    "{user_query}"
    """
    print(f"[RAG Pipeline] Final Prompt:\n{prompt}")

    # 5. Llama 生成
    output = LLM(prompt, max_tokens=256)
    response = output['choices'][0]['text'].strip()
    return response
