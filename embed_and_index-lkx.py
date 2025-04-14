from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

# cpu
# MODEL = SentenceTransformer('all-MiniLM-L6-v2')
# INDEX_FILE = 'faiss.index'
#
#
# def build_index():
#     with open('data/empathetic_dialogues.json', 'r') as f:
#         data = json.load(f)
#     texts = [entry['utterance'] for entry in data]
#     embeddings = MODEL.encode(texts)
#
#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(np.array(embeddings))
#
#     faiss.write_index(index, INDEX_FILE)
#     print(f"Indexed {len(data)} documents.")

# gpu
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)

print(f"[Embedding] Model loaded on {device}")

def build_index():
    with open('data/empathetic_dialogues.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = [item['utterance'] for item in data]

    # ➤ 增加 batch_size，充分利用 GPU 性能
    embeddings = MODEL.encode(texts, batch_size=64, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, 'faiss.index')

    print(f"[FAISS] Indexed {len(texts)} entries.")

build_index()
