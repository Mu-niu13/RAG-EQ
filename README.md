# RAG-EQ: Retrieval-Augmented Generation - Emotional Intelligence

**Authors:** Hongyi Duan, Kaixin Lu, Mu Niu  
**Institution:** Duke University  
**Project Advisors:** Dr. Hai Li and TAs  

## Overview

**RAG-EQ** is a Retrieval-Augmented Generation (RAG) system designed to enhance emotionally intelligent communication in everyday and professional contexts. It combines a vector-based retrieval system with large language models (LLMs) to deliver responses that are not only contextually relevant but also empathetic and emotionally attuned.

---

## Motivation

Emotionally intelligent responses are critical in:

- Professional communication  
- Conflict resolution  
- Daily interpersonal interactions  

However, people often struggle to express themselves with empathy and clarity. RAG-EQ addresses this gap by grounding responses in a curated knowledge base of high emotional intelligence (EI) examples.

---

## System Architecture

The RAG-EQ pipeline includes:

1. **Emotion Classification** of user queries  
2. **Query Vectorization** for semantic matching  
3. **Vector-based Knowledge Base** built from:
   - LinkedIn, Reddit, Google articles
   - GPT-generated high-EI responses  
4. **Semantic and Emotion-based Re-ranking** to retrieve the most emotionally relevant responses  
5. **Prompt Construction and LLM Generation** using LLaMA-2-7b with query expansion

---

## Data Structure

Each entry in the dataset follows a structured JSON format:
```json
{
  "topic": "relationship advice",
  "query": "How can I navigate cultural differences with my parents?",
  "emotion_q": "curious",
  "response": "It's important to approach with empathy and curiosity. Acknowledge your parents' values and gently express your perspective...",
  "emotion_r": "supportive"
}
```

---

## Evaluation

Performance was assessed using both **human ratings** and **ChatGPT-4o evaluations** across different contexts:

- **School**: Largest improvement observed and reduced variance compared to baseline
- **Work**: Most consistent performance  
- **Daily Life**

### Scoring Rubric (1–5 Scale):

- **5** – Excellent emotional alignment and clarity  
- **4** – Good with minor tone/structure issues  
- **3** – Fair, partially addresses emotional needs  
- **2** – Poor, limited relevance or empathy  
- **1** – Very Poor, irrelevant or inappropriate  

---

## Results

- RAG-EQ **outperforms LLaMA-2-7b baseline** in both emotional relevance and clarity  
- Demonstrated value of **emotion-driven re-ranking** and **contextual grounding**  
- Showcases the **synergy between retrieval systems and generative LLMs**

---

## References

1. Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 33*  
2. Meta AI. (2023). LLaMA 2: Open Foundation and Fine-Tuned Chat Models.

---

## Acknowledgments

Special thanks to Dr. Hai Li and the Teaching Assistants at Duke University for their continuous support and guidance throughout the development of this project.
