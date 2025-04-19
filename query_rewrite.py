# from emotion_classifier import EmotionClassifier
# from llama_cpp import Llama

def rewrite_query_with_emotion(LLM, EMOTION_MODEL, original_query: str) -> str:
    primary_emotion = EMOTION_MODEL.predict(original_query)

    prompt = f"""
You are a high-EQ assistant. Your job is to rewrite emotionally charged user messages into clear, constructive questions that are emotionally balanced but still expressive.

The user currently feels: {primary_emotion}

Original query:
"{original_query}"

Please rewrite this message as a clear, calm, and constructive question for advice retrieval:
Rewritten:
""".strip()

    output = LLM(prompt, max_tokens=80)
    rewritten = output['choices'][0]['text'].strip().strip('"')
    return rewritten

