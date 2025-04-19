from emotion_classifier import EmotionClassifier
from llama_cpp import Llama

EMOTION_MODEL = EmotionClassifier()
LLM = None

def init_llama():
    global LLM
    if LLM is None:
        LLM = Llama(model_path="models/llama-2-7b-chat.Q4_K_M.gguf", n_ctx=1024, n_threads=4)


def rewrite_query_with_emotion(original_query: str) -> str:
    init_llama()
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


if __name__ == "__main__":
    user_input = "I’m so mad at my coworker. They keep ignoring everything I say!"
    rewritten = rewrite_query_with_emotion(user_input)
    print(f"[Original] {user_input}\n[Rewritten] {rewritten}")