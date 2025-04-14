emotion_map = {
    "anger": ["calm", "confidence"],
    "fear": ["reassurance", "trust"],
    "sadness": ["hope", "support"],
    "frustration": ["validation", "clarity", "empathy"],
    "nervousness": ["reassurance", "calm"],
    "disappointment": ["encouragement", "optimism"],
    "neutral": ["helpfulness", "respect"],
    #TODO
}

def related_emotions(emotion):
    emotion = emotion.lower()
    if emotion in emotion_map:
        return emotion_map[emotion]
    else:
        print(f"[PlutchikEngine] Unknown emotion: {emotion} — fallback to defaults.")
        return ["support", "understanding", "calm"]