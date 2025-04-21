# Plutchik’s Wheel of Emotions
emotion_map = {
    "anger": ["calm", "confidence", "understanding"],
    "fear": ["reassurance", "trust", "safety"],
    "sadness": ["hope", "support", "companionship"],
    "frustration": ["validation", "clarity", "empathy"],
    "nervousness": ["reassurance", "calm", "encouragement"],
    "disappointment": ["encouragement", "optimism", "motivation"],
    "neutral": ["helpfulness", "respect", "curiosity"],

    "joy": ["gratitude", "celebration", "connection"],
    "gratitude": ["warmth", "reciprocity", "acknowledgment"],
    "love": ["connection", "safety", "trust"],
    "admiration": ["respect", "aspiration", "recognition"],

    "confusion": ["clarity", "patience", "guidance"],
    "embarrassment": ["acceptance", "forgiveness", "relief"],
    "disgust": ["distance", "boundaries", "calm"],
    "annoyance": ["patience", "perspective", "relaxation"],
    "realization": ["reflection", "clarity", "action"],
    "approval": ["recognition", "encouragement", "confidence"],

    "curiosity": ["exploration", "clarification", "discovery"],
    "remorse": ["forgiveness", "healing", "growth"],
    "optimism": ["momentum", "positivity", "focus"],
    "caring": ["compassion", "attentiveness", "reliability"],

    "surprise": ["adaptability", "acceptance", "preparation"],
    "amusement": ["connection", "joy", "engagement"],
    "pride": ["humility", "confidence", "inspiration"],
    "grief": ["solace", "memory", "companionship"]
}


def related_emotions(emotion):
    emotion = emotion.lower()
    if emotion in emotion_map:
        return emotion_map[emotion]
    else:
        print(f"[PlutchikEngine] Unknown emotion: {emotion} — fallback to defaults.")
        return ["support", "understanding", "calm"]