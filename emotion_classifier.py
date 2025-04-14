# emotion_classifier.py
from transformers import pipeline
import numpy as np

class EmotionClassifier:
    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
            top_k=None
        )
        self.emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness",
            "optimism", "pride", "realization", "relief", "remorse",
            "sadness", "surprise", "neutral"
        ]

    def get_emotion_vector(self, text: str) -> np.ndarray:
        scores = self.classifier(text)[0]
        score_dict = {item["label"]: item["score"] for item in scores}
        return np.array([score_dict.get(label, 0.0) for label in self.emotion_labels])

    def predict(self, text: str, return_distribution=False):
        scores = self.classifier(text)[0]
        score_dict = {item["label"]: item["score"] for item in scores}
        top_emotion = max(score_dict, key=score_dict.get)
        return (top_emotion, score_dict) if return_distribution else top_emotion
