from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class EmotionClassifier:
    def __init__(self):
        self.model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = torch.argmax(probs).item()
        labels = self.model.config.id2label
        return labels[label]

# 测试
# clf = EmotionClassifier()
# print(clf.predict("I'm feeling anxious."))
