# from datasets import load_dataset
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# # from add_project_root import add_project_root
# # add_project_root()
# # from emotion_classifier import EmotionClassifier
# import ./emotion_classifier
#
# # Load test set (GoEmotions or EmpatheticDialogues with labels)
# dataset = load_dataset("go_emotions", split="test")
#
# # Emotion classifier instance
# clf = EmotionClassifier()
#
# # Limit to single-label examples only (GoEmotions has multi-label)
# filtered = dataset.filter(lambda x: len(x['labels']) == 1)
#
# # Map ID to label names
# id2label = dataset.features['labels'].feature.names
#
# # Collect predictions and ground truths
# true_labels = []
# pred_labels = []
#
# print("Evaluating on", len(filtered), "samples")
#
# for example in filtered:
#     text = example['text']
#     true = id2label[example['labels'][0]]
#     pred = clf.predict(text)
#     true_labels.append(true)
#     pred_labels.append(pred)
#
# # Print classification report
# report = classification_report(true_labels, pred_labels, zero_division=0)
# print(report)
#
# # Confusion Matrix
# labels = sorted(list(set(true_labels + pred_labels)))
# cm = confusion_matrix(true_labels, pred_labels, labels=labels)
# df_cm = pd.DataFrame(cm, index=labels, columns=labels)
#
# plt.figure(figsize=(12, 10))
# sns.heatmap(df_cm, annot=False, fmt='d', cmap="Blues")
# plt.title("Confusion Matrix - Emotion Classifier")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.tight_layout()
# plt.show()
