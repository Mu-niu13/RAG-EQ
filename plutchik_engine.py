# 简化 Plutchik 相关情绪模型
PLUTCHIK_RELATIONS = {
    "Fear": ["Trust", "Anticipation"],
    "Sadness": ["Joy", "Trust"],
    "Anger": ["Fear", "Anticipation"],
    "Joy": ["Trust", "Anticipation"],
    "Trust": ["Joy", "Fear"],
    # ...
}

def related_emotions(emotion):
    return PLUTCHIK_RELATIONS.get(emotion, [])
