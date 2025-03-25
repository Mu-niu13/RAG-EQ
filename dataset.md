## 1. 情绪识别和表达类数据
Emotion Datasets

GoEmotions (Google)
58种情绪标签，涵盖微妙细腻的情绪表达，适合用来训练情绪识别或理解情绪语境。

链接：GoEmotions GitHub https://github.com/google-research/google-research/tree/master/goemotions

ISEAR Dataset (International Survey on Emotion Antecedents and Reactions)
采集了人们对情绪（愤怒、恐惧、喜悦等）的个人描述，适合生成情绪反应或者共情型回复。

## 2. 共情和对话数据
EmpatheticDialogues (Facebook AI)
2.5万个基于情绪标签的人类对话，设计初衷就是为了训练能理解和回应情绪的对话模型。
特别适合做共情回应或EQ场景分析。

链接：EmpatheticDialogues https://github.com/facebookresearch/EmpatheticDialogues

CounselChat Dataset
收集自心理咨询网站上的问答数据，内容多与心理支持和共情有关，适合用于训练具有情感支持能力的模型。
（需要注意版权和隐私问题，可能要做脱敏处理）

## 3. 人际沟通与社交情境数据
DailyDialog Dataset
包含多种日常对话，注重礼貌性、劝解、安慰等人际交往技能的展现。
能帮助训练对不同社交情境作出合适回应的能力。

SEMAINE Dataset
多模态情感交互数据，虽然偏向音视频，但文本也很有价值，展示了细腻的人际情感交流。

## Insights

如果你做 多轮对话RAG，可以考虑在 EmpatheticDialogues 上加场景标签和用户意图识别
如果偏职场情商，可以引入 DISC行为模式、非暴力沟通（NVC） 等模型
加入心理学理论，比如 Plutchik’s Wheel of Emotions，帮助生成细腻情绪标签检索