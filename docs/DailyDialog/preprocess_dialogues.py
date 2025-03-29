import json
import os

# Mapping from emotion ID to emotion label
emotion_map = {
    0: "no emotion",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise"
}

# Mapping from dialogue act ID to act label
act_map = {
    1: "inform",
    2: "question",
    3: "directive",
    4: "commissive"
}

# Mapping from topic ID to topic label
topic_map = {
    1: "Ordinary Life",
    2: "School Life",
    3: "Culture & Education",
    4: "Attitude & Emotion",
    5: "Relationship",
    6: "Tourism",
    7: "Health",
    8: "Work",
    9: "Politics",
    10: "Finance"
}

# Get the current script directory
base_dir = os.path.dirname(__file__)

# Initialize unique ID for each sample
unique_id = 0

# Open all input and output files
with open(os.path.join(base_dir, 'dialogues_text.txt'), 'r', encoding='utf-8') as f_text, \
     open(os.path.join(base_dir, 'dialogues_emotion.txt'), 'r', encoding='utf-8') as f_emotion, \
     open(os.path.join(base_dir, 'dialogues_act.txt'), 'r', encoding='utf-8') as f_act, \
     open(os.path.join(base_dir, 'dialogues_topic.txt'), 'r', encoding='utf-8') as f_topic, \
     open(os.path.join(base_dir, '../clean/cleaned_dialogues.jsonl'), 'w', encoding='utf-8') as f_out:

    # Iterate through all lines in the dataset
    for text_line, emo_line, act_line, topic_line in zip(f_text, f_emotion, f_act, f_topic):
        utterances = [u.strip() for u in text_line.strip().split('__eou__') if u.strip()]
        emotions = [int(e) for e in emo_line.strip().split()]
        acts = [int(a) for a in act_line.strip().split()]
        topic_id = int(topic_line.strip())
        topic = topic_map.get(topic_id, "Unknown")

        # Skip inconsistent entries
        if len(utterances) != len(emotions) or len(utterances) != len(acts):
            continue

        n = len(utterances)
        i = 0

        while i < n:
            if acts[i] == 2:  # Detect a question
                query_emotion_id = emotions[i]
                query_emotion = emotion_map[query_emotion_id]

                # Collect previous utterances with same emotion and act == inform
                query_indices = [i]
                j = i - 1
                while j >= 0 and emotions[j] == query_emotion_id and acts[j] == 1:
                    query_indices.insert(0, j)
                    j -= 1

                # Collect following utterances with same emotion, act != question, and no '?' ending
                response_indices = []
                response_emotion_id = None
                k = i + 1
                while k < n and not utterances[k].endswith("?") and acts[k] != 2:
                    if response_emotion_id is None:
                        response_emotion_id = emotions[k]
                    if emotions[k] == response_emotion_id:
                        response_indices.append(k)
                        k += 1
                    else:
                        break

                if response_indices:
                    sample = {
                        "id": str(unique_id),
                        "topic": topic,
                        "query": " ".join([utterances[idx] for idx in query_indices]),
                        "query_emotion": query_emotion,
                        "response": " ".join([utterances[idx] for idx in response_indices]),
                        "response_emotion": emotion_map[response_emotion_id]
                    }
                    unique_id += 1

                    # Write sample to output JSONL file
                    f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")

                i = k  # Skip processed lines
            else:
                i += 1