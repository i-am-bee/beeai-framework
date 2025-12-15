from datasets import load_dataset
import pandas as pd
import json
import os

# טוענים את HotPotQA גרסת distractor/train
dataset = load_dataset("hotpot_qa", "distractor", split="train")
df = pd.DataFrame(dataset)
sampled_df = df.head(2)  # רק שתי דוגמאות לדוגמה

def get_relevant_sentences(context, supporting_facts):
    relevant_sentences = []

    titles = context["title"]
    sentences = context["sentences"]

    fact_titles = supporting_facts["title"]
    fact_ids = supporting_facts["sent_id"]

    # יוצרים מילון כותרת → set של אינדקסים רלוונטיים
    fact_dict = {}
    for t, idx in zip(fact_titles, fact_ids):
        if t not in fact_dict:
            fact_dict[t] = set()
        fact_dict[t].add(idx)

    # עוברים על כל מאמר ב-context
    for title, sents in zip(titles, sentences):
        if title in fact_dict:
            for idx in fact_dict[title]:
                if idx < len(sents):
                    relevant_sentences.append(sents[idx])

    return relevant_sentences

# יוצרים JSON חדש
new_data = []
for _, row in sampled_df.iterrows():
    question = row["question"]
    answer = row["answer"]
    context = row["context"]
    supporting_facts = row["supporting_facts"]
    relevant_sentences = get_relevant_sentences(context, supporting_facts)

    # סופרים כמה כותרות ויקיפדיה שונות יש ב-supporting_facts
    wiki_times = len(set(supporting_facts["title"]))

    new_data.append({
        "question": question,
        "answer": answer,
        "relevant_sentences": relevant_sentences,
        "wiki_times": wiki_times,
        "supporting_titles": supporting_facts["title"],

    })

output_path = os.path.join(os.path.dirname(_file_),"evaluation_dataset_2_clean.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(new_data)} examples to {output_path}")