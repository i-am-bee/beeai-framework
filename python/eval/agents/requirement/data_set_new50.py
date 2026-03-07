from datasets import load_dataset
import pandas as pd
import json
import os


# Load HotPotQA distractor/train
dataset = load_dataset("hotpot_qa", "distractor", split="train")
df = pd.DataFrame(dataset)

# --- Random sampling of 50 examples ---
sampled_df = df.sample(n=50, random_state=42).reset_index(drop=True)

# --- Function to extract relevant sentences ---
def get_relevant_sentences(context, supporting_facts):
    relevant_sentences = []

    titles = context["title"]
    sentences = context["sentences"]

    fact_titles = supporting_facts["title"]
    fact_ids = supporting_facts["sent_id"]

    # Create a mapping: title -> set of relevant sentence indices
    fact_dict = {}
    for t, idx in zip(fact_titles, fact_ids):
        if t not in fact_dict:
            fact_dict[t] = set()
        fact_dict[t].add(idx)

    # Extract sentences
    for title, sents in zip(titles, sentences):
        if title in fact_dict:
            for idx in fact_dict[title]:
                if idx < len(sents):
                    relevant_sentences.append(sents[idx])

    return relevant_sentences

# --- Create JSON with relevant sentences and WIKI_TIMES ---
new_data = []
for _, row in sampled_df.iterrows():
    question = row["question"]
    answer = row["answer"]
    context = row["context"]
    supporting_facts = row["supporting_facts"]
    relevant_sentences = get_relevant_sentences(context, supporting_facts)

    # Count unique Wikipedia titles in supporting_facts
    wiki_times = len(set(supporting_facts["title"]))

    new_data.append({
        "question": question,
        "answer": answer,
        "relevant_sentences": relevant_sentences,
        "wiki_times": wiki_times,
        "supporting_titles": supporting_facts["title"],
    })

output_path = os.path.join(os.path.dirname(_file_),"evaluation_dataset_50_clean.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(new_data)} examples to {output_path}")