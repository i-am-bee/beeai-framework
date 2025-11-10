from datasets import load_dataset
import pandas as pd


dataset = load_dataset("hotpot_qa", "distractor", split="train")
print(dataset)

df = pd.DataFrame(dataset)
print(df['type'].value_counts())
print(df['level'].value_counts())


# קיבוץ לפי type ו-level
grouped = df.groupby(['type', 'level'])

# כמה דוגמאות לקחת מכל קבוצה
samples_per_group = max(1, 50 // len(grouped))

# דגימה
sampled_df = grouped.apply(lambda x: x.sample(n=min(samples_per_group, len(x)), random_state=42)).reset_index(drop=True)

remaining = 50 - len(sampled_df)
if remaining > 0:
    extra = df.sample(n=remaining, random_state=42)
    sampled_df = pd.concat([sampled_df, extra])

sampled_df[['question', 'answer', 'type', 'level', 'context']].to_json("evaluation_dataset.json", orient="records", indent=2)

import json
with open("evaluation_dataset.json") as f:
    data = json.load(f)

print(len(data))  # אמור להיות 50
print({d['type'] for d in data})
print({d['level'] for d in data})



