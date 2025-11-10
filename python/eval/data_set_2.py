from datasets import load_dataset
import pandas as pd
import json

# טוענים את HotPotQA גרסת distractor/train
dataset = load_dataset("hotpot_qa", "distractor", split="train")
print(dataset)

# ממירים ל-DataFrame
df = pd.DataFrame(dataset)

# לוקחים רק את שתי הדוגמאות הראשונות
sampled_df = df.head(2)

# שומרים לקובץ JSON
output_path = "evaluation_dataset_2.json"
sampled_df[['question', 'answer', 'type', 'level', 'context']].to_json(
    output_path, orient="records", indent=2
)

# בדיקה
with open(output_path) as f:
    data = json.load(f)

print(f"\nSaved {len(data)} examples to {output_path}")
print({d['type'] for d in data})
print({d['level'] for d in data})
