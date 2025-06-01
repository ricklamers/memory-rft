import pandas as pd
import json

# Read the simple format dataset
df = pd.read_parquet('data/podcast_questions_mini.parquet')

# Convert to verl format
verl_data = []
for idx, row in df.iterrows():
    data = {
        "data_source": "podcast_qa",
        "prompt": [
            {
                "role": "user", 
                "content": row['question']
            }
        ],
        "ability": "qa",
        "reward_model": {
            "style": "rule",
            "ground_truth": row['answer']
        },
        "extra_info": {
            "split": "train",
            "index": idx,
            "question": row['question'],
            "answer": row['answer']
        }
    }
    verl_data.append(data)

# Save as parquet
verl_df = pd.DataFrame(verl_data)
verl_df.to_parquet('data/podcast_questions_verl_format.parquet', index=False)

print(f"Converted {len(verl_df)} samples to verl format")
print("\nFirst sample:")
print(json.dumps(verl_data[0], indent=2)) 