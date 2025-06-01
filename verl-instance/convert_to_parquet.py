import pandas as pd
import json

# Read the JSONL file
data = []
with open('data/podcast_questions_mini.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line.strip()))

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as Parquet
df.to_parquet('data/podcast_questions_mini.parquet', index=False)

print(f"Converted {len(data)} records to Parquet format")
print("Columns:", df.columns.tolist())
print("Sample data:")
print(df.head()) 