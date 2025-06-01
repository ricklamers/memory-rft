import pandas as pd

# Create more dummy samples 
data = []
for i in range(16):  # Create 16 samples to ensure we have enough for 8 GPUs
    data.append({
        'question': f'What did Joe Rogan say about topic {i+1}?',
        'answer': f'Joe Rogan discussed topic {i+1} in great detail during episode {i+100}.'
    })

# Convert to DataFrame and save as parquet
df = pd.DataFrame(data)
df.to_parquet('data/podcast_questions_mini.parquet', index=False)
print(f'Created dataset with {len(df)} samples')
print(df.head()) 