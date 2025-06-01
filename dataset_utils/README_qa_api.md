# QA Batch API Documentation

The Lex Fridman Search API now includes endpoints for retrieving shuffled batches of QA pairs, perfect for training and evaluation purposes.

## QA Endpoints

### 1. Get QA Batch
**POST** `/api/qa/batch`

Get a shuffled batch of QA pairs with optional filtering.

**Request Body:**
```json
{
  "batch_size": 32,           // Number of QA pairs to return (default: 32)
  "shuffle": true,            // Whether to shuffle the pairs (default: true)
  "difficulty_filter": null,  // Filter by difficulty: "hard", etc. (optional)
  "episode_filter": null,     // Filter by episode filename (optional)
  "seed": 42                  // Random seed for reproducible shuffling (optional)
}
```

**Response:**
```json
{
  "qa_pairs": [
    {
      "question": "How does the guest's perspective on X change throughout the conversation?",
      "answer": "The guest initially...",
      "episode": "Full episode title",
      "episode_file": "episode-filename",
      "difficulty": "hard",
      "requires_synthesis": true,
      "qa_index": 0
    }
    // ... more QA pairs
  ],
  "batch_size": 32,
  "total_available": 1606,
  "seed_used": 42
}
```

### 2. Get QA Statistics
**GET** `/api/qa/stats`

Get statistics about available QA pairs.

**Response:**
```json
{
  "total_qa_pairs": 1606,
  "difficulty_distribution": {
    "hard": 1606
  },
  "episodes_with_qa": 81,
  "average_qa_per_episode": 19.8,
  "qa_table_name": "lex_fridman_qa"
}
```

### 3. Get Episodes with QA
**GET** `/api/qa/episodes`

Get list of all episodes that have QA pairs.

**Response:**
```json
{
  "episodes": [
    "adam-frank",
    "andrew-huberman-5",
    "bill-ackman",
    // ... more episodes
  ]
}
```

### 4. General API Stats
**GET** `/api/stats`

Includes QA information in general stats.

**Response:**
```json
{
  "total_transcripts": 52827,
  "total_episodes": 98,
  "total_speakers": 150,
  "database_path": "dataset/lex_fridman_vectordb",
  "semantic_search_available": true,
  "qa_pairs_available": true,
  "total_qa_pairs": 1606
}
```

## Usage Examples

### Python Example for Training Loop
```python
import requests
import random

API_URL = "http://localhost:8000"

def get_training_batch(batch_size=32):
    """Get a random batch of QA pairs for training."""
    response = requests.post(
        f"{API_URL}/api/qa/batch",
        json={
            "batch_size": batch_size,
            "shuffle": True,
            "seed": random.randint(0, 1000000)
        }
    )
    if response.status_code == 200:
        return response.json()["qa_pairs"]
    else:
        raise Exception(f"API error: {response.status_code}")

# Training loop
for epoch in range(10):
    batch = get_training_batch(32)
    for qa in batch:
        question = qa["question"]
        answer = qa["answer"]
        episode = qa["episode_file"]
        # Train your model here...
```

### Filtering Examples

**Get only hard questions:**
```python
response = requests.post(
    f"{API_URL}/api/qa/batch",
    json={
        "batch_size": 50,
        "shuffle": True,
        "difficulty_filter": "hard"
    }
)
```

**Get questions from a specific episode:**
```python
response = requests.post(
    f"{API_URL}/api/qa/batch",
    json={
        "batch_size": 20,
        "shuffle": False,
        "episode_filter": "elon-musk-4"
    }
)
```

**Reproducible batches (same seed = same order):**
```python
response = requests.post(
    f"{API_URL}/api/qa/batch",
    json={
        "batch_size": 100,
        "shuffle": True,
        "seed": 12345  # Use same seed for same results
    }
)
```

## Running the API

1. Start the API server:
   ```bash
   python dataset_utils/api.py
   ```

2. The API will be available at `http://localhost:8000`

3. Test the API:
   ```bash
   python dataset_utils/test_qa_api.py
   ```

## Features

- **Caching**: QA pairs are cached in memory for 5 minutes to improve performance
- **Shuffling**: Random shuffling with optional seed for reproducibility
- **Filtering**: Filter by difficulty level or specific episodes
- **Batch Size**: Configurable batch size (default: 32)
- **Performance**: Fast retrieval with pre-computed embeddings in LanceDB

## Use Cases

1. **Training QA Models**: Get shuffled batches for training question-answering systems
2. **Evaluation Sets**: Create consistent evaluation sets using fixed seeds
3. **Episode Analysis**: Study QA pairs from specific episodes
4. **Difficulty Progression**: Filter by difficulty for curriculum learning
5. **Data Exploration**: Use the stats endpoints to understand the dataset

The QA batch API makes it easy to integrate the Lex Fridman QA dataset into your machine learning workflows! 