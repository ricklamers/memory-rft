# QA Pair Generation for Lex Fridman Transcripts

This directory contains scripts to generate high-quality question-answer pairs from Lex Fridman podcast transcripts using Google Gemini 2.5 Pro.

## Prerequisites

1. **Google API Key**: You need a Google API key with access to Gemini models. Add it to your `.env` file:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

2. **OpenAI API Key**: For embedding generation (already required for other scripts):
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Install dependencies**:
   ```bash
   uv add google-generativeai
   # or update all dependencies
   uv sync
   ```

## Scripts Overview

### 1. `generate_qa_pairs.py`

Generates QA pairs from transcript JSON files using Google Gemini 2.5 Pro with automatic LanceDB ingestion.

**Key Features:**
- Preprocesses transcripts to remove JSON structure and timestamps
- Uses Gemini 2.5 Pro's large context window to process entire transcripts
- Generates questions that require synthesis from multiple parts of the transcript
- Configurable number of QA pairs per episode (default: 20)
- **Streaming output**: See tokens as they're generated from Gemini
- **Logging**: All output saved to log files for monitoring
- **Continuation support**: Resume from where you left off if interrupted
- **Automatic LanceDB ingestion**: No need to run a separate script

**Usage:**
```bash
# Generate QA pairs for all episodes with automatic ingestion
python dataset_utils/generate_qa_pairs.py

# Continue from where you left off (skip already processed episodes)
python dataset_utils/generate_qa_pairs.py --continue

# Generate with custom settings
python dataset_utils/generate_qa_pairs.py \
  --input-dir dataset/lex_fridman_transcripts \
  --output-dir dataset/lex_fridman_qa_pairs \
  --num-pairs 30 \
  --continue

# Process specific episodes
python dataset_utils/generate_qa_pairs.py \
  --episodes "elon-musk" "andrew-huberman" \
  --num-pairs 25

# Limit number of episodes (for testing)
python dataset_utils/generate_qa_pairs.py \
  --max-episodes 5

# Skip LanceDB ingestion
python dataset_utils/generate_qa_pairs.py --no-ingest
```

**Output Structure:**
- QA JSON files: `dataset/lex_fridman_qa_pairs/{episode_name}_qa.json`
- Log files: `dataset/lex_fridman_qa_pairs/logs/`
  - `qa_generation_*.log`: Main processing logs
  - `model_stream_*.log`: Streaming output from Gemini

Each episode generates a `{episode_name}_qa.json` file containing:
```json
{
  "episode": "Episode title",
  "episode_file": "episode-name",
  "speakers": ["Speaker 1", "Speaker 2"],
  "num_qa_pairs": 20,
  "generated_at": "2024-01-01T00:00:00",
  "qa_pairs": [
    {
      "question": "Complex question requiring synthesis...",
      "answer": "Comprehensive answer...",
      "difficulty": "hard",
      "requires_synthesis": true
    }
  ]
}
```

### 2. `monitor_qa_logs.py`

Monitor QA generation progress in real-time by following the log files.

**Usage:**
```bash
# Monitor both main and streaming logs
python dataset_utils/monitor_qa_logs.py

# Monitor only main logs (skip model streaming)
python dataset_utils/monitor_qa_logs.py --no-stream

# Show last 50 lines when starting
python dataset_utils/monitor_qa_logs.py --tail 50
```

### 3. `ingest_qa_to_lancedb.py` (Now integrated into main script)

This functionality is now integrated into `generate_qa_pairs.py` and runs automatically after QA generation. You can still run it separately if needed:

```bash
# Manual ingestion (if you used --no-ingest earlier)
python dataset_utils/ingest_qa_to_lancedb.py
```

## Database Schema

Each record in LanceDB contains:
- `type`: "qa_question" or "qa_answer"
- `episode`: Full episode title
- `episode_file`: Episode filename
- `qa_index`: Index of QA pair within episode
- `text`: Question or answer text
- `answer`/`question`: Corresponding answer/question
- `difficulty`: Question difficulty level
- `requires_synthesis`: Whether multiple parts needed
- `vector`: OpenAI embedding (1536 dimensions)

## QA Generation Strategy

The system is designed to create challenging QA pairs that test deep understanding:

1. **Multi-part Synthesis**: Questions require information from different parts of the conversation
2. **Thematic Connections**: Focus on how topics relate and evolve throughout the episode
3. **Complex Reasoning**: Questions that require understanding context and drawing conclusions
4. **Comprehensive Answers**: Detailed responses that reference multiple discussion points

## Example QA Types

1. **Evolution of Ideas**: "How does the guest's perspective on X change throughout the conversation?"
2. **Topic Connections**: "What connections does the guest draw between their personal experiences and broader societal issues?"
3. **Synthesis**: "Based on the discussion of A, B, and C, what overall philosophy emerges?"
4. **Contextual Understanding**: "How do the anecdotes shared relate to the main thesis being discussed?"

## Performance Considerations

- **Gemini API**: Very generous rate limits, but we add 2-second delays between requests
- **Context Window**: Gemini 2.5 Pro can handle entire transcripts (most are under 1M tokens)
- **Embedding Generation**: Batch processing (50 texts per API call) for efficiency
- **Storage**: QA pairs add relatively little storage compared to full transcripts
- **Continuation**: If interrupted, use `--continue` to resume without reprocessing

## Monitoring Progress

The system provides multiple ways to monitor progress:

1. **Console output**: Real-time progress updates
2. **Log files**: Detailed logs saved to `logs/` directory
3. **Streaming monitor**: Use `monitor_qa_logs.py` to watch generation in real-time
4. **Summary file**: `generation_summary.json` tracks all processed episodes

## Integration with Existing System

The QA pairs are automatically integrated into the LanceDB vector search system:

```python
# Search for relevant questions
import lancedb
import openai

db = lancedb.connect("dataset/lex_fridman_vectordb")
table = db.open_table("lex_fridman_qa")

# Create query embedding
query = "How does technology impact human connection?"
response = openai.embeddings.create(input=query, model="text-embedding-3-small")
embedding = response.data[0].embedding

# Search
results = table.search(embedding).limit(10).to_pandas()

# Results include both questions and their answers
for _, result in results.iterrows():
    if result['type'] == 'qa_question':
        print(f"Q: {result['text']}")
        print(f"A: {result['answer']}")
        print(f"Episode: {result['episode_file']}")
        print()
```

This enables building advanced applications like:
- Question-answering systems
- Study guides for episodes
- Thematic analysis tools
- Content recommendation based on complex queries

## Troubleshooting

1. **Out of memory**: Reduce `--batch-size` for embedding generation
2. **API errors**: Check your API keys and rate limits
3. **Interrupted processing**: Use `--continue` to resume
4. **LanceDB errors**: Ensure the database path exists and is writable 