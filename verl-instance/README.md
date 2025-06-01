# Podcast Memory Retrieval with veRL and SGLang

This project implements a reinforcement learning system that trains a language model to answer questions about podcast content using a memory retrieval tool. The model learns to call a retrieval function during its chain-of-thought reasoning to fetch relevant transcript chunks.

## Architecture

- **Model**: DeepSeek-R1-0528-Qwen3-8B (8B parameter reasoning model)
- **RL Framework**: veRL with PPO algorithm
- **Inference Engine**: SGLang for efficient serving
- **Memory Retrieval**: Mock retriever (can be replaced with real vector DB)
- **Reward**: LLM judge evaluating semantic correctness

## Components

### 1. Memory Tool (`memory_tool.py`)
- Mock retriever returning fake podcast transcript chunks
- Replace with real FAISS/LanceDB implementation for production

### 2. CoT Tool Wrapper (`cot_tool_sglang.py`)
- Handles streaming generation with SGLang
- Detects `#call memory_retrieval {...}` in `<think>` blocks
- Injects retrieval results back into reasoning

### 3. Reward Function (`podcast_reward.py`)
- Uses LLM judge to score semantic correctness
- Accepts variants (e.g., "Japan" for "country with capital Tokyo")
- Optional tool usage penalty

### 4. Environment (`verl_env.py`)
- Integrates agent, retriever, and reward function
- Single-turn Q&A episodes

## Setup

### Prerequisites

1. **In Docker container (verl)**:
   ```bash
   cd /workspace/verl
   pip install -e .[sglang]
   ```

2. **Start SGLang server** (in Docker container):
   ```bash
   python -m sglang.launch_server \
     --model-path deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
     --port 30000 \
     --dtype bfloat16 \
     --mem-fraction-static 0.85
   ```

### Directory Structure
```
memory-rft/verl-instance/
├── configs/
│   ├── model/
│   │   └── deepseek_r1_8b.yaml
│   ├── reward/
│   │   └── podcast.yaml
│   └── podcast_ppo.yaml
├── data/
│   └── podcast_questions.jsonl
├── memory_tool.py
├── cot_tool_sglang.py
├── podcast_reward.py
├── verl_env.py
├── run_podcast_ppo.py
├── test_components.py
└── README.md
```

## Usage

### 1. Test Components
```bash
python test_components.py
```

### 2. Run PPO Training
```bash
python run_podcast_ppo.py
```

### 3. Override Configuration
```bash
python run_podcast_ppo.py \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  run.total_steps=100000 \
  tracker.experiment_name=my_experiment
```

## Configuration

### Key Parameters (in `configs/podcast_ppo.yaml`)

- **Learning rate**: `1e-5` for actor, `5e-6` for critic
- **Batch size**: 256 tokens
- **PPO epochs**: 4
- **KL penalty**: 0.05
- **Temperature**: 0.7

### Memory/Compute Requirements

- **1 × H100 80GB**: ~4 hours for full training
- **4 × H100 80GB**: ~1 hour (recommended)
- **8 × H100 80GB**: ~35 minutes

## Monitoring

- **Console**: Real-time metrics during training
- **WandB**: Loss curves, rewards, KL divergence
- **Checkpoints**: Saved every 1000 steps

## Example Interaction

**Question**: "Which country did Joe Rogan visit in 2024?"

**Model reasoning**:
```
<think>
I need to find information about Joe Rogan's travels in 2024.
#call memory_retrieval {"query": "Joe Rogan 2024 visit country", "k": 3}

#memory_retrieval_result:
[1] In episode 1823, Joe Rogan discussed his recent trip to Japan...
[2] ...

Based on the retrieved information, Joe visited Japan in 2024.
</think>

Joe Rogan visited Japan in 2024, where he explored Tokyo and experienced the local culture.
```

## Customization

### Use Real Memory Retrieval
Replace `MockMemoryRetriever` in `memory_tool.py` with:
```python
class MemoryRetriever:
    def __init__(self, index_path, embedder):
        self.index = faiss.read_index(index_path)
        self.embedder = SentenceTransformer(embedder)
        # ... implement real retrieval
```

### Use External Judge
Modify `podcast_reward.py` to use GPT-4 or other APIs:
```python
if not use_local_judge:
    # Call OpenAI/Anthropic API
    response = openai.chat.completions.create(...)
```

### Add Tool Usage Penalty
In `configs/reward/podcast.yaml`:
```yaml
reward:
  compute_score_fn: podcast_with_tool_penalty
  tool_penalty: -0.1
```

## Troubleshooting

1. **SGLang connection error**: Ensure server is running on port 30000
2. **OOM errors**: Reduce `ppo_micro_batch_size_per_gpu`
3. **Slow training**: Check GPU utilization, increase batch size

## Future Improvements

- [ ] Real vector database integration
- [ ] Multi-turn conversations
- [ ] Citation tracking in responses
- [ ] Streaming tool calls for better UX
- [ ] GRPO algorithm option for faster training 

verl version ab97d9b2906b44612b024d016081f553b39b8a30