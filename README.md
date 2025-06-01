# Podcast Memory RFT with veRL

A reinforcement learning training pipeline that teaches language models to perform memory retrieval during Chain-of-Thought (CoT) reasoning using the veRL framework.

## Overview

This project implements **in-CoT tool execution** for podcast memory retrieval using PPO (Proximal Policy Optimization). The model learns to:

- Generate structured CoT reasoning with `<think>...</think>` tags
- Make real-time tool calls during generation using `#call memory_retrieval {...}` 
- Process actual tool results and continue reasoning
- Retrieve relevant information from a podcast knowledge base

## Key Components

### RL Training with veRL
- **Framework**: [veRL](https://github.com/volcengine/verl) for distributed PPO training
- **Model**: DeepSeek-R1-0528-Qwen3-8B as both actor and critic
- **Custom Rollout**: `CoTToolSGLangRollout` enables in-CoT tool execution during generation
- **Reward Function**: Custom scoring based on CoT quality and tool usage patterns

### In-CoT Tool Calling System
```python
# Example of learned behavior:
<think>
The user is asking about a specific topic. Let me search my memory for relevant information.
</think>

#call memory_retrieval {"query": "specific topic from user question"}

<think>
Based on the retrieved information: [tool results], I can now provide a comprehensive answer...
</think>
```

### Memory Infrastructure
- **Vector DB**: LanceDB for efficient similarity search
- **Data Pipeline**: Automated transcript processing, chunking, and QA pair generation
- **API Layer**: Fast retrieval service for real-time tool calls during training

## Quick Start

### 1. Setup Memory Database
```bash
# Process podcast transcripts and build vector index
cd dataset_utils
python transcript_to_chunks.py
python create_embeddings.py
python ingest_to_lancedb.py
```

### 2. Launch Training
```bash
# Start PPO training with in-CoT tool execution
python run_podcast_ppo_with_cot_tools.py
```

The training will:
- Patch the SGLang rollout to use our custom `CoTToolSGLangRollout`
- Enable real tool calls during model generation
- Train the model to use memory retrieval effectively in its reasoning

### 3. Configuration
Key settings in `configs/podcast_ppo_working.yaml`:
- `max_tool_calls_per_cot: 5` - Limit tool calls per reasoning chain
- `max_segment_length: 256` - Tokens generated before checking for tool calls
- Custom reward function weights CoT quality and tool usage

## Training Approach

The model learns through PPO to:
1. **Identify** when external memory is needed during reasoning
2. **Formulate** appropriate search queries for the memory system  
3. **Integrate** retrieved information into ongoing CoT reasoning
4. **Generate** high-quality responses using both internal knowledge and retrieved context

This creates an agent that can dynamically access external memory while maintaining natural reasoning flow.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PPO Training  │───▶│  CoT Generation  │───▶│ Memory Retrieval│
│   (veRL)        │    │  with Tools      │    │   (LanceDB)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌──────────────────┐             │
         └──────────────│  Reward Function │◀────────────┘
                        │   (CoT + Tools)  │
                        └──────────────────┘
```

## Files Structure

- `run_podcast_ppo_with_cot_tools.py` - Main training script with custom rollout
- `custom_sglang_rollout.py` - In-CoT tool execution implementation  
- `configs/podcast_ppo_working.yaml` - PPO training configuration
- `dataset_utils/` - Memory system components (transcripts, embeddings, QA generation)
- `podcast_reward_with_tools.py` - Custom reward function for tool usage

## Requirements

- veRL framework
- SGLang for model serving
- LanceDB for vector storage
- DeepSeek model access
- Multi-GPU setup (8 GPUs recommended)
