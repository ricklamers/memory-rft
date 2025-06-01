#!/usr/bin/env python3
"""Training script for podcast memory retrieval with PPO using in-CoT tool execution."""
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our custom rollout
from custom_sglang_rollout import CoTToolSGLangRollout

from verl.trainer.main_ppo import main as ppo_main


@hydra.main(config_path="configs", config_name="podcast_ppo_working", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for PPO training with in-CoT tool execution.
    
    Args:
        cfg: Hydra configuration
    """
    # Disable NVLS to fix NCCL communication issues with 4+ GPUs
    os.environ['NCCL_NVLS_ENABLE'] = '0'
    
    # Print configuration
    print("Starting PPO training with in-CoT tool execution:")
    print(OmegaConf.to_yaml(cfg))
    
    # Ensure output directory exists
    os.makedirs(cfg.run.output_dir, exist_ok=True)
    
    # Patch the rollout class to use our custom implementation
    import verl.workers.rollout.sglang_rollout
    
    # Store the original SGLangRollout class
    original_SGLangRollout = verl.workers.rollout.sglang_rollout.SGLangRollout
    
    # Replace with our custom implementation
    verl.workers.rollout.sglang_rollout.SGLangRollout = CoTToolSGLangRollout
    
    print("="*60)
    print("Patched SGLangRollout with CoTToolSGLangRollout")
    print("This enables in-CoT tool execution during generation")
    print("="*60)
    
    # Check if SGLang server is specified
    sglang_url = getattr(cfg.actor_rollout_ref.rollout, 'sglang_url', None)
    if sglang_url:
        print(f"Using SGLang server at: {sglang_url}")
        print("Make sure the SGLang server is running with the model!")
    
    print("\n" + "="*60)
    print("TRAINING APPROACH: In-CoT Tool Execution")
    print("="*60)
    print("The model will learn to:")
    print("- Generate CoT reasoning with <think>...</think>")
    print("- Call tools with #call memory_retrieval {...}")
    print("- Process REAL tool results during generation")
    print("- Continue reasoning after receiving tool outputs")
    print("="*60 + "\n")
    
    try:
        # Launch PPO training with custom rollout
        ppo_main(cfg)
    finally:
        # Restore original class (cleanup)
        verl.workers.rollout.sglang_rollout.SGLangRollout = original_SGLangRollout


if __name__ == "__main__":
    main() 