#!/usr/bin/env python3
"""Main training script for podcast memory retrieval with PPO."""
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Note: Reward functions are now loaded as custom functions via config
# No need to import them here anymore

from verl.trainer.main_ppo import main as ppo_main


@hydra.main(config_path="configs", config_name="podcast_ppo", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for PPO training.
    
    Args:
        cfg: Hydra configuration
    """
    # Print configuration
    print("Starting PPO training with configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Ensure output directory exists
    os.makedirs(cfg.run.output_dir, exist_ok=True)
    
    # Check if SGLang server is specified
    sglang_url = getattr(cfg.actor_rollout_ref.rollout, 'sglang_url', None)
    if sglang_url:
        print(f"Using SGLang server at: {sglang_url}")
        print("Make sure the SGLang server is running with the model!")
    
    # Launch PPO training
    ppo_main(cfg)


if __name__ == "__main__":
    main() 