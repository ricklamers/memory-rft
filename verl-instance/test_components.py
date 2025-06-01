#!/usr/bin/env python3
"""Test script to verify all components work together."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from memory_tool import MockMemoryRetriever
from cot_tool_sglang import CoTToolAgentSGLang
from podcast_reward import compute_score_fn


def test_memory_retriever():
    """Test the mock memory retriever."""
    print("Testing Memory Retriever...")
    retriever = MockMemoryRetriever()
    
    # Test retrieval
    chunks = retriever("Joe Rogan Japan", k=2)
    print(f"Retrieved {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk['text'][:100]}...")
    print("✓ Memory retriever working\n")


def test_sglang_agent():
    """Test the SGLang agent with tool calling."""
    print("Testing SGLang Agent...")
    try:
        agent = CoTToolAgentSGLang()
        
        # Simple test without actual SGLang connection
        print("✓ Agent initialized successfully")
        
        # Test prompt construction
        test_prompt = "What country did Joe Rogan visit?"
        formatted_prompt = agent._format_prompt(test_prompt)
        print(f"✓ Prompt formatting works")
        print(f"  Formatted prompt preview: {formatted_prompt[:100]}...")
        
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
    print()


def test_reward_function():
    """Test the reward function."""
    print("Testing Reward Function...")
    
    # Create test batch
    test_batch = {
        "prompt": [
            "Which country did Joe Rogan visit in 2024?",
            "What did Joe say about AI?"
        ],
        "response": [
            "Joe Rogan visited Japan in 2024.",
            "Joe mentioned that AI is changing the podcast industry."
        ]
    }
    
    try:
        # Test with local judge disabled (heuristic mode)
        scores = compute_score_fn(test_batch, use_local_judge=False)
        print(f"✓ Reward function works")
        print(f"  Scores: {scores}")
        
        # Test that scores are in valid range
        for score in scores:
            assert -1.0 <= score <= 1.0, f"Score {score} out of range [-1, 1]"
        print("✓ Scores in valid range [-1, 1]")
        
    except Exception as e:
        print(f"✗ Reward function test failed: {e}")
    print()


def test_sglang_connection():
    """Test connection to SGLang server."""
    print("Testing SGLang Connection...")
    import requests
    
    try:
        response = requests.get("http://localhost:30000/health", timeout=5)
        if response.status_code == 200:
            print("✓ SGLang server is reachable")
            
            # Test a simple generation
            gen_response = requests.post(
                "http://localhost:30000/generate",
                json={
                    "text": "Hello, how are you?",
                    "sampling_params": {
                        "max_new_tokens": 10,
                        "temperature": 0.1
                    }
                },
                timeout=10
            )
            
            if gen_response.status_code == 200:
                print("✓ SGLang generation endpoint working")
            else:
                print("✗ SGLang generation endpoint not working")
                
        else:
            print(f"✗ SGLang server responded with status {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to SGLang server: {e}")
        print("  Make sure SGLang is running on http://localhost:30000")
    print()


def main():
    """Run all component tests."""
    print("=" * 50)
    print("TESTING PODCAST MEMORY RETRIEVAL COMPONENTS")
    print("=" * 50)
    print()
    
    test_memory_retriever()
    test_sglang_agent()
    test_reward_function()
    test_sglang_connection()
    
    print("=" * 50)
    print("COMPONENT TESTING COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main() 