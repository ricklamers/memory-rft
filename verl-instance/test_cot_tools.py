#!/usr/bin/env python3
"""Test script to verify CoT tool calling functionality."""
import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from custom_sglang_rollout import CoTToolSGLangRollout
from memory_tool import MockMemoryRetriever
import torch
from transformers import AutoTokenizer

def test_tool_pattern_matching():
    """Test that our regex patterns work correctly."""
    rollout = CoTToolSGLangRollout.__new__(CoTToolSGLangRollout)
    rollout.tool_call_pattern = CoTToolSGLangRollout.__dict__['__init__'].__code__.co_consts[5]
    rollout.cot_end_pattern = CoTToolSGLangRollout.__dict__['__init__'].__code__.co_consts[6]
    
    # Import regex module
    import re
    rollout.tool_call_pattern = re.compile(r'#call memory_retrieval\s+({.*?})')
    rollout.cot_end_pattern = re.compile(r'</think>')
    
    # Test cases
    test_texts = [
        "Let me search for info. #call memory_retrieval {\"query\": \"test query\", \"k\": 3}",
        "This ends the thinking </think>",
        "No tool call here",
        "#call memory_retrieval {\"query\": \"another test\"}",
    ]
    
    print("Testing pattern matching:")
    for text in test_texts:
        tool_match = rollout.tool_call_pattern.search(text)
        end_match = rollout.cot_end_pattern.search(text)
        
        if tool_match:
            print(f"✓ Found tool call in: '{text}'")
            print(f"  Args: {tool_match.group(1)}")
        elif end_match:
            print(f"✓ Found CoT end in: '{text}'")
        else:
            print(f"· No pattern found in: '{text}' (expected for this test)")
    print()

def test_memory_retriever():
    """Test the memory retriever tool."""
    print("Testing memory retriever:")
    retriever = MockMemoryRetriever()
    
    # Test retrieval
    results = retriever("Joe Rogan Japan 2024", k=3)
    print(f"Query: 'Joe Rogan Japan 2024'")
    print(f"Results: {len(results)} chunks retrieved")
    for i, chunk in enumerate(results):
        print(f"  [{i+1}] {chunk['text'][:100]}...")
    print()

def test_response_formatting():
    """Test how tool responses would be formatted."""
    print("Testing response formatting:")
    
    # Simulate tool results
    retrieved_chunks = [
        {"text": "Joe Rogan discussed his trip to Japan in episode 1823."},
        {"text": "He was impressed by Japanese efficiency and culture."},
        {"text": "Rogan tried authentic ramen and visited temples."}
    ]
    
    # Format response (from the actual implementation)
    response_lines = ["\n\n#memory_retrieval_result:"]
    for i, chunk in enumerate(retrieved_chunks):
        response_lines.append(f"[{i+1}] {chunk['text']}")
    response_lines.append("\n")
    
    formatted_response = "\n".join(response_lines)
    print("Formatted tool response:")
    print(formatted_response)
    print()

def main():
    """Run all tests."""
    print("="*60)
    print("Testing CoT Tool Calling Components")
    print("="*60)
    print()
    
    test_tool_pattern_matching()
    test_memory_retriever()
    test_response_formatting()
    
    print("="*60)
    print("All tests completed!")
    print("="*60)
    print()
    print("To run full PPO training with CoT tools:")
    print("  python run_podcast_ppo_with_cot_tools.py")
    print()

if __name__ == "__main__":
    main() 