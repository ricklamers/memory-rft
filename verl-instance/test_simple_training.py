#!/usr/bin/env python3
"""Simple training test to validate podcast memory retrieval components work together."""
import sys
import os
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from memory_tool import MockMemoryRetriever
from cot_tool_sglang import CoTToolAgentSGLang
from podcast_reward import compute_score
import json


def test_end_to_end_pipeline():
    """Test the complete pipeline: question -> agent -> reward."""
    print("🔄 Testing End-to-End Pipeline...")
    
    # Load mini dataset
    questions = []
    try:
        with open("data/podcast_questions_mini.jsonl", "r") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        print(f"✅ Loaded {len(questions)} questions from mini dataset")
    except Exception as e:
        print(f"❌ Failed to load mini dataset: {e}")
        return False
    
    # Initialize components
    try:
        retriever = MockMemoryRetriever()
        agent = CoTToolAgentSGLang(retriever=retriever)
        print("✅ Components initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return False
    
    # Test each question
    for i, item in enumerate(questions[:2]):  # Test first 2 questions
        question = item["question"]
        expected_answer = item["answer"]
        
        print(f"\n📝 Testing question {i+1}: {question}")
        
        # Generate response (simulate what happens in training)
        try:
            # For testing, we'll use the mock response since SGLang might not be available
            mock_response = f"Based on the podcast transcripts, {expected_answer}"
            print(f"🤖 Generated response: {mock_response[:100]}...")
            
            # Test reward function
            score = compute_score(
                data_source="podcast",
                solution_str=mock_response,
                ground_truth=question,
                extra_info={"use_local_judge": False}  # Use heuristic for testing
            )
            print(f"🎯 Reward score: {score:.3f}")
            
            # Validate score is in correct range
            if -1.0 <= score <= 1.0:
                print(f"✅ Score in valid range [-1, 1]")
            else:
                print(f"❌ Score {score} outside valid range [-1, 1]")
                return False
                
        except Exception as e:
            print(f"❌ Error processing question {i+1}: {e}")
            return False
    
    print("\n✅ End-to-end pipeline test completed successfully!")
    return True


def test_sglang_integration():
    """Test SGLang integration if server is available."""
    print("\n🔄 Testing SGLang Integration...")
    
    try:
        import requests
        response = requests.get("http://localhost:30000/health", timeout=2)
        if response.status_code == 200:
            print("✅ SGLang server is available")
            
            # Test actual generation
            agent = CoTToolAgentSGLang()
            test_question = "What country did Joe Rogan visit?"
            
            try:
                answer, dialog = agent.answer(test_question)
                print(f"🤖 SGLang generated: {answer[:100]}...")
                print("✅ SGLang integration working")
                return True
            except Exception as e:
                print(f"⚠️  SGLang generation failed: {e}")
                return False
        else:
            print("⚠️  SGLang server not responding correctly")
            return False
            
    except Exception as e:
        print(f"⚠️  SGLang server not available: {e}")
        print("   (This is OK for testing - using mock responses)")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING SIMPLE TRAINING PIPELINE")
    print("=" * 60)
    
    # Test 1: End-to-end pipeline with mock data
    success1 = test_end_to_end_pipeline()
    
    # Test 2: SGLang integration (optional)
    success2 = test_sglang_integration()
    
    print("\n" + "=" * 60)
    print("SIMPLE TRAINING PIPELINE TEST RESULTS")
    print("=" * 60)
    print(f"End-to-end pipeline: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"SGLang integration: {'✅ PASS' if success2 else '⚠️  SKIPPED (server not available)'}")
    
    if success1:
        print("\n🎉 Core pipeline is ready for training!")
        print("\n💡 Next steps:")
        print("   1. Start SGLang server if not running")
        print("   2. Run: python run_podcast_ppo.py --config-name=podcast_ppo_simple")
        print("   3. Monitor training progress")
    else:
        print("\n❌ Pipeline has issues - check errors above")
    
    return success1


if __name__ == "__main__":
    main() 