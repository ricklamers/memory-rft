#!/usr/bin/env python3
"""
Example script demonstrating how to use the QA batch API
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_qa_batch_api():
    """Test the QA batch API endpoints."""
    
    print("Testing QA Batch API")
    print("=" * 50)
    
    # 1. Check API stats
    print("\n1. Checking API stats...")
    response = requests.get(f"{BASE_URL}/api/stats")
    if response.status_code == 200:
        stats = response.json()
        print(f"   Total transcripts: {stats.get('total_transcripts', 0)}")
        print(f"   QA pairs available: {stats.get('qa_pairs_available', False)}")
        if stats.get('qa_pairs_available'):
            print(f"   Total QA pairs: {stats.get('total_qa_pairs', 0)}")
    else:
        print(f"   Error: {response.status_code}")
    
    # 2. Get QA statistics
    print("\n2. Getting QA statistics...")
    response = requests.get(f"{BASE_URL}/api/qa/stats")
    if response.status_code == 200:
        qa_stats = response.json()
        print(f"   Total QA pairs: {qa_stats['total_qa_pairs']}")
        print(f"   Episodes with QA: {qa_stats['episodes_with_qa']}")
        print(f"   Average QA per episode: {qa_stats['average_qa_per_episode']:.1f}")
        print(f"   Difficulty distribution: {qa_stats['difficulty_distribution']}")
    else:
        print(f"   Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")
    
    # 3. Get episodes with QA pairs
    print("\n3. Getting episodes with QA pairs...")
    response = requests.get(f"{BASE_URL}/api/qa/episodes")
    if response.status_code == 200:
        episodes = response.json()['episodes']
        print(f"   Found {len(episodes)} episodes with QA pairs")
        if episodes:
            print(f"   First 5 episodes: {episodes[:5]}")
    else:
        print(f"   Error: {response.status_code}")
    
    # 4. Get a random batch of QA pairs
    print("\n4. Getting a random batch of QA pairs...")
    batch_request = {
        "batch_size": 5,
        "shuffle": True,
        "difficulty_filter": None,
        "seed": 42  # For reproducibility
    }
    
    response = requests.post(f"{BASE_URL}/api/qa/batch", json=batch_request)
    if response.status_code == 200:
        batch = response.json()
        print(f"   Received {batch['batch_size']} QA pairs")
        print(f"   Total available: {batch['total_available']}")
        print(f"   Seed used: {batch['seed_used']}")
        
        # Display first 2 QA pairs
        for i, qa in enumerate(batch['qa_pairs'][:2], 1):
            print(f"\n   QA Pair {i}:")
            print(f"   Episode: {qa['episode_file']}")
            print(f"   Question: {qa['question'][:100]}...")
            print(f"   Answer: {qa['answer'][:100]}...")
            print(f"   Difficulty: {qa['difficulty']}")
    else:
        print(f"   Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")
    
    # 5. Get filtered batch (hard questions only)
    print("\n\n5. Getting filtered batch (hard questions only)...")
    filtered_request = {
        "batch_size": 3,
        "shuffle": True,
        "difficulty_filter": "hard"
    }
    
    response = requests.post(f"{BASE_URL}/api/qa/batch", json=filtered_request)
    if response.status_code == 200:
        batch = response.json()
        print(f"   Received {batch['batch_size']} hard QA pairs")
        print(f"   Total hard questions available: {batch['total_available']}")
    else:
        print(f"   Error: {response.status_code}")
    
    # 6. Get batch from specific episode
    print("\n6. Getting batch from specific episode...")
    if episodes and len(episodes) > 0:
        episode_request = {
            "batch_size": 2,
            "shuffle": False,
            "episode_filter": episodes[0]
        }
        
        response = requests.post(f"{BASE_URL}/api/qa/batch", json=episode_request)
        if response.status_code == 200:
            batch = response.json()
            print(f"   Episode: {episodes[0]}")
            print(f"   Received {batch['batch_size']} QA pairs from this episode")
            print(f"   Total available in episode: {batch['total_available']}")
        else:
            print(f"   Error: {response.status_code}")

def demonstrate_training_loop():
    """Demonstrate how to use the API in a training loop."""
    print("\n\nDemonstrating Training Loop")
    print("=" * 50)
    
    # Configuration
    batch_size = 32
    num_batches = 3
    
    print(f"Simulating {num_batches} training batches of size {batch_size}")
    
    for batch_num in range(num_batches):
        print(f"\nBatch {batch_num + 1}/{num_batches}")
        
        # Request a new shuffled batch
        batch_request = {
            "batch_size": batch_size,
            "shuffle": True,
            "seed": None  # Random seed each time
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/qa/batch", json=batch_request)
        request_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            batch = response.json()
            print(f"   ✓ Received {batch['batch_size']} QA pairs in {request_time:.1f}ms")
            print(f"   Seed: {batch['seed_used']}")
            
            # Simulate training
            print("   Training on batch...")
            time.sleep(0.5)  # Simulate training time
        else:
            print(f"   ✗ Error getting batch: {response.status_code}")
            break

def main():
    print("QA Batch API Test Script")
    print("Make sure the API is running: python dataset_utils/api.py")
    print()
    
    # Test if API is running
    try:
        response = requests.get(f"{BASE_URL}/api/stats", timeout=2)
        if response.status_code != 200:
            print("❌ API is not responding correctly")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to API at", BASE_URL)
        print("Please start the API with: python dataset_utils/api.py")
        return
    
    print("✅ API is running\n")
    
    # Run tests
    test_qa_batch_api()
    demonstrate_training_loop()
    
    print("\n\nAll tests completed!")

if __name__ == "__main__":
    main() 