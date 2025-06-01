"""Mock memory retrieval tool for testing."""
import json
import random
from typing import List, Dict, Any


class MockMemoryRetriever:
    """Mock memory retriever that returns fake podcast transcript chunks."""
    
    def __init__(self, idx_path: str = None, chunk_path: str = None):
        # Mock chunks for testing
        self.mock_chunks = [
            {
                "id": 0,
                "text": "In episode 1823, Joe Rogan discussed his recent trip to Japan where he experienced the culture and visited Tokyo. He mentioned being fascinated by the technology and the food.",
                "meta": {"episode": 1823, "timestamp": "00:15:30"}
            },
            {
                "id": 1,
                "text": "During the conversation with the guest, they talked about artificial intelligence and its impact on society. Joe mentioned how AI is changing the podcast industry.",
                "meta": {"episode": 1824, "timestamp": "01:23:45"}
            },
            {
                "id": 2,
                "text": "The discussion turned to health and fitness. Joe talked about his workout routine and mentioned trying new supplements. He emphasized the importance of consistency.",
                "meta": {"episode": 1825, "timestamp": "00:45:20"}
            },
            {
                "id": 3,
                "text": "In a recent episode, they discussed comedy and stand-up. Joe reflected on his early days in comedy clubs and how the industry has evolved with social media.",
                "meta": {"episode": 1826, "timestamp": "00:30:15"}
            },
            {
                "id": 4,
                "text": "The guest brought up the topic of space exploration. Joe expressed his excitement about SpaceX and the possibility of humans living on Mars in the future.",
                "meta": {"episode": 1827, "timestamp": "02:10:30"}
            }
        ]
    
    def __call__(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Return k mock chunks most 'relevant' to the query."""
        # For testing, just return random chunks
        # In real implementation, this would do vector similarity search
        selected = random.sample(self.mock_chunks, min(k, len(self.mock_chunks)))
        return selected 