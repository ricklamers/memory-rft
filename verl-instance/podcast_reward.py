"""Reward function for podcast QA using heuristic scoring."""
import re
import json
import requests
from typing import List, Union


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute reward score for verl training.
    
    This function is called by NaiveRewardManager with specific parameters.
    
    Args:
        data_source: The data source identifier (e.g., "podcast_qa")
        solution_str: The model's generated response
        ground_truth: The expected answer from the dataset
        extra_info: Additional information from the dataset
        
    Returns:
        float: Reward score between 0 and 1
    """
    # Simple heuristic scoring based on answer similarity
    
    # Normalize both strings for comparison
    response_lower = solution_str.lower().strip()
    ground_truth_lower = ground_truth.lower().strip()
    
    # Check for exact match (highest score)
    if response_lower == ground_truth_lower:
        return 1.0
    
    # Check if key information is present
    score = 0.0
    
    # Extract key terms from ground truth (words longer than 3 chars)
    key_terms = [word for word in ground_truth_lower.split() if len(word) > 3]
    
    # Calculate score based on how many key terms are present
    if key_terms:
        matches = sum(1 for term in key_terms if term in response_lower)
        score = matches / len(key_terms)
    
    # Bonus for mentioning specific entities
    if "joe" in response_lower or "rogan" in response_lower:
        score = min(score + 0.1, 1.0)
    
    # Penalty for very short responses
    if len(response_lower.split()) < 3:
        score *= 0.5
    
    # Penalty for very long responses (likely off-topic)
    if len(response_lower.split()) > 50:
        score *= 0.8
    
    return float(score)


def compute_score_simple(data_source, solution_str, ground_truth, extra_info=None):
    """
    Simple reward function for direct evaluation.
    
    Args:
        data_source: The dataset name/source
        solution_str: The model's response/answer
        ground_truth: The ground truth answer
        extra_info: Additional information
        
    Returns:
        Float score in range [0, 1]
    """
    if extra_info is None:
        extra_info = {}
    
    use_local_judge = extra_info.get("use_local_judge", False)
    
    if use_local_judge:
        # Use heuristic judge since SGLang server may not be available
        return _simple_heuristic_judge(ground_truth, solution_str)
    else:
        # Simple length-based scoring
        return min(1.0, len(solution_str) / 100.0)


def _simple_heuristic_judge(question: str, answer: str) -> float:
    """Simple heuristic judge for testing when SGLang not available."""
    if not answer or not question:
        return 0.0
        
    # Simple keyword matching heuristic
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())
    
    # Basic overlap score
    overlap = len(question_words.intersection(answer_words))
    overlap_ratio = overlap / max(len(question_words), 1)
    
    # Convert to [0, 1] range
    return min(1.0, max(0.0, overlap_ratio))


def _judge_with_sglang(question: str, answer: str, judge_url: str) -> float:
    """Judge answer quality using SGLang server."""
    judge_prompt = f"""
    Please evaluate if the following answer correctly responds to the question.
    
    Question: {question}
    Answer: {answer}
    
    Rate the answer on a scale of 0-10 where:
    - 0: Completely wrong or irrelevant
    - 5: Partially correct but missing key information
    - 10: Fully correct and complete
    
    Respond with just the number (0-10):
    """
    
    try:
        response = requests.post(
            f"{judge_url}/generate",
            json={
                "text": judge_prompt,
                "sampling_params": {
                    "max_new_tokens": 10,
                    "temperature": 0.1
                }
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            judge_output = result.get("text", "5")
            
            # Extract number from response
            numbers = re.findall(r'\d+', judge_output)
            if numbers:
                score_0_10 = min(10, max(0, int(numbers[0])))
                # Convert to [0, 1] range
                return score_0_10 / 10.0
            else:
                return 0.5  # Neutral score if can't parse
        else:
            return 0.5
            
    except Exception as e:
        print(f"Judge error: {e}")
        return 0.5


# Legacy function for backwards compatibility
def compute_score_fn(batch, **kwargs):
    """
    Legacy wrapper for batch processing.
    """
    questions = batch["prompt"]
    answers = batch["response"]
    
    scores = []
    for question, answer in zip(questions, answers):
        score = compute_score_simple("podcast", answer, question, kwargs)
        scores.append(score)
    
    return scores


# Alternative reward function with tool penalty
def compute_score_with_tool_penalty(data_source, solution_str, ground_truth, extra_info=None):
    """Reward function that penalizes excessive tool use."""
    if extra_info is None:
        extra_info = {}
    
    # Get base score
    base_score = compute_score_simple(data_source, solution_str, ground_truth, extra_info)
    
    # Apply tool penalty
    tool_penalty = extra_info.get("tool_penalty", -0.1)
    tool_calls = solution_str.count("#call memory_retrieval")
    penalty = tool_calls * tool_penalty
    final_score = max(0.0, min(1.0, base_score + penalty))
    
    return final_score 