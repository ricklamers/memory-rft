"""
Reward function that encourages CoT + tool calling patterns.
This approach teaches the model to use tools by rewarding proper format.

IMPORTANT: This only does TEXT PATTERN MATCHING - no additional models are run!
"""
import re
import json
from typing import Dict, List, Any


def evaluate_tool_usage_quality(response: str) -> float:
    """
    Evaluate the quality of tool usage in the response.
    
    Args:
        response: The generated response to evaluate
        
    Returns:
        Score for tool usage quality (0.0 to 1.0)
    """
    score = 0.0
    
    # Check if response contains thinking block
    if "<think>" in response and "</think>" in response:
        score += 0.2
        
        # Extract thinking content
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if think_match:
            thinking = think_match.group(1)
            
            # Check for tool call pattern
            tool_calls = re.findall(r"#call memory_retrieval\s+({.*?})", thinking)
            if tool_calls:
                score += 0.3
                
                # Check if tool call has valid JSON
                for call in tool_calls:
                    try:
                        params = json.loads(call)
                        if "query" in params and isinstance(params["query"], str):
                            score += 0.2
                        if "k" in params and isinstance(params["k"], int):
                            score += 0.1
                    except json.JSONDecodeError:
                        score -= 0.1
                
                # Check for tool results pattern
                if "#memory_retrieval_result:" in thinking:
                    score += 0.2
    
    return min(score, 1.0)


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None, **kwargs) -> float:
    """
    Compute rewards that encourage CoT + tool calling patterns.
    
    Args:
        data_source: The source of the data (e.g., dataset name)
        solution_str: The generated response to evaluate
        ground_truth: The ground truth answer (not used in this implementation)
        extra_info: Additional information (optional)
        **kwargs: Additional arguments
        
    Returns:
        Single reward score (float)
    """
    # Base content quality score (you can customize this)
    base_score = 0.5  # Baseline score
    
    # Tool usage quality bonus
    tool_score = evaluate_tool_usage_quality(solution_str)
    
    # Combine scores
    total_score = base_score + tool_score
    
    # Ensure score is in reasonable range
    final_score = min(max(total_score, 0.0), 2.0)
    
    return final_score


def compute_score_fn(data_source: str, solution_str: str, ground_truth: str, extra_info=None, **kwargs) -> float:
    """
    Main compute score function for veRL.
    """
    return compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs) 