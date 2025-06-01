"""CoT tool wrapper for SGLang with memory retrieval."""
import re
import json
import requests
from typing import Tuple, List, Dict, Any
from memory_tool import MockMemoryRetriever


CALL_RE = re.compile(r"#call\s+memory_retrieval\s+({.*?})")


class CoTToolAgentSGLang:
    """SGLang-based agent that can call memory retrieval during reasoning."""
    
    def __init__(self, 
                 sglang_url: str = "http://localhost:30000",
                 retriever: MockMemoryRetriever = None,
                 model_name: str = None):
        self.sglang_url = sglang_url
        self.retriever = retriever or MockMemoryRetriever()
        self.model_name = model_name
        
        # Default sampling params
        self.sampling_params = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": 512,
            "stop": ["<|user|>", "</think>"]
        }
    
    def _format_prompt(self, question: str) -> str:
        """Format a simple question into a prompt for testing."""
        return f"""You are a helpful assistant. Think step-by-step inside <think>...</think>.

User: {question}"""
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for the model (using simple format for now)."""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}"
            elif role == "tool":
                formatted += f"\nTool Result: {content}\n"
        return formatted
    
    def _inject_chunks(self, assistant_msg: Dict[str, str], call_json: str) -> Dict[str, str]:
        """Inject retrieval results into the assistant's thinking."""
        try:
            req = json.loads(call_json)
            chunks = self.retriever(req.get("query", ""), k=req.get("k", 3))
            
            # Format chunks as a result block
            result = "\n\n#memory_retrieval_result:\n"
            for i, chunk in enumerate(chunks):
                result += f"[{i+1}] {chunk['text']}\n"
            result += "\n"
            
            # Append to the assistant's content
            assistant_msg["content"] += result
        except Exception as e:
            print(f"Error in retrieval: {e}")
            assistant_msg["content"] += "\n\n#memory_retrieval_error: Failed to retrieve\n\n"
        
        return assistant_msg
    
    def _generate_completion(self, prompt: str, stop_at_call: bool = False) -> str:
        """Generate completion using SGLang API."""
        data = {
            "text": prompt,
            "sampling_params": self.sampling_params.copy()
        }
        
        if stop_at_call:
            # Add regex stop pattern for tool calls
            data["sampling_params"]["stop"] = self.sampling_params["stop"] + ["#call memory_retrieval"]
        
        try:
            response = requests.post(f"{self.sglang_url}/generate", json=data)
            if response.status_code == 200:
                result = response.json()
                return result.get("text", "")
            else:
                print(f"SGLang error: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            print(f"Connection error: {e}")
            return ""
    
    def answer(self, question: str) -> Tuple[str, List[Dict[str, str]]]:
        """Generate answer with CoT and tool calls."""
        dialog = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Think step-by-step inside <think>...</think>. "
                    "When you need information from podcast transcripts, use:\n"
                    "#call memory_retrieval {\"query\": \"...\", \"k\": 3}\n"
                    "After the retrieval results appear, continue your reasoning and provide a final answer."
                )
            },
            {"role": "user", "content": question}
        ]
        
        # Start assistant's thinking
        assistant_msg = {"role": "assistant", "content": "<think>\n"}
        dialog.append(assistant_msg)
        
        # Main reasoning loop
        max_iterations = 5  # Prevent infinite loops
        for _ in range(max_iterations):
            prompt = self._format_messages(dialog)
            
            # Generate until we hit a tool call or finish thinking
            completion = self._generate_completion(prompt, stop_at_call=True)
            assistant_msg["content"] += completion
            
            # Check if there's a tool call
            match = CALL_RE.search(completion)
            if match:
                # Add the complete tool call to the message
                assistant_msg["content"] += "#call memory_retrieval " + match.group(1)
                # Inject the retrieval results
                assistant_msg = self._inject_chunks(assistant_msg, match.group(1))
                continue
            
            # Check if we finished the thinking block
            if "</think>" in completion or not completion:
                break
        
        # Close thinking if not already closed
        if not assistant_msg["content"].rstrip().endswith("</think>"):
            assistant_msg["content"] += "\n</think>\n"
        
        # Generate the final answer
        prompt = self._format_messages(dialog)
        final_answer = self._generate_completion(prompt)
        
        # Create a new message for the visible answer
        if final_answer:
            dialog.append({"role": "assistant", "content": final_answer})
        
        return final_answer.strip(), dialog