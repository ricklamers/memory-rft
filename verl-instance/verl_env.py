"""veRL environment for podcast QA with memory retrieval."""
from typing import Dict, Any, Tuple, List
from cot_tool_sglang import CoTToolAgentSGLang
from memory_tool import MockMemoryRetriever
from podcast_reward import compute_score_fn


class PodcastEnv:
    """Environment for podcast QA with memory retrieval."""
    
    def __init__(self, 
                 questions: List[Dict[str, str]],
                 sglang_url: str = "http://localhost:30000",
                 judge_url: str = None,
                 use_local_judge: bool = True):
        self.questions = questions
        self.current_idx = 0
        self.current_question = None
        
        # Initialize the agent with mock retriever
        self.retriever = MockMemoryRetriever()
        self.agent = CoTToolAgentSGLang(
            sglang_url=sglang_url,
            retriever=self.retriever
        )
        
        # Judge configuration
        self.judge_url = judge_url or sglang_url
        self.use_local_judge = use_local_judge
        
    def reset(self, idx: int = None) -> str:
        """
        Reset environment and return the question.
        
        Args:
            idx: Index of question to use, if None uses current_idx
            
        Returns:
            Question string
        """
        if idx is not None:
            self.current_idx = idx
        
        if self.current_idx >= len(self.questions):
            self.current_idx = 0
            
        question_data = self.questions[self.current_idx]
        self.current_question = question_data["question"]
        
        return self.current_question
    
    def step(self, action: str = None) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Response from the agent (if None, agent generates response)
            
        Returns:
            Tuple of (response, reward, done, info)
        """
        if self.current_question is None:
            raise ValueError("Environment not reset. Call reset() first.")
        
        # Generate response if not provided
        if action is None:
            try:
                response, transcript = self.agent.answer(self.current_question)
            except Exception as e:
                # Fallback response if agent fails
                response = f"I need to search for information about: {self.current_question}"
                transcript = [{"role": "assistant", "content": response}]
        else:
            response = action
            transcript = [{"role": "assistant", "content": response}]
        
        # Compute reward using the reward function
        batch = {
            "prompt": [self.current_question],
            "response": [response]
        }
        
        try:
            scores = compute_score_fn(
                batch, 
                judge_url=self.judge_url,
                use_local_judge=self.use_local_judge
            )
            reward = scores[0] if scores else 0.0
        except Exception as e:
            print(f"Reward computation error: {e}")
            reward = 0.0
        
        # Environment is "done" after each question
        done = True
        
        # Info dictionary with additional details
        info = {
            "question": self.current_question,
            "response": response,
            "transcript": transcript,
            "question_idx": self.current_idx
        }
        
        # Move to next question for future resets
        self.current_idx += 1
        
        return response, reward, done, info
    
    def get_question_count(self) -> int:
        """Return total number of questions."""
        return len(self.questions)
    
    def get_current_question_idx(self) -> int:
        """Return current question index."""
        return self.current_idx


def create_env_fn(config: Dict[str, Any]) -> PodcastEnv:
    """Factory function to create environment instances."""
    # Load questions from file or config
    questions = config.get("questions", [
        {"question": "Which country did Joe Rogan visit in 2024?"},
        {"question": "What did Joe say about AI's impact on podcasts?"},
        {"question": "What supplements did Joe mention trying?"},
        {"question": "What are Joe's thoughts on space exploration?"},
        {"question": "How has comedy evolved according to Joe?"}
    ])
    
    return PodcastEnv(
        questions=questions,
        sglang_url=config.get("sglang_url", "http://localhost:30000"),
        judge_url=config.get("judge_url", None),
        use_local_judge=config.get("use_local_judge", True)
    ) 