"""
Custom SGLangRollout that performs tool calls within a single CoT generation.
"""
import re
import json
import asyncio
from typing import List, Dict, Any, Tuple
from uuid import uuid4
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tensordict import TensorDict

from verl import DataProto
from verl.workers.rollout.sglang_rollout import SGLangRollout
from verl.utils.torch_functional import pad_sequence_to_length
from verl.utils.model import compute_position_id_with_mask
from verl.utils.debug import GPUMemoryLogger
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj
from memory_tool import MockMemoryRetriever

import logging
import os

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class CoTToolSGLangRollout(SGLangRollout):
    """SGLangRollout with in-CoT tool calling support."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize our custom memory retriever tool
        self._memory_retriever = MockMemoryRetriever()
        
        # Define patterns for tool calls and CoT end
        self.tool_call_pattern = re.compile(r'#call memory_retrieval\s+({.*?})')
        self.cot_end_pattern = re.compile(r'</think>')
        
        # Configuration for in-CoT tool calling
        self.max_tool_calls_per_cot = self.config.get('max_tool_calls_per_cot', 5)
        self.max_segment_length = self.config.get('max_segment_length', 256)
        
        logger.info(f"Initialized CoTToolSGLangRollout with max_tool_calls_per_cot={self.max_tool_calls_per_cot}")
    
    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Override to use our custom CoT tool calling generation."""
        # Check if we should use CoT tool calling
        use_cot_tools = prompts.meta_info.get('use_cot_tools', True)
        if not use_cot_tools or self.config.multi_turn.enable:
            # Fall back to standard generation
            return super().generate_sequences(prompts, **kwargs)
        
        # Use our custom CoT tool calling generation
        return self._generate_sequences_with_cot_tools(prompts, **kwargs)
    
    async def _generate_single_sequence_with_cot_tools(
        self,
        initial_prompt_token_ids: List[int],
        sampling_params: Dict[str, Any],
        max_tool_calls: int,
        max_segment_len: int,
        max_total_response_len: int
    ) -> Tuple[List[int], List[float]]:
        """Generate a single sequence with in-CoT tool calls."""
        
        current_token_ids = list(initial_prompt_token_ids)
        full_response_ids = []
        full_logprobs = []
        tool_call_count = 0
        
        for iteration in range(max_tool_calls + 1):
            if len(full_response_ids) >= max_total_response_len:
                break
            
            # Prepare sampling params for this segment
            segment_params = sampling_params.copy()
            segment_params["max_new_tokens"] = min(
                max_segment_len, 
                max_total_response_len - len(full_response_ids)
            )
            
            if segment_params["max_new_tokens"] <= 0:
                break
            
            # Generate a segment
            segment_output = None
            if self._tp_rank == 0:
                loop = asyncio.get_event_loop()
                segment_output_list = await self._engine.async_generate(
                    prompt=None,
                    sampling_params=segment_params,
                    prompt_token_ids=[current_token_ids],
                    return_logprob=True,
                )
                segment_output = segment_output_list[0]
            
            # Broadcast result to other TP ranks
            [segment_output] = broadcast_pyobj(
                data=[segment_output],
                rank=self._rank,
                dist_group=self._device_mesh_cpu["tp"].get_group(),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            
            # Extract generated tokens and logprobs
            output_token_logprobs = segment_output["meta_info"]["output_token_logprobs"]
            segment_token_ids = [tpl[0] for tpl in output_token_logprobs]
            segment_logprobs = [tpl[1] for tpl in output_token_logprobs]
            
            # Add to full response
            full_response_ids.extend(segment_token_ids)
            full_logprobs.extend(segment_logprobs)
            current_token_ids.extend(segment_token_ids)
            
            # Decode to check for patterns
            generated_text = self.tokenizer.decode(segment_token_ids, skip_special_tokens=False)
            
            # Check for CoT end
            if self.cot_end_pattern.search(generated_text):
                logger.debug(f"Found CoT end pattern in iteration {iteration}")
                break
            
            # Check for tool call
            tool_matches = list(self.tool_call_pattern.finditer(generated_text))
            if tool_matches and tool_call_count < max_tool_calls:
                # Process the last tool call in this segment
                match = tool_matches[-1]
                tool_args_str = match.group(1)
                
                try:
                    tool_args = json.loads(tool_args_str)
                    query = tool_args.get("query", "")
                    k = tool_args.get("k", 3)
                    
                    logger.debug(f"Executing tool call: query='{query}', k={k}")
                    
                    # Execute the memory retrieval
                    retrieved_chunks = self._memory_retriever(query, k=k)
                    
                    # Format tool response
                    response_lines = ["\n\n#memory_retrieval_result:"]
                    for i, chunk in enumerate(retrieved_chunks):
                        response_lines.append(f"[{i+1}] {chunk['text']}")
                    response_lines.append("\n")
                    
                    formatted_response = "\n".join(response_lines)
                    
                    # Tokenize and add tool response
                    response_token_ids = self.tokenizer.encode(
                        formatted_response, 
                        add_special_tokens=False
                    )
                    
                    full_response_ids.extend(response_token_ids)
                    # Tool response tokens get 0 logprob (not generated by policy)
                    full_logprobs.extend([0.0] * len(response_token_ids))
                    current_token_ids.extend(response_token_ids)
                    
                    tool_call_count += 1
                    logger.debug(f"Tool call {tool_call_count} executed successfully")
                    
                except Exception as e:
                    logger.error(f"Tool call failed: {e}")
                    # Add error message
                    error_msg = f"\n\n#memory_retrieval_error: {str(e)[:100]}\n\n"
                    error_tokens = self.tokenizer.encode(error_msg, add_special_tokens=False)
                    full_response_ids.extend(error_tokens)
                    full_logprobs.extend([0.0] * len(error_tokens))
                    current_token_ids.extend(error_tokens)
            
            # Check if we hit natural stop
            finish_reason = segment_output["meta_info"]["finish_reason"]["type"]
            if finish_reason == "stop" or finish_reason == "eos":
                break
        
        return full_response_ids, full_logprobs
    
    @GPUMemoryLogger(role="sglang rollout with cot tools", logger=logger)
    @torch.no_grad()
    def _generate_sequences_with_cot_tools(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences with in-CoT tool calling."""
        
        # Extract prompts
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)
        
        # Get raw prompt IDs
        initial_prompt_ids_list = []
        for i in range(batch_size):
            prompt_ids = []
            for j in range(idx.size(1)):
                if attention_mask[i, j] == 1:
                    prompt_ids.append(idx[i, j].item())
            initial_prompt_ids_list.append(prompt_ids)
        
        # Prepare sampling params
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        
        if not do_sample or is_validate:
            sampling_params = dict(
                n=1,
                temperature=0,
                top_p=1,
                top_k=-1,
            )
        else:
            sampling_params = self.sampling_params.copy()
        
        # Update with any kwargs
        sampling_params.update(kwargs)
        
        # Generate for each prompt
        all_responses = []
        all_logprobs = []
        
        # Process each prompt (could be parallelized with asyncio.gather)
        loop = asyncio.get_event_loop()
        tasks = []
        for prompt_ids in initial_prompt_ids_list:
            task = self._generate_single_sequence_with_cot_tools(
                initial_prompt_token_ids=prompt_ids,
                sampling_params=sampling_params,
                max_tool_calls=self.max_tool_calls_per_cot,
                max_segment_len=self.max_segment_length,
                max_total_response_len=self.config.response_length
            )
            tasks.append(task)
        
        results = loop.run_until_complete(asyncio.gather(*tasks))
        
        # Convert to tensors
        for response_ids, logprobs in results:
            all_responses.append(torch.tensor(response_ids, dtype=torch.long))
            all_logprobs.append(torch.tensor(logprobs, dtype=torch.float))
        
        # Pad sequences
        device = idx.device
        padded_responses = pad_sequence(
            all_responses, 
            batch_first=True, 
            padding_value=self.pad_token_id
        ).to(device)
        
        padded_logprobs = pad_sequence(
            all_logprobs, 
            batch_first=True, 
            padding_value=0.0
        ).to(device)
        
        # Ensure correct length
        if padded_responses.shape[1] < self.config.response_length:
            padded_responses = pad_sequence_to_length(
                padded_responses, 
                self.config.response_length, 
                self.pad_token_id
            )
            padded_logprobs = pad_sequence_to_length(
                padded_logprobs, 
                self.config.response_length, 
                0.0
            )
        elif padded_responses.shape[1] > self.config.response_length:
            padded_responses = padded_responses[:, :self.config.response_length]
            padded_logprobs = padded_logprobs[:, :self.config.response_length]
        
        # Create full sequences
        full_input_ids = torch.cat([idx, padded_responses], dim=1)
        
        # Update attention mask and position IDs
        response_attention_mask = (padded_responses != self.pad_token_id).long()
        full_attention_mask = torch.cat([attention_mask, response_attention_mask], dim=1)
        
        # Compute position IDs for responses
        prompt_last_position = position_ids[:, -1:] 
        response_length = padded_responses.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = prompt_last_position + delta_position_id
        full_position_ids = torch.cat([position_ids, response_position_ids], dim=1)
        
        # Create batch
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": padded_responses,
                "input_ids": full_input_ids,
                "rollout_log_probs": padded_logprobs,
                "attention_mask": full_attention_mask,
                "position_ids": full_position_ids,
            },
            batch_size=batch_size,
        )
        
        # Free cache if configured
        if self.config.free_cache_engine and self._engine is not None:
            self._engine.flush_cache()
        
        # Preserve non-tensor batch data
        non_tensor_batch = prompts.non_tensor_batch.copy()
        
        return DataProto(
            batch=batch, 
            non_tensor_batch=non_tensor_batch,
            meta_info=prompts.meta_info
        ) 