"""
Single-round RL workflow for code tasks (without MCTS)
This workflow performs a single generation-evaluation cycle for code problems.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

from openrlhf.utils.logging_utils import init_logger
from openrlhf.agent_workflows.utils import apply_template_with_tokenizer
from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.llm_generation_interface import (
    GenerationRequest,
    GenerationResult,
)
from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.tasks.code.task import CodeProblem
from openrlhf.agent_workflows.third_party.mcts_utils.code_prompt import problem_prompt
from vllm import SamplingParams

logger = init_logger(__name__)


async def workflow(
    workflow_args,
    prompt: str,
    label: str,
    agents: List[Dict[str, Any]],
    metadata: str = None,
    is_eval=False,
    prompt_id=0,
    task: str = "code",
    **kwargs,
) -> Dict[str, Any]:
    """
    Single-round RL workflow for code tasks.
    
    This workflow:
    1. Takes a code problem prompt
    2. Uses the first agent to generate a solution
    3. Evaluates the solution using CodeProblem
    4. Returns the trajectory in the same format as ab_mcts_workflow
    
    Args:
        workflow_args: Workflow configuration arguments
        prompt: Code problem prompt
        label: Ground truth answer (can be string or dict)
        agents: List of agent configurations (uses first agent)
        metadata: Additional metadata (can be string or dict)
        is_eval: Whether this is an evaluation run
        prompt_id: Unique identifier for the prompt
        task: Task identifier (should be "code")
    
    Returns:
        Dictionary with:
            - prompt: Original prompt
            - label: Ground truth label
            - trajectory: List of node records (single entry for this workflow)
            - final_reward: Final reward value
    """
    start_time = time.time()
    
    # Validate task type
    if task != "code":
        logger.warning(f"Task type is '{task}', but this workflow is designed for 'code' tasks")
    
    # Parse metadata
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
    elif metadata is None:
        metadata = {}
    
    # Parse label - for code tasks, label can be the ground truth answer
    if isinstance(label, str):
        try:
            label_dict = json.loads(label)
            if isinstance(label_dict, dict):
                ground_truth = label_dict.get("ground_truth", label)
            else:
                ground_truth = label
        except (json.JSONDecodeError, TypeError):
            ground_truth = label
    elif isinstance(label, dict):
        ground_truth = label.get("ground_truth", label)
    else:
        ground_truth = label
    
    # Ensure ground_truth is a list (as expected by CodeProblem)
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    
    # Parse agents
    assert isinstance(agents, list), f"agents must be list, but got {type(agents)}"
    for idx, agent in enumerate(agents):
        if not isinstance(agent, dict):
            try:
                agents[idx] = json.loads(agent)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to parse agent at index {idx}, using empty dict")
                agents[idx] = {}
    assert len(agents) > 0 and isinstance(agents[0], dict), "agents must be non-empty list of dicts"
    
    # Use the first agent for generation
    agent = agents[0]
    agent_id = agent.get("agent_id", "0")
    agent_role = agent.get("agent_role", "generator")
    llm = agent.get("llm")
    tokenizer = agent.get("tokenizer")
    sampling_params = agent.get("sampling_params")
    
    assert llm is not None, "LLM engine is required"
    assert tokenizer is not None, "Tokenizer is required"
    assert sampling_params is not None, "Sampling parameters are required"
    assert isinstance(sampling_params, SamplingParams), "sampling_params must be SamplingParams instance"
    
    # Create CodeProblem instance
    # CodeProblem expects: prompt, ground_truth, meta_data, indice
    indice = str(prompt_id) if prompt_id else "0"
    code_problem = CodeProblem.load_data(
        prompt=prompt,
        ground_truth=ground_truth,
        meta_data=metadata,
        indice=indice,
    )
    
    # Build the code prompt (add instruction suffix)
    code_prompt = problem_prompt(code_problem)
    
    # Apply chat template
    input_messages = [{"role": "user", "content": code_prompt}]
    formatted_prompt = apply_template_with_tokenizer(tokenizer, code_prompt)
    
    # Tokenize the input prompt
    input_token_ids = tokenizer(
        formatted_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].tolist()
    
    # Generate response using LLM
    logger.info(f"Generating response for prompt_id={prompt_id}")
    # Use async generation (standard for this workflow)
    request_output = await llm.generate_async.remote(
        prompt_ids=input_token_ids,
        sampling_params=sampling_params,
    )
    
    # Extract generation results
    action_text = request_output.outputs[0].text
    action_tokens = request_output.outputs[0].token_ids
    
    # Build sequence_ids (input + output tokens)
    sequence_ids = input_token_ids + action_tokens
    
    # Calculate rollout_log_probs if logprobs are available
    # Following the same pattern as ab_mcts_workflow.py and utils.py
    if sampling_params.logprobs is not None:
        rollout_log_probs = [0.0] * len(input_token_ids)  # Input tokens have logprob = 0.0
        # Output tokens have their actual logprobs
        if hasattr(request_output.outputs[0], "logprobs") and request_output.outputs[0].logprobs is not None:
            # logprobs is a list of dicts, where each dict maps token_id -> LogprobData
            for i, logprob_dict in enumerate(request_output.outputs[0].logprobs):
                if i < len(action_tokens) and action_tokens[i] in logprob_dict:
                    # Get the logprob value from LogprobData object
                    rollout_log_probs.append(logprob_dict[action_tokens[i]].logprob)
                else:
                    rollout_log_probs.append(0.0)
        else:
            # If logprobs not available, create dummy list for output tokens
            rollout_log_probs.extend([0.0] * len(action_tokens))
    else:
        # If logprobs not available, create dummy list with zeros
        rollout_log_probs = [0.0] * len(sequence_ids)
    
    # Create GenerationRequest with messages
    # Use the original input_messages format (list of dicts) for consistency
    gen_request = GenerationRequest(messages=input_messages)
    
    # Create GenerationResult
    # For code tasks, generation should be the same as action_text
    gen_result = GenerationResult(
        request=gen_request,
        action_text=action_text,
        action_tokens=action_tokens,
        sequence_ids=sequence_ids,
        rollout_log_probs=rollout_log_probs,
        agent_id=agent_id,
    )
    
    # Set generation attribute for compatibility with CodeProblem.evaluate_on_test
    # CodeProblem uses llm_answer.generation, but GenerationResult has action_text
    # We add generation as an attribute that points to action_text
    gen_result.generation = action_text
    
    # Evaluate the generation using CodeProblem
    # Use generate_eval_results for consistency with ab_mcts_workflow.py
    logger.info(f"Evaluating generation for prompt_id={prompt_id}")
    eval_results = code_problem.generate_eval_results(llm_answer=gen_result, kind="transform")
    
    # Calculate reward as average score from eval_results (same as ab_mcts_workflow.py)
    if eval_results is None or len(eval_results) == 0:
        final_reward = 0.0
        logger.warning(f"No eval results for prompt_id={prompt_id}, setting reward to 0.0")
    else:
        final_reward = sum([eval_result.get_score() for eval_result in eval_results]) / len(eval_results)
    
    # Create trajectory record (compatible with ab_mcts_workflow format)
    node_record = {
        "node_id": 0,
        "agent_id": agent_id,
        "agent_role": agent_role,
        "input": input_messages,  # Format: list of message dicts
        "output": action_text,
        "output_ids": action_tokens,
        "sequence_ids": sequence_ids,
        "reward": final_reward,
        "rollout_log_prob": rollout_log_probs,
        "metadata": {
            "expand_idx": 0,  # Single node, so expand_idx is 0
            "timestamp": time.time(),
        },
    }
    
    trajectory = [node_record]
    
    # Final reward for compatibility (same as node reward for single-round workflow)
    final_reward_list = [final_reward, final_reward]  # Format: [coverage_reward, best_reward]
    
    end_time = time.time()
    logger.info(
        f"Completed single-round workflow for prompt_id={prompt_id}, "
        f"reward={final_reward:.4f}, time={end_time - start_time:.2f}s"
    )
    
    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "final_reward": final_reward_list,
    }

