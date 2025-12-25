"""
Tool workflow for multi-turn tool-calling tasks.
This workflow enables multi-turn generation with tool calls (code interpreter, search, etc.)
for various tasks like math problems, QA with search, and more.

"""

import os
import json
import time
import srsly
from typing import Dict, List, Any, Optional, Tuple

from openrlhf.utils.logging_utils import init_logger
from openrlhf.agent_workflows.utils import apply_template_with_tokenizer
from openrlhf.verifiers.auto_verify import auto_verify

# from openrlhf.openrlhf.tools.manager import ToolManager
# from openrlhf.agent_workflows.tools.search import SearchToolExecutor
# from openrlhf.agent_workflows.tools.sandbox import SandboxFusionExecutor

logger = init_logger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))

def assign_action_mask(turn: str) -> int:
    """
    Assign action mask based on turn content.
    Returns 0 for tool response turns (don't compute gradient),
    Returns 1 for model output turns (compute gradient).
    """

    has_tool_response = "\n<|im_start|>user\n<tool_response>" in turn and "</tool_response><|im_end|>\n<|im_start|>assistant" in turn
    has_information = "\n<|im_start|>user\n<information>" in turn and "</information><|im_end|>\n<|im_start|>assistant" in turn
    
    if has_tool_response or has_information:
        return 0
    else:
        return 1

async def workflow(
    workflow_args: Dict,
    prompt: str,
    label: str,
    agents: List[Dict[str, Any]],
    max_length: int,
    tool_manager=None,
    metadata: str = None,
    is_eval: bool = False,
    prompt_id: int = 0,
    task: str = "math",
    **kwargs,
) -> Dict[str, Any]:
    """
    TIR workflow for tasks with tool-integrated reasoning.
    
    This workflow:
    1. Takes a problem prompt
    2. Allows multi-turn generation with tool calls
    3. Evaluates the final solution using auto_verify
    4. Returns the trajectory
    
    Args:
        workflow_args: Workflow configuration including tools_config
        prompt: Problem prompt
        label: Ground truth answer
        agents: List of agent configurations
        metadata: Additional metadata
        is_eval: Whether this is an evaluation run
        prompt_id: Unique identifier for the prompt
        task: Task identifier 
    
    Returns:
        Dictionary with prompt, label, trajectory, and final_reward
    """
    start_time = time.time()
    
    # Parse metadata
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
    elif metadata is None:
        metadata = {}
    
    # Parse agents
    assert isinstance(agents, list), f"agents must be list, but got {type(agents)}"
    for idx, agent in enumerate(agents):
        if not isinstance(agent, dict):
            try:
                agents[idx] = json.loads(agent)
            except (json.JSONDecodeError, TypeError):
                agents[idx] = {}
    assert len(agents) > 0 and isinstance(agents[0], dict), "agents must be non-empty list of dicts"
    
    # Get configuration
    # Get task from workflow_args, metadata, or use default
    # Priority: explicit task parameter > workflow_args > metadata > default "math"
    # if task == "math":  # Only use default if not explicitly provided
    task = workflow_args.get("task", metadata.get("task", "math") if isinstance(metadata, dict) else "math")
    max_turns = workflow_args.get("max_turns", 2)
    chat_template = workflow_args.get("chat_template", 
        "Answer the given math problem. You must conduct reasoning inside <think> and </think> first "
        "every time you get new information or need to solve a subproblem. After reasoning, if you find "
        "that a calculation or code is needed, you can use the code interpreter inside <tool_call> and "
        "</tool_call>; it will return the result between <tool_response> and </tool_response>. You can "
        "run as many code cells as needed. If you find no further computation is necessary, put your "
        "final answer within \\boxed{{}}. Question: {question}"
    )
    agent_step = kwargs.get("agent_step", None)
    assert agent_step is not None, f"tool workflow must have agent_step as tool"
    assert tool_manager is not None, f"tool workflow needs tool_manager"

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
    
    # Format prompt with chat template
    formatted_question = chat_template.format(question=prompt)
    input_prompt = apply_template_with_tokenizer(
        tokenizer,
        formatted_question,
        tools=tool_manager.tools,
        enable_thinking=agent.get("enable_thinking", False),
    )
    
    # Multi-turn generation loop
    observation = [input_prompt]
    all_outputs = []
    action_mask = []
    tools_used = {}
    turn_count = 0
    done = False

    input_ids = tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
    observation_tokens_len = len(input_ids)
    sequence_ids = input_ids
    output_ids = None

    # setting initial rollout log probs
    rollout_log_probs = None
    if sampling_params.logprobs is not None:
        rollout_log_probs = []
        rollout_log_probs.append([0.0] * observation_tokens_len)
        rollout_log_probs_list = []
        action_ids = []

    while not done and turn_count < max_turns:
        # Current context is the full conversation so far
        current_context = "".join(observation)

        # Tokenize and generate
        input_ids = tokenizer(current_context, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        observation_tokens_len = len(input_ids)

        if observation_tokens_len >= max_length:
            logger.warning(f"Trajectory exceeded max length at turn {turn_count}")
            break

        # Generate response asynchronously
        request_output = await llm.generate_async.remote(
            prompt_ids=input_ids,
            sampling_params=sampling_params,
        )
        action_text = request_output.outputs[0].text
        action_tokens = request_output.outputs[0].token_ids

        result = await agent_step(observation, action_text, tool_manager, metadata=metadata, label=label)

        # Collect tool usage
        step_tools = result.get("extra_logs", {}).get("tools_used", {})
        for tool, count in step_tools.items():
            tools_used[tool] = tools_used.get(tool, 0) + count

        # Calculate rollout log probs
        if sampling_params.logprobs is not None:
            current_action_log_probs = []
            assert len(action_tokens) == len(request_output.outputs[0].logprobs)
            # action tokens logprobs
            for i, logprob in enumerate(request_output.outputs[0].logprobs):
                current_action_log_probs.append(logprob[action_tokens[i]].logprob)
            # Collect output ids
            rollout_log_probs_list.append(current_action_log_probs)
            action_ids.append(action_tokens)
        observation = result["next_observation"]
        done = result["done"]
        turn_count += 1
    
    # Evaluate final answer
    if "final_reward" in result:
        final_reward = result["final_reward"]
    else:
        final_reward = auto_verify(task, 1, ["".join(observation[1:])], [label])[0]
    final_reward = float(final_reward)

    # process data
    output_token_ids = [
        tokenizer(output, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        for output in observation[1:] if len(output) > 0
    ]
    all_output_ids = []
    for output_ids in output_token_ids:
        all_output_ids.extend(output_ids)
        sequence_ids.extend(output_ids)

    action_masks_list = [assign_action_mask(out) for out in observation[1:] if len(out) > 0]
    rollout_idx = 0
    current_action_mask = []

    for action, output in zip(action_masks_list, output_token_ids):
        if sampling_params.logprobs is not None:
            if action == 1:
                rollout_log_probs.append(rollout_log_probs_list[rollout_idx])
                rollout_idx += 1
            else:
                rollout_log_probs.append([0.0] * len(output))
        current_action_mask.append([action]*len(output))
    if sampling_params.logprobs is not None:
        rollout_log_probs = sum(rollout_log_probs, [])

    action_mask = [v for sublist in current_action_mask for v in sublist]
    output_token_ids = sum(output_token_ids, [])

    # assert len(sequence_ids) == len(action_mask) == len(rollout_log_probs)
    # assert sum(action_mask) == sum(len(x) for x in rollout_log_probs_list)

    # Create trajectory record compatible with MARTI training format
    trajectory = {
        "node_id": 0,
        "agent_id": agent_id,
        "agent_role": agent_role,
        "agent_input": input_prompt,
        "agent_output": "".join(observation[1:]),
        "output_ids": all_output_ids,  
        "sequence_ids": sequence_ids,
        "action_mask": action_mask,
        "reward": final_reward,
        "rollout_log_prob": rollout_log_probs,
        "metadata": {},
    }
    
    trajectory_list = [trajectory]
    
    end_time = time.time()
    logger.info(
        f"Completed TIR workflow for prompt_id={prompt_id}, "
        f"reward={final_reward:.4f}, turns={turn_count}, time={end_time - start_time:.2f}s"
    )
    
    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory_list,
        "final_reward": final_reward,
    }
