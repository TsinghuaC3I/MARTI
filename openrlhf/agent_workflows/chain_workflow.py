"""
Chain workflow: Sequential multi-agent interaction
Pattern: Generator -> Verifier -> Refiner
"""

import os
from typing import Dict, List, Any, Optional
import asyncio
from openrlhf.utils.logging_utils import init_logger
from openrlhf.verifiers.auto_verify import auto_verify
from openrlhf.verifiers.qwen.qwen_eval import majority_vote
from openrlhf.agent_workflows.utils import apply_template_with_tokenizer

logger = init_logger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))


def get_chat_template(agent: Dict[str, Any]) -> str:
    """
    Extract chat_template from agent configuration.
    
    Args:
        agent: Agent configuration dictionary
        
    Returns:
        Chat template string
    """
    # Try different possible locations for chat_template
    if "chat_template" in agent:
        template = agent["chat_template"]
        # If it's a list, take the first element
        if isinstance(template, list) and len(template) > 0:
            return template[0]
        elif isinstance(template, str):
            return template
    return ""


def format_template(template: str, question: str = "", generator: str = "", verifier: str = "") -> str:
    """
    Format template string by replacing variables.
    
    Args:
        template: Template string with variables like $question, $generator, $verifier
        question: Value for $question variable
        generator: Value for $generator variable
        verifier: Value for $verifier variable
        
    Returns:
        Formatted string
    """
    result = template
    result = result.replace("$question", question)
    result = result.replace("$generator", generator)
    result = result.replace("$verifier", verifier)
    return result


async def workflow(
    prompt: str,
    label: str,
    agents: List[Dict[str, Any]],
    tool_manager=None,
    task: str = None,
    metadata: Optional[Dict] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Chain workflow: Generator -> Verifier -> Refiner
    
    Args:
        prompt: Initial problem prompt
        label: Expected answer/label
        agents: List of agent configurations, each should have:
            - agent_id: Unique identifier
            - agent_role: Role name (generator/verifier/refiner)
            - chat_template: Template string or list with template
            - tokenizer: Tokenizer instance
            - llm: LLM instance with generate_async method
            - sampling_params: Sampling parameters
        tool_manager: Tool manager instance (optional)
        task: Task identifier
        metadata: Additional metadata
    """
    workflow_args = kwargs.get("workflow_args", {})
    
    # Get task from workflow_args if not provided
    if task is None:
        task = workflow_args.get("task", "math")
    
    # Find agents by role
    generator_agent = None
    verifier_agent = None
    refiner_agent = None
    
    for agent in agents:
        role = agent.get("agent_role", "").lower()
        if role == "generator":
            generator_agent = agent
        elif role == "verifier":
            verifier_agent = agent
        elif role == "refiner":
            refiner_agent = agent
    
    # Fallback to index-based if roles not found
    if generator_agent is None and len(agents) >= 1:
        generator_agent = agents[0]
    if verifier_agent is None and len(agents) >= 2:
        verifier_agent = agents[1]
    if refiner_agent is None and len(agents) >= 3:
        refiner_agent = agents[2]
    
    # Ensure we have all required agents
    if generator_agent is None:
        raise ValueError("Generator agent not found")
    if verifier_agent is None:
        raise ValueError("Verifier agent not found")
    if refiner_agent is None:
        raise ValueError("Refiner agent not found")
    
    # Get chat templates
    generator_template = get_chat_template(generator_agent)
    verifier_template = get_chat_template(verifier_agent)
    refiner_template = get_chat_template(refiner_agent)
    
    # If templates are empty, use default templates
    if not generator_template:
        generator_template = "$question\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    if not verifier_template:
        verifier_template = "You are tasked with analyzing an answer to a problem and providing constructive feedback.\n\nProblem: $question\n\nSolution: $generator\n\nDo NOT provide direct solutions."
    if not refiner_template:
        refiner_template = "You are tasked with revising a draft solution to a problem based on the critique provided. Please provide a revised solution that addresses the feedback and improves the overall quality of the solution.\n\nProblem: $question\n\nSolution: $generator\n\nCritique: $verifier\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    
    trajectory = []
    rewards = []
    
    # Step 1: Generator produces initial solution
    generator_prompt = format_template(generator_template, question=prompt)
    generator_input_prompt = apply_template_with_tokenizer(
        generator_agent["tokenizer"], generator_prompt
    )
    generator_input_token_ids = generator_agent["tokenizer"](
        generator_input_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].tolist()
    
    generator_result = await generator_agent["llm"].generate_async.remote(
        prompt_ids=generator_input_token_ids,
        sampling_params=generator_agent["sampling_params"]
    )
    generator_output = generator_result.outputs[0].text
    generator_output_token_ids = generator_result.outputs[0].token_ids
    generator_sequence_ids = generator_input_token_ids + generator_output_token_ids
    
    # Extract rollout_log_probs for generator
    generator_rollout_log_probs = None
    if generator_agent["sampling_params"].logprobs is not None:
        generator_rollout_log_probs = [0.0] * len(generator_input_token_ids)
        if hasattr(generator_result.outputs[0], "logprobs") and generator_result.outputs[0].logprobs is not None:
            for i, logprob_dict in enumerate(generator_result.outputs[0].logprobs):
                if i < len(generator_output_token_ids) and generator_output_token_ids[i] in logprob_dict:
                    generator_rollout_log_probs.append(logprob_dict[generator_output_token_ids[i]].logprob)
                else:
                    generator_rollout_log_probs.append(0.0)
        else:
            generator_rollout_log_probs.extend([0.0] * len(generator_output_token_ids))
    
    generator_reward = auto_verify(task, 1, [generator_output], [label])[0]
    
    trajectory.append({
        "turn_id": 0,
        "agent_index": 0,
        "agent_id": generator_agent["agent_id"],
        "agent_name": generator_agent["agent_id"],
        "agent_role": generator_agent.get("agent_role", "generator"),
        "agent_input": generator_input_prompt,
        "agent_output": generator_output,
        "output_ids": generator_output_token_ids,
        "sequence_ids": generator_sequence_ids,
        "rollout_log_prob": generator_rollout_log_probs,
        "reward": generator_reward,
        "metadata": {},
    })
    rewards.append([generator_reward])
    
    # Step 2: Verifier analyzes generator's solution
    verifier_prompt = format_template(
        verifier_template, 
        question=prompt, 
        generator=generator_output
    )
    verifier_input_prompt = apply_template_with_tokenizer(
        verifier_agent["tokenizer"], verifier_prompt
    )
    verifier_input_token_ids = verifier_agent["tokenizer"](
        verifier_input_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].tolist()
    
    verifier_result = await verifier_agent["llm"].generate_async.remote(
        prompt_ids=verifier_input_token_ids,
        sampling_params=verifier_agent["sampling_params"]
    )
    verifier_output = verifier_result.outputs[0].text
    verifier_output_token_ids = verifier_result.outputs[0].token_ids
    verifier_sequence_ids = verifier_input_token_ids + verifier_output_token_ids
    
    # Extract rollout_log_probs for verifier
    verifier_rollout_log_probs = None
    if verifier_agent["sampling_params"].logprobs is not None:
        verifier_rollout_log_probs = [0.0] * len(verifier_input_token_ids)
        if hasattr(verifier_result.outputs[0], "logprobs") and verifier_result.outputs[0].logprobs is not None:
            for i, logprob_dict in enumerate(verifier_result.outputs[0].logprobs):
                if i < len(verifier_output_token_ids) and verifier_output_token_ids[i] in logprob_dict:
                    verifier_rollout_log_probs.append(logprob_dict[verifier_output_token_ids[i]].logprob)
                else:
                    verifier_rollout_log_probs.append(0.0)
        else:
            verifier_rollout_log_probs.extend([0.0] * len(verifier_output_token_ids))
    
    verifier_reward = auto_verify(task, 1, [verifier_output], [label])[0]
    
    trajectory.append({
        "turn_id": 1,
        "agent_index": 1,
        "agent_id": verifier_agent["agent_id"],
        "agent_name": verifier_agent["agent_id"],
        "agent_role": verifier_agent.get("agent_role", "verifier"),
        "agent_input": verifier_input_prompt,
        "agent_output": verifier_output,
        "output_ids": verifier_output_token_ids,
        "sequence_ids": verifier_sequence_ids,
        "rollout_log_prob": verifier_rollout_log_probs,
        "reward": verifier_reward,
        "metadata": {},
    })
    rewards.append([verifier_reward])
    
    # Step 3: Refiner improves solution based on verifier's feedback
    refiner_prompt = format_template(
        refiner_template,
        question=prompt,
        generator=generator_output,
        verifier=verifier_output
    )
    refiner_input_prompt = apply_template_with_tokenizer(
        refiner_agent["tokenizer"], refiner_prompt
    )
    refiner_input_token_ids = refiner_agent["tokenizer"](
        refiner_input_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].tolist()
    
    refiner_result = await refiner_agent["llm"].generate_async.remote(
        prompt_ids=refiner_input_token_ids,
        sampling_params=refiner_agent["sampling_params"]
    )
    refiner_output = refiner_result.outputs[0].text
    refiner_output_token_ids = refiner_result.outputs[0].token_ids
    refiner_sequence_ids = refiner_input_token_ids + refiner_output_token_ids
    
    # Extract rollout_log_probs for refiner
    refiner_rollout_log_probs = None
    if refiner_agent["sampling_params"].logprobs is not None:
        refiner_rollout_log_probs = [0.0] * len(refiner_input_token_ids)
        if hasattr(refiner_result.outputs[0], "logprobs") and refiner_result.outputs[0].logprobs is not None:
            for i, logprob_dict in enumerate(refiner_result.outputs[0].logprobs):
                if i < len(refiner_output_token_ids) and refiner_output_token_ids[i] in logprob_dict:
                    refiner_rollout_log_probs.append(logprob_dict[refiner_output_token_ids[i]].logprob)
                else:
                    refiner_rollout_log_probs.append(0.0)
        else:
            refiner_rollout_log_probs.extend([0.0] * len(refiner_output_token_ids))
    
    refiner_reward = auto_verify(task, 1, [refiner_output], [label])[0]
    
    trajectory.append({
        "turn_id": 2,
        "agent_index": 2,
        "agent_id": refiner_agent["agent_id"],
        "agent_name": refiner_agent["agent_id"],
        "agent_role": refiner_agent.get("agent_role", "refiner"),
        "agent_input": refiner_input_prompt,
        "agent_output": refiner_output,
        "output_ids": refiner_output_token_ids,
        "sequence_ids": refiner_sequence_ids,
        "rollout_log_prob": refiner_rollout_log_probs,
        "reward": refiner_reward,
        "metadata": {},
    })
    rewards.append([refiner_reward])
    
    # Calculate final reward using majority vote on all solutions
    all_solutions = [generator_output, verifier_output, refiner_output]
    final_reward = majority_vote(
        solutions=all_solutions,
        ground_truth=label,
        task=task,
    )
    
    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "reward_matrix": rewards,
        "final_reward": final_reward,
    }

