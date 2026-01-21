"""
Example workflow: MathChat-style multi-agent interaction
Pattern: Generator -> Coder -> Refiner -> Coder -> Refiner -> ...
"""
import os
from typing import Dict, List, Any, Optional
import json
import asyncio
from marti.utils.logging_utils import init_logger
# from marti.verifiers.auto_verify import auto_verify
from marti.agent_workflows.utils import apply_template_with_tokenizer
from marti.verifiers.qwen.qwen_eval_timeout import qwen_reward_fn_timeout

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
    tool_manager,
    task: str="math",
    metadata: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    MathChat workflow: Generator -> Coder -> Refiner cycle

    Args:
        prompt: Initial problem prompt
        label: Expected answer/label
        agents: List of agent configurations
        tool_manager: Tool manager instance
        max_length: Maximum token length
        task: Task identifier
        metadata: Additional metadata
        default_sampling_params: Default sampling parameters
    """

    assert tool_manager is not None

    # Identify agents by role
    generator_agent = agents[0]
    coder_agent = agents[1]
    refiner_agent = agents[2]

    # Get chat templates
    generator_prompt = get_chat_template(generator_agent)
    coder_prompt = get_chat_template(coder_agent)
    refiner_prompt = get_chat_template(refiner_agent)

    # If templates are empty, use default templates
    if not generator_prompt:
        generator_prompt = "You are Agent Problem Solver, and your role is to collaborate with other agents to address various challenges.\nFor each problem, please follow these steps:\n    1. **Document Your Solution**: Write your solution step by step, ensuring it is independent of the solutions provided by other agents.\n    2. **Engage in Discussion**: Once you have outlined your solution, discuss your approach and findings with the other agents.\n\nProblem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    if not coder_prompt:
        coder_prompt = "You are Agent Code Executor. You can solve problems only writing commented Python code.\nFor each problem, please follow these steps:\n    1. **Develop Your Solution**: Write your solution in Python code, detailing each step independently from the solutions provided by other agents.\n    2. **Utilize SymPy**: Feel free to use the SymPy package to facilitate calculations and enhance your code's efficiency.\n    3. **Display Results**: Ensure that you **print the final result at the end of your Python code** (e.g., `print(_result_)`).\n    4. **Engage in Discussion**: After obtaining the result from your Python code, discuss your findings with the other agents.\nAlways format your Python code within:\n```python\n# your code here\nprint(_result_)\n```\n\nProblem: {problem}\n\nHere is the output from Agent Problem Solver:\n{solution}"
    if not refiner_prompt:
        refiner_prompt = "You are Agent Verifier.\nYour role is to critically evaluate the solutions proposed by other agents step by step and provide a final solution.\n    1. **Solution Requirement**: Before making any decisions, ensure you have received solutions from both Agent Code Executor and Agent Problem Solver.\n    2. **Avoid Assumptions**: Pay attention to the variables provided in the original problem statement versus those assumed by the agents. **Assumed values are not valid for the solution** and can lead to inaccuracies. Never base your solution on assumed values. Always base your solution on the explicitly given variables to ensure correctness. If a problem is deemed unsolvable due to missing information, return: **SOLUTION_FOUND \\boxed{{'None'}}**.\n    3. **Evaluating Conflicting Solutions**: If different answers are presented during the discussion, choose the most appropriate solution based on your evidence or initiate further discussion to clarify.\n    4. **Final Solution Declaration**: When you are confident about the final solution, return it as follows: **SOLUTION_FOUND \\boxed{{_solution_value_here_}}**. Ensure that only numerical values are placed inside the \\boxed{{}}; any accompanying text should be outside.\n\nProblem: {problem}\n\nHere is the output from Agent Problem Solver:\n{solution}\n\nHere is the output from Agent Code Executor:\n{execution}"

    trajectory = []

    # Generator
    generator_input_prompt = apply_template_with_tokenizer(
        generator_agent["tokenizer"],
        generator_prompt.format(problem=prompt)
    )

    generator_input_token_ids = generator_agent["tokenizer"](
        generator_input_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].tolist()

    generator_response = await generator_agent["llm"].generate_async.remote(
        prompt_ids=generator_input_token_ids,
        sampling_params=generator_agent["sampling_params"]
    )
    generator_output = generator_response.outputs[0].text
    generator_output_token_ids = generator_response.outputs[0].token_ids
    generator_sequence_ids = generator_input_token_ids + generator_output_token_ids

    # Extract rollout_log_probs for generator
    generator_rollout_log_probs = None
    if generator_agent["sampling_params"].logprobs is not None:
        generator_rollout_log_probs = [0.0] * len(generator_input_token_ids)
        if hasattr(generator_response.outputs[0], "logprobs") and generator_response.outputs[0].logprobs is not None:
            for i, logprob_dict in enumerate(generator_response.outputs[0].logprobs):
                if i < len(generator_output_token_ids) and generator_output_token_ids[i] in logprob_dict:
                    generator_rollout_log_probs.append(logprob_dict[generator_output_token_ids[i]].logprob)
                else:
                    generator_rollout_log_probs.append(0.0)
        else:
            generator_rollout_log_probs.extend([0.0] * len(generator_output_token_ids))

    trajectory.append({
        "turn_id": 0,
        "agent_index": 0,
        "agent_id": generator_agent["agent_id"],
        "agent_name": generator_agent["agent_id"],
        "agent_role": generator_agent["agent_role"],
        "agent_input": generator_input_prompt,
        "agent_output": generator_output,
        "output_ids": generator_output_token_ids,
        "sequence_ids": generator_sequence_ids,
        "rollout_log_prob": generator_rollout_log_probs,
        "metadata": {}
    })

    # Coder
    coder_input_prompt = apply_template_with_tokenizer(
        coder_agent["tokenizer"],
        coder_prompt.format(problem=prompt, solution=generator_output)
    )
    coder_input_token_ids = coder_agent["tokenizer"](
        coder_input_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].tolist()

    coder_response = await coder_agent["llm"].generate_async.remote(
        prompt_ids=coder_input_token_ids,
        sampling_params=coder_agent["sampling_params"]
    )
    coder_output = coder_response.outputs[0].text
    coder_output_token_ids = coder_response.outputs[0].token_ids
    coder_sequence_ids = coder_input_token_ids + coder_output_token_ids
    coder_content = coder_output.split("```python")[-1].split("```")[0].strip()

    # Extract rollout_log_probs for coder
    coder_rollout_log_probs = None
    if coder_agent["sampling_params"].logprobs is not None:
        coder_rollout_log_probs = [0.0] * len(coder_input_token_ids)
        if hasattr(coder_response.outputs[0], "logprobs") and coder_response.outputs[0].logprobs is not None:
            for i, logprob_dict in enumerate(coder_response.outputs[0].logprobs):
                if i < len(coder_output_token_ids) and coder_output_token_ids[i] in logprob_dict:
                    coder_rollout_log_probs.append(logprob_dict[coder_output_token_ids[i]].logprob)
                else:
                    coder_rollout_log_probs.append(0.0)
        else:
            coder_rollout_log_probs.extend([0.0] * len(coder_output_token_ids))

    # Execute any tools in coder output
    try:
        response_content, response_metadata = await tool_manager.execute_tool(
            "code_interpreter", {"code": coder_content}, metadata=metadata
        )
        status = response_metadata["status"]
    except Exception as e:
        response_content = f"ERROR"
        status = "failed"

    execution = coder_content + \
        f"\nExecution status: {status}\nCode output: {response_content[:512]}"
    trajectory.append({
        "turn_id": 1,
        "agent_index": 1,
        "agent_id": coder_agent["agent_id"],
        "agent_name": coder_agent["agent_id"],
        "agent_role": coder_agent["agent_role"],
        "agent_input": coder_input_prompt,
        "agent_output": coder_output,
        "output_ids": coder_output_token_ids,
        "sequence_ids": coder_sequence_ids,
        "rollout_log_prob": coder_rollout_log_probs,
        "metadata": {
            "status": status,
            "response": response_content
        }
    })

    # Refiner
    refiner_input_prompt = apply_template_with_tokenizer(
        refiner_agent["tokenizer"],
        refiner_prompt.format(
            problem=prompt, solution=generator_output, execution=execution)
    )
    refiner_input_token_ids = refiner_agent["tokenizer"](
        refiner_input_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].tolist()

    refiner_response = await refiner_agent["llm"].generate_async.remote(
        prompt_ids=refiner_input_token_ids,
        sampling_params=refiner_agent["sampling_params"]
    )
    refiner_output = refiner_response.outputs[0].text
    refiner_output_token_ids = refiner_response.outputs[0].token_ids
    refiner_sequence_ids = refiner_input_token_ids + refiner_output_token_ids

    # Extract rollout_log_probs for refiner
    refiner_rollout_log_probs = None
    if refiner_agent["sampling_params"].logprobs is not None:
        refiner_rollout_log_probs = [0.0] * len(refiner_input_token_ids)
        if hasattr(refiner_response.outputs[0], "logprobs") and refiner_response.outputs[0].logprobs is not None:
            for i, logprob_dict in enumerate(refiner_response.outputs[0].logprobs):
                if i < len(refiner_output_token_ids) and refiner_output_token_ids[i] in logprob_dict:
                    refiner_rollout_log_probs.append(logprob_dict[refiner_output_token_ids[i]].logprob)
                else:
                    refiner_rollout_log_probs.append(0.0)
        else:
            refiner_rollout_log_probs.extend([0.0] * len(refiner_output_token_ids))

    trajectory.append({
        "turn_id": 2,
        "agent_index": 2,
        "agent_id": refiner_agent["agent_id"],
        "agent_name": refiner_agent["agent_id"],
        "agent_role": refiner_agent["agent_role"],
        "agent_input": refiner_input_prompt,
        "agent_output": refiner_output,
        "output_ids": refiner_output_token_ids,
        "sequence_ids": refiner_sequence_ids,
        "rollout_log_prob": refiner_rollout_log_probs,
        "metadata": {}
    })

    # Verify final solution
    all_outputs = [
        generator_output,
        f"Answer is \\boxed{response_content.strip()}" if status else "Answer is \\boxed{None}",
        refiner_output
    ]
    # all_rewards = auto_verify(task, 1, all_outputs, [label] * len(all_outputs))
    all_rewards = [qwen_reward_fn_timeout(output, label) for output in all_outputs]

    for turn, reward in zip(trajectory, all_rewards):
        # turn["agent_reward"] = reward
        turn["reward"] = reward

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "reward_matrix": all_rewards,
        "final_reward": all_rewards[-1]
    }
