"""
Example workflow: MathChat-style multi-agent interaction
Pattern: Generator -> Coder -> Refiner -> Coder -> Refiner -> ...
"""
import os
from typing import Dict, List, Any, Optional
import json
import asyncio
from openrlhf.utils.logging_utils import init_loggerfrom openrlhf.verifiers.auto_verify import auto_verify
from marti.worlds.workflows.utils import apply_template_with_tokenizer
from openrlhf.verifiers.qwen.qwen_eval_timeout import qwen_reward_fn_timeout

logger = init_logger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))


async def workflow(
    prompt: str,
    label: str,
    agents: List[Dict[str, Any]],
    tool_manager,
    task: str,
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

    generator_prompt = "You are Agent Problem Solver, and your role is to collaborate with other agents to address various challenges.\nFor each problem, please follow these steps:\n    1. **Document Your Solution**: Write your solution step by step, ensuring it is independent of the solutions provided by other agents.\n    2. **Engage in Discussion**: Once you have outlined your solution, discuss your approach and findings with the other agents.\n\nProblem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

    coder_prompt = "You are Agent Code Executor. You can solve problems only writing commented Python code.\nFor each problem, please follow these steps:\n    1. **Develop Your Solution**: Write your solution in Python code, detailing each step independently from the solutions provided by other agents.\n    2. **Utilize SymPy**: Feel free to use the SymPy package to facilitate calculations and enhance your code's efficiency.\n    3. **Display Results**: Ensure that you **print the final result at the end of your Python code** (e.g., `print(_result_)`).\n    4. **Engage in Discussion**: After obtaining the result from your Python code, discuss your findings with the other agents.\nAlways format your Python code within:\n```python\n# your code here\nprint(_result_)\n```\n\nProblem: {problem}\n\nHere is the output from Agent Problem Solver:\n{solution}"

    refiner_prompt = "You are Agent Verifier.\nYour role is to critically evaluate the solutions proposed by other agents step by step and provide a final solution.\n    1. **Solution Requirement**: Before making any decisions, ensure you have received solutions from both Agent Code Executor and Agent Problem Solver.\n    2. **Avoid Assumptions**: Pay attention to the variables provided in the original problem statement versus those assumed by the agents. **Assumed values are not valid for the solution** and can lead to inaccuracies. Never base your solution on assumed values. Always base your solution on the explicitly given variables to ensure correctness. If a problem is deemed unsolvable due to missing information, return: **SOLUTION_FOUND \\boxed{{'None'}}**.\n    3. **Evaluating Conflicting Solutions**: If different answers are presented during the discussion, choose the most appropriate solution based on your evidence or initiate further discussion to clarify.\n    4. **Final Solution Declaration**: When you are confident about the final solution, return it as follows: **SOLUTION_FOUND \\boxed{{_solution_value_here_}}**. Ensure that only numerical values are placed inside the \\boxed{{}}; any accompanying text should be outside.\n\nProblem: {problem}\n\nHere is the output from Agent Problem Solver:\n{solution}\n\nHere is the output from Agent Code Executor:\n{execution}"

    trajectory = []

    # Generator
    generator_input = apply_template_with_tokenizer(
        generator_agent["tokenizer"],
        generator_prompt.format(problem=prompt)
    )
    generator_response = await generator_agent["llm"].generate_async.remote(
        generator_input,
        generator_agent["sampling_params"]
    )
    generator_output = generator_response.outputs[0].text
    trajectory.append({
        "turn_id": 0,
        "agent_index": 0,
        "agent_name": generator_agent["agent_id"],
        "agent_role": generator_agent["agent_role"],
        "agent_input": generator_input,
        "agent_output": generator_output,
        "metadata": {}
    })

    # Coder
    coder_input = apply_template_with_tokenizer(
        coder_agent["tokenizer"],
        coder_prompt.format(problem=prompt, solution=generator_output)
    )
    coder_response = await coder_agent["llm"].generate_async.remote(
        coder_input,
        coder_agent["sampling_params"]
    )
    coder_output = coder_response.outputs[0].text
    coder_content = coder_output.split("```python")[-1].split("```")[0].strip()
    # Execute any tools in coder output
    try:
        response_content, response_metadata = await tool_manager.execute_tool(
            "code_interpreter", {"code": coder_content}, metadata=metadata
        )
        status = response_metadata["status"]
    except Exception as e:
        response_content = f"ERROR"
        status = "failed"

    # status = response_metadata["status"]
    execution = coder_content + \
        f"\nExecution status: {status}\nCode output: {response_content[:512]}"
    trajectory.append({
        "turn_id": 1,
        "agent_index": 1,
        "agent_name": coder_agent["agent_id"],
        "agent_role": coder_agent["agent_role"],
        "agent_input": coder_input,
        "agent_output": coder_output,
        "metadata": {
            "status": status,
            "response": response_content
        }
    })

    # Refiner
    refiner_input = apply_template_with_tokenizer(
        refiner_agent["tokenizer"],
        refiner_prompt.format(
            problem=prompt, solution=generator_output, execution=execution)
    )
    refiner_response = await refiner_agent["llm"].generate_async.remote(
        refiner_input,
        refiner_agent["sampling_params"]
    )
    refiner_output = refiner_response.outputs[0].text
    trajectory.append({
        "turn_id": 2,
        "agent_index": 2,
        "agent_name":  refiner_agent["agent_id"],
        "agent_role": refiner_agent["agent_role"],
        "agent_input": refiner_input,
        "agent_output": refiner_output,
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
        turn["agent_reward"] = reward

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "final_reward": all_rewards[-1]
    }
