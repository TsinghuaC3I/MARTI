"""
Example workflow: MathChat-style multi-agent interaction
Pattern: Generator -> Coder -> Refiner -> Coder -> Refiner -> ...
"""

import os
import random
from typing import Dict, List, Any, Optional
import json
import asyncio
import ray
from openrlhf.utils.logging_utils import init_logger
from openrlhf.verifiers.auto_verify import auto_verify
from openrlhf.verifiers.qwen.qwen_eval import majority_vote

logger = init_logger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))


def apply_template_with_tokenizer(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


async def workflow(
    prompt: str,
    label: str,
    agents: List[Dict[str, Any]],
    tool_manager,
    task: str,
    metadata: Optional[Dict] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    MathChat workflow: Generator -> Coder -> Refiner cycle

    Args:
        prompt: Initial problem prompt
        label: Expected answer/label
        agents: List of agent configurations
        tool_manager: Tool manager instance
        task: Task identifier
        metadata: Additional metadata
    """
    workflow_args = kwargs.get("workflow_args", {})

    initial_prompt = f"{prompt}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

    debate_template = """Here are solutions from other agents:

{responses_str}

Using these responses as additional advice, can you give an updated bullet by bullet answer to the following question:
{question}

Please reason step by step, and put your final answer within \\boxed{{}}."""

    num_agents = len(agents)
    num_rounds = workflow_args.get("num_rounds", num_agents)
    max_others = workflow_args.get("max_others", 5)
    contain_self = workflow_args.get("contain_self", True)
    shuffle_responses = workflow_args.get("shuffle_responses", True)

    trajectory = [[] for _ in range(num_agents)]
    rewards = [[] for _ in range(num_agents)]

    for num_round in range(num_rounds):
        if num_round == 0:
            agent_inputs = [initial_prompt for _ in agents]
        else:
            prev_responses = [agent[-1]["agent_output"] for agent in trajectory]
            agent_inputs = []
            for aid in range(num_agents):
                if contain_self:
                    other_responses = prev_responses
                else:
                    other_responses = prev_responses[:aid] + prev_responses[aid + 1 :]

                if shuffle_responses:
                    random.shuffle(other_responses)

                other_responses = other_responses[:max_others]

                responses_str = "\n\n".join(
                    [f"Agent {i+1}: {resp}" for i, resp in enumerate(other_responses)]
                )

                agent_inputs.append(
                    debate_template.format(question=prompt, responses_str=responses_str)
                )

        tasks = []
        input_prompts = []
        for agent, agent_input in zip(agents, agent_inputs):
            input_prompt = apply_template_with_tokenizer(
                agent["tokenizer"], agent_input
            )
            input_prompts.append(input_prompt)
            tasks.append(
                agent["llm"].generate_async.remote(
                    input_prompt, agent["sampling_params"]
                )
            )

        agent_results = await asyncio.gather(*tasks)
        agent_outputs = [result.outputs[0].text for result in agent_results]
        agent_rewards = auto_verify(
            task, 1, agent_outputs, [label] * len(agent_outputs)
        )

        for index, agent in enumerate(agents):
            trajectory[index].append(
                {
                    "turn_id": num_round,
                    "agent_index": index,
                    "agent_name": agent["agent_id"],
                    "agent_role": agent["agent_role"],
                    "agent_input": input_prompts[index],
                    "agent_output": agent_outputs[index],
                    "metadata": {},
                }
            )
            rewards[index].append(agent_rewards[index])

    final_reward = majority_vote(
        solutions=[agent_turns[-1]["agent_output"] for agent_turns in trajectory],
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
