import ray
import random
import numpy as np
from copy import deepcopy
from typing import List
from marti.verifiers.qwen.qwen_eval import qwen_reward_fn
from marti.verifiers.qwen.qwen_math_parser import extract_answer

from marti.worlds.third_party.base import BaseMultiAgentGame

id2template = {
    "0": {
        "generator": "Problem: {query}\n\nYou are the Solver. Please analyze the problem step by step, then provide your final answer in the form:\n\\boxed{{Your Final Answer Here}}",
        "verifier": """Please evaluate the following answer based on the problem. Provide a clear rationale and a final score on a scale of 1 to 10 (e.g., "Score: 8/10").

[Problem]
{prompt}

[Generated Answer]
{generated_answer}
"""
    }
}

class MultiAgentWorkflow(BaseMultiAgentGame):
    def __init__(
        self,
        agent_list,
        template_id,
        sampling_params,
        *args, **kwargs
    ):
        super().__init__(agent_list=agent_list, template_id=template_id, sampling_params=sampling_params)

        role2template = id2template[str(template_id)]

        self.generator_template = role2template["generator"]
        self.verifier_template = role2template["verifier"]
        self.histories = []

    def distribute_prompts(self, prompts: List[str], agent_id=0, turn_id=0) -> List[str]:
        llms = self.agent_list[agent_id]["llms"]
        tokenizer = self.agent_list[agent_id]["tokenizer"]
        is_reasoning_model = self.agent_list[agent_id]["is_reasoning_model"]
        
        chat_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            ) for prompt in prompts
        ]

        print(f"Prepare and generate with {agent_id}")
        all_output_refs = []
        batch_size = (len(chat_prompts) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            current_prompts = chat_prompts[i *
                                           batch_size: (i + 1) * batch_size]
            if is_reasoning_model:
                current_prompts = [prompt + "<think>" for prompt in current_prompts]

            if current_prompts:
                all_output_refs.append(llm.generate.remote(
                    current_prompts,
                    sampling_params=self.sampling_params,
                ))
        all_outputs = sum(ray.get(all_output_refs), [])
        all_texts = [output.outputs[0].text for output in all_outputs]
        if is_reasoning_model:
            all_texts = [
                "<think>" + text for text in all_texts
            ]

        for idx in random.sample(list(range(len(chat_prompts))), 2):
            print(f"Prompt ({agent_id}) >>>> " + repr(chat_prompts[idx]))
            print(f"Output ({agent_id}) >>>> " + repr(all_texts[idx]))

        self.organize_responses_by_problem_and_agents(
            num_problems=len(prompts),
            all_prompts=prompts,
            all_responses=all_texts,
            agent_id=agent_id,
            turn_id=turn_id
        )

        return all_texts

    def organize_responses_by_problem_and_agents(self,
                                                 num_problems,
                                                 all_prompts,
                                                 all_responses,
                                                 agent_id,
                                                 turn_id):
        assert num_problems == len(all_prompts) == len(all_responses)
        for prob_idx in range(num_problems):
            self.histories[prob_idx].append({
                "user": all_prompts[prob_idx],
                "assistant": all_responses[prob_idx],
                "agent_id": agent_id,
                "turn_id": turn_id
            })

    def process_prompt_for_thinking_model(self, solution, agent_id):
        if self.agent_list[agent_id]["is_reasoning_model"]:
            solution = solution.split("</think>")[-1].strip().strip("</answer>").strip("<answer>")
        return solution

    def run(self, problems: List[str], *args, **kwargs):
        labels = kwargs.get("labels", None)

        num_problems = len(problems)
        self.histories = [
            [] for _ in range(num_problems)
        ]

        # generate solutions
        generator_prompts = [
            self.generator_template.format(query=problem) for problem in problems
        ]

        solutions = self.distribute_prompts(
            generator_prompts, agent_id=0, turn_id=0)

        # verify solutions
        verifier_prompts = [
            self.verifier_template.format(
                query=problem,
                solution=self.process_prompt_for_thinking_model(solution, 0)
            ) for problem, solution in zip(
                problems, solutions
            )
        ]
        feedbacks = self.distribute_prompts(
            verifier_prompts, agent_id=1, turn_id=1)

    def get_history(self):
        return self.histories


class MultiAgentReward:
    def __init__(self,
                 verify="math",
                 name=None,
                 alpha=0.5, 
                 beta=0.5,
                 *args, **kwargs):
        print("Multi-agent Reward Allocation", locals())
        self.verify = verify
        self.name = name
        self.alpha = alpha
        self.beta = beta

    def run(self, histories, golden_answers, n_samples_per_prompt=None, answer_key="assistant"):
        pass