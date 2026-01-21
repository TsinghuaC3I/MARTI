import os
import ray
import random
import numpy as np
from typing import List
import concurrent.futures
from marti.verifiers.qwen.qwen_eval import qwen_reward_fn
from marti.verifiers.auto_verify import get_repetition_penalty_reward
from marti.worlds.third_party.base import BaseMultiAgentGame

cand_template = "{query}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

agg_template_default = """### Context
You are one of several collaborative agents working together to solve complex mathematical problems. Your role is NOT predefined. Instead, you should carefully read the discussion context, then decide and clearly state your next action. Possible actions include but are not limited to:
- Propose a new detailed solution or reasoning steps.
- Critically analyze, find errors or limitations in previous answers.
- Reflect or summarize previous arguments, then propose improvements or corrections.
- Combine multiple previous solutions and critiques into a refined solution.

### Problem:
{query}

### Discussion History:
{history}


### Your Response:
First explicitly state your chosen action in brackets, e.g. [Critique], [Solution], [Reflection], or [Combination].  
Then, clearly provide your detailed response step-by-step. Please put your final answer within \\boxed{{}}."""

agg_template_decision_prefix = """You are the top decision-maker agent specializing in mathematical reasoning. Your primary strengths include detailed analysis of mathematical problems, careful evaluation of solutions from other agents, and precise summarization of reasoning steps. You will receive:

1. A mathematical problem.
2. Analyses, reasoning steps, and computational results from other agents.

Your task:
- Identify and carefully evaluate the analyses and solutions provided by other agents.
- Determine the most accurate and reliable solution.
- Clearly explain your reasoning step-by-step, referencing specific details from other agents' analyses.
- Present your final answer clearly and succinctly.

### Problem:
{query}

### Responses from Other Agents:
{history}

Please reason step by step, and put your final answer within \\boxed{{}}.
"""

agg_template_decision_suffix = """### Problem:
{query}

### Responses from Other Agents:
{history}

You are the top decision-maker agent specializing in mathematical reasoning. Your primary strengths include detailed analysis of mathematical problems, careful evaluation of solutions from other agents, and precise summarization of reasoning steps. You will receive:

1. A mathematical problem.
2. Analyses, reasoning steps, and computational results from other agents.

Your task:
- Identify and carefully evaluate the analyses and solutions provided by other agents.
- Determine the most accurate and reliable solution.
- Clearly explain your reasoning step-by-step, referencing specific details from other agents' analyses.
- Present your final answer clearly and succinctly.

Please reason step by step, and put your final answer within \\boxed{{}}.
"""

template_dict = {
    "0": agg_template_default,
    "1": agg_template_decision_prefix,
    "2": agg_template_decision_suffix
}

class MultiAgentMixture(BaseMultiAgentGame):
    def __init__(
        self,
        agent_list,
        template_id,
        sampling_params,
        *args, **kwargs
    ):
        super().__init__(agent_list=agent_list, template_id=template_id, sampling_params=sampling_params)

        self.all_llms = self.agent_list[:-1]
        self.agg_llms = self.agent_list[-1]

        self.cand_template = cand_template
        self.agg_template = template_dict[str(template_id)]
        self.shuffle_responses = kwargs.get("shuffle_responses", True)
        self.histories = []

    def distribute_prompts(self, prompts: List[str], llm_dict, agent_id, turn_id) -> List[str]:
        tokenizer = llm_dict["tokenizer"]
        llms = llm_dict["llms"]
        thinking_mode = llm_dict["is_reasoning_model"]

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
            if thinking_mode:
                current_prompts = [prompt + "<think>" for prompt in current_prompts]

            if current_prompts:
                all_output_refs.append(llm.generate.remote(
                    current_prompts,
                    sampling_params=self.sampling_params,
                ))
        all_outputs = sum(ray.get(all_output_refs), [])
        all_texts = [output.outputs[0].text for output in all_outputs]
        if thinking_mode:
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
            self.histories[prob_idx][agent_id] = {
                "user": all_prompts[prob_idx],
                "assistant": all_responses[prob_idx],
                "agent_id": agent_id,
                "turn_id": turn_id
            }

    def process_prompt_for_thinking_mode(self, solution, agent_id):
        if self.agent_list[agent_id]["is_reasoning_model"]:
            solution = solution.split("</think>")[-1].strip().strip("</answer>").strip("<answer>")
        return solution

    def run(self, problems: List[str], *args, **kwargs):
        num_problems = len(problems)
        self.histories = [
            [None for _ in range(self.num_agents)] for _ in range(num_problems)
        ]

        # generate solutions concurrently
        agent2solutions = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.all_llms)) as executor:
            formated_problems = [self.cand_template.format(query=problem) for problem in problems]
            # submit each distribute_prompts call to the executor
            future_to_agent = {
                executor.submit(self.distribute_prompts, formated_problems, agent, agent_id, 0): agent_id 
                for agent_id, agent in enumerate(self.all_llms)
            }
            # process completed futures as they finish
            for future in concurrent.futures.as_completed(future_to_agent):
                agent_id = future_to_agent[future]
                try:
                    agent2solutions[agent_id] = future.result()
                except Exception as exc:
                    print(f"Agent {agent_id} generated an exception: {exc}")
                    agent2solutions[agent_id] = None

        agg_prompts = []
        for pid in range(num_problems):
            agent_history = []
            for agent, solutions in agent2solutions.items():
                agent_history.append(
                    self.process_prompt_for_thinking_mode(solutions[pid], agent_id)
                )
            if self.shuffle_responses:
                random.shuffle(agent_history)
            
            agent_history = [f"Agent {i+1}:\n{res}" for i, res in enumerate(agent_history)]

            agg_prompts.append(
                self.agg_template.format(
                    query=problems[pid],
                    history="\n\n".join(agent_history)
                )
            )

        final_solutions = self.distribute_prompts(
            agg_prompts, self.agg_llms, agent_id=self.num_agents - 1, turn_id=1)

        # sorted the candidates by agent_id
        # [[agent 1, agent 2, agent 3], [agent agg]]
        # for prob_idx in range(num_problems):
        #     prob_history = self.histories[prob_idx]
        #     assert prob_history[-1]["id"] == self.num_agents - 1
        #     prob_history = sorted(prob_history[:-1], key= lambda item: item["id"]) + prob_history[-1:]
        #     self.histories[prob_idx] = [prob_history[:-1], prob_history[-1:]]

    def get_history(self):
        return self.histories
