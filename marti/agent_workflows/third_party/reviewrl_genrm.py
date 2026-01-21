import ray
import random
import numpy as np
from copy import deepcopy
from typing import List
from marti.verifiers.qwen.qwen_eval import qwen_reward_fn
from marti.verifiers.qwen.qwen_math_parser import extract_answer
from marti.verifiers.auto_verify import auto_verify

from marti.worlds.third_party.base import BaseMultiAgentGame

def _parse_score_keywords(text: str) -> float:
    """
    Keyword-based score parser for binary judgments.
    Example: "REVIEW_1_BETTER" -> 1.0, "REVIEW_2_BETTER" -> 0.0
    """
    response_text = text.upper().strip()
    if "REVIEW_1_BETTER" in response_text or "FIRST_BETTER" in response_text or "OPTION_1" in response_text:
        return 1.0
    elif "REVIEW_2_BETTER" in response_text or "SECOND_BETTER" in response_text or "OPTION_2" in response_text:
        return 0.0
    else:
        print(f"Judge output did not match expected keywords: '{response_text}'. Assigning default score 0.5.")
        return 0.5

id2template = {
    "0": {
        "generator": "{query}",
        "verifier": """
You are an expert academic peer reviewer. You will be shown the abstract/content of a research paper and two peer reviews for that paper. Your task is to determine which peer review is of higher quality based on the following criteria:

1. **Factual Accuracy & Soundness:** Does the review accurately understand the paper's contributions and limitations? Is the critique based on sound reasoning?
2. **Completeness & Coverage:** Does the review address the core aspects of the paper (e.g., methodology, results, significance)?
3. **Level of Detail & Specificity:** Does the review provide specific examples and detailed comments rather than vague statements?
4. **Comparison with Existing Work:** Does the review appropriately contextualize the paper within the existing literature and compare it to relevant methods?
5. **Constructiveness:** Is the feedback helpful for the authors to improve the paper? Is the tone professional and constructive?
6. **Clarity & Organization:** Is the review well-structured and easy to understand?

[Paper Context (Abstract/Content)]
{prompt}

[Review 1]
{generated_answer}

[Review 2]
{label}

Which peer review is of higher quality based on the criteria above? Respond with EXACTLY one of these options:
- REVIEW_1_BETTER
- REVIEW_2_BETTER

YOU MUST CHOOSE A BETTER REVIEW. A TIE IS NOT ALLOWED.

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

        if labels:
            ground_truth_content = [label.split("||DIV REVIEW SCORE||")[0] for label in labels]
        else:
            ground_truth_content = [""] * len(problems)

        # verify solutions
        verifier_prompts = [
            self.verifier_template.format(
                prompt=problem,
                generated_answer=self.process_prompt_for_thinking_model(solution, 0),
                label=label if labels else ""
            ) for problem, solution, label in zip(
                problems, solutions, ground_truth_content if labels else [""] * len(problems)
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
                 judge_weight=1.0,
                 label_separator=None,
                 task="review",
                 *args, **kwargs):
        print("Multi-agent Reward Allocation", locals())
        self.verify = verify
        self.name = name
        self.alpha = alpha
        self.beta = beta
        self.judge_weight = judge_weight
        self.label_separator = label_separator
        self.task = task

    def run(self, histories, golden_answers, n_samples_per_prompt=None, answer_key="assistant"):
        """
        Process histories and compute rewards following judge_workflow.py pattern.
        Returns (local_rewards, outcome_rewards) tuple like auto_reward_alloc.py
        """
        local_rewards = []
        outcome_rewards = []
        
        for history, label in zip(histories, golden_answers):
            # Extract generator and judge responses from history
            generator_response = None
            judge_response = None
            
            for entry in history:
                if entry["agent_id"] == 0:  # generator
                    generator_response = entry[answer_key]
                elif entry["agent_id"] == 1:  # judge/verifier
                    judge_response = entry[answer_key]
            
            if generator_response is None or judge_response is None:
                print("Warning: Missing generator or judge response in history")
                local_rewards.append([0.0, 0.0])  # [generator_reward, judge_reward]
                outcome_rewards.append(0.0)  # outcome reward
                continue
            
            # Parse label if structured (same as judge_workflow.py)
            # if self.label_separator and self.label_separator in label:
            #     label_parts = label.split(self.label_separator)
            #     ground_truth_content = label_parts[0]
            #     ground_truth_score = label_parts[1] if len(label_parts) > 1 else label
            # else:
            #     ground_truth_content = label
            #     ground_truth_score = label
            
            ground_truth_content, ground_truth_score = label.split("||DIV REVIEW SCORE||")

            # Process generator response for thinking models (same as original)
            processed_generator_response = generator_response
            if "</think>" in generator_response:
                processed_generator_response = generator_response.split("</think>")[-1].strip().strip("</answer>").strip("<answer>")
            
            # Extract score using keyword parser (same as judge_workflow.py)
            judge_score = _parse_score_keywords(judge_response)
            
            # Get rule-based reward if weight < 1.0 (same as judge_workflow.py)
            if self.judge_weight < 1.0:
                rule_reward = auto_verify(self.task, 1, [processed_generator_response], [ground_truth_score])[0]
            else:
                rule_reward = 0.0
            
            # Combine rewards (same as judge_workflow.py)
            generator_judge_reward = judge_score
            generator_combined_reward = generator_judge_reward * self.judge_weight + rule_reward * (1 - self.judge_weight)
            judge_combined_reward = 0.0  # Judge should not be updated
            
            # Clip rewards to [0, 1] (same as judge_workflow.py)
            generator_combined_reward = max(0.0, min(1.0, generator_combined_reward))
            judge_combined_reward = 0.0
            
            # Local rewards for this problem: [generator_reward, judge_reward]
            local_rewards.append([generator_combined_reward, judge_combined_reward])
            # Outcome reward is the generator's final reward
            outcome_rewards.append(generator_combined_reward)
        
        return local_rewards, outcome_rewards