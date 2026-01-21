from collections import defaultdict
from typing import List, Optional

from marti.agent_workflows.third_party.mcts_utils.ab_mcts.data_types import Action
from marti.agent_workflows.third_party.mcts_utils.ab_mcts.llm_generation_interface import GenerationRequest, GenerationResult
from marti.agent_workflows.third_party.mcts_utils.ab_mcts.eval_result import EvalResultWithAns
from marti.agent_workflows.third_party.mcts_utils.ab_mcts.prompts.prompt_configs import PromptConfig
from marti.agent_workflows.third_party.mcts_utils.ab_mcts.prompts.base import PromptTemplate
from marti.agent_workflows.third_party.mcts_utils.ab_mcts.tasks.math.task import MathProblem
from marti.agent_workflows.third_party.mcts_utils.ab_mcts.tasks.base import Task

from marti.verifiers.qwen.qwen_math_parser import extract_answer

from marti.utils.logging_utils import init_logger

logger = init_logger(__name__)


class MathPrompt(PromptTemplate):
    version = "baseline"

    def __init__(self, prompt_config: PromptConfig, problem: MathProblem):
        self.problem = problem

    def initial_prompt(self) -> str:
        prompt = problem_prompt(self.problem)
        return prompt

    def feedback_prompt(
        self,
        action: Action,
        eval_results: Optional[List[EvalResultWithAns]],
        generation_result: GenerationResult,
    ) -> str:
        generation_result=generation_result.generation
        code = extract_answer(generation_result, data_name="math")
        # try:
        #     generation_result=generation_result.generation
        #     code = extract_answer(generation_result)
        # except:
        #     code = ""
        match action:
            case "transform":
                prompt = transform_feedback_prompt(
                    problem=self.problem,
                    eval_results=eval_results,
                    pycode=code,
                )
                return prompt
            case _:
                raise NotImplementedError(
                    f"feedback_prompt not implemented for action {action}"
                )

    def add_next_action_instruction(
        self, action: Action, next_prompt: GenerationRequest
    ) -> GenerationRequest:
        last_user_msg = next_prompt.messages[-1]
        assert last_user_msg.role == "user"
        # Only use the last user message
        next_prompt.messages = next_prompt.messages[-1:]

        return next_prompt


def problem_prompt(problem: MathProblem) -> str:
    prompt = problem.prompt
    add_prompt_suffix = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    prompt += add_prompt_suffix
    return prompt


def transform_feedback_prompt(
    problem: MathProblem, eval_results: List[EvalResultWithAns], pycode: Optional[str]
) -> str:
    # Since there’s no task information without the initial prompt code, it is required for single-turn scenarios.
    prompt = problem_prompt(problem)
    # if pycode == "" or eval_results is None:
    #     prompt += f"### Answer: Your answer doesn't include any valid content."
    #     prompt += "\n\n"
    #     # prompt += "# Again, here we show the input and output grids for the problem."
    #     # prompt += problem_prompt(problem)
    #     return prompt

    prompt += f"\nYour previous answer:\n\n\\boxed{pycode}\n\n"
    prompt += "Here are the feedbacks based on the answer above.\n"

    if pycode is "":
        prompt += (
            "Your answer can't be extracted from your response, so please fix it accordingly.\n\n"
        )
    num_correct = 0
    for i, eval_result in enumerate(eval_results):
        output = eval_result.answer
        is_correct = eval_result.get_score() == 1.0
        prompt += f"# Example {i}\n\n"
        if is_correct is True:
            prompt += "Result: The answer is Correct\n\n"
            num_correct += 1
        else:
            prompt += f"""
            Result: You get a Wrong answer.
            """
    return prompt


def next_task_prompt(kind: Action, is_first_turn: bool) -> str:
    first_line = (
        "Given the above result, reflect what was correct and/or wrong with your understanding and correct it accordingly inside <reflection></reflection> block, and w"
        if not is_first_turn
        else "W"
    )

    if kind == "transform":
        return (
            f"{first_line}"
            + "rite your reasoning and details, and then write a new transform Python function which takes input grid as an argument inside code block surrounded by ```python and ```.\n"
            "Also, be careful to find pattern from example input and output and try to generalize it to additional inputs. "
            "DO NOT hardcode output into your `transform` function and return it for each example. Please remember that your task is to identify general transform pattern from examples.\n"
        )
    else:
        raise NotImplementedError()
