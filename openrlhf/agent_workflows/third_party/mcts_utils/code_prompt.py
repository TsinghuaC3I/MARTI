from collections import defaultdict
from typing import List, Optional

from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.data_types import Action
from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.llm_generation_interface import GenerationRequest, GenerationResult
from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.eval_result import EvalResultWithAns
from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.prompts.prompt_configs import PromptConfig
from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.prompts.base import PromptTemplate
from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.tasks.code.task import CodeProblem
from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.tasks.base import Task

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


class CodePrompt(PromptTemplate):
    version = "baseline"

    def __init__(self, prompt_config: PromptConfig, problem: CodeProblem):
        self.problem = problem

    def initial_prompt(self) -> str:
        prompt = ""
        prompt += problem_prompt(self.problem)
        return prompt

    def feedback_prompt(
        self,
        action: Action,
        eval_results: Optional[List[EvalResultWithAns]],
        generation_result: GenerationResult,
    ) -> str:
        try:
            code = generation_result.parse_python_code()
        except:
            code = ""
        # if code == "":
        #     logger.info(f"extract python code from generation failed,\n")
        match action:
            case "transform":
                return transform_feedback_prompt(
                    problem=self.problem,
                    eval_results=eval_results,
                    pycode=code,
                )
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

def problem_prompt(problem: CodeProblem) -> str:
    prompt = problem.prompt
    return prompt

def transform_feedback_prompt(
    problem: CodeProblem, eval_results: List[EvalResultWithAns], pycode: Optional[str]
) -> str:
    # Since there’s no task information without the initial prompt code, it is required for single-turn scenarios.
    prompt = ""
    if pycode == "" or eval_results is None:
        prompt += f"### Answer: Your answer doesn't include any code."
        prompt += "\n\n"
        prompt += "# Again, here we show the input and output grids for the problem."
        prompt += "\n\n"
        prompt += problem_prompt(problem)
        return prompt

    prompt += f"\nYour previous code:\n```\n{pycode}\n```\n\n"
    prompt += "Here are the results based on the code above.\n"

    num_correct = 0
    for i, eval_result in enumerate(eval_results):
        output = eval_result.answer
        is_correct = eval_result.get_score() == 1.0
        prompt += f"# Example {i}\n\n"
        if is_correct is True:
            prompt += "\n\nResult: Correct\n\n"
            num_correct += 1
        else:
            prompt += f"\n\nResult: Wrong"


    if num_correct == len(eval_results):
        prompt += "# Summary\n\nYour solution is correct for all the problems!\n\n"
    else:
        prompt += f"# Summary\n\nYour solution is correct for {num_correct} problems among {len(eval_results)}!\n\n"

    # We also show transform function's result on additional inputs
    if pycode is None:
        prompt += (
            "Your `transform` function was malformed, so please fix it accordingly.\n\n"
        )
#     else:
#         prompt += "Also, here are the outputs of your `transform` function on additional inputs. Please check if your `transform` worked on additional inputs as intended, and correct your mistake in your next turns.\n\n"
#         # outputs = problem.run_transform_on_tests(pycode)
#         outputs = problem.run_transform_on_demos(pycode)
#         for i, eval_result in enumerate(outputs):
#             output = eval_result.answer
#             prompt += f"# Transformed output on Additional Input {i}\n\n"
#             if output is None:
#                 prompt += (
#                     f"Your `transform` function is invalid for Additional Input {i}\n\n"
#                 )
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
