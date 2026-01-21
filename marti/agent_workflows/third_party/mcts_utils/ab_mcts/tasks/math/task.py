import json
from pathlib import Path
from typing import List, Optional, Tuple

from marti.agent_workflows.third_party.mcts_utils.ab_mcts.data_types import Action, Grid
from marti.agent_workflows.third_party.mcts_utils.ab_mcts.llm_generation_interface import GenerationResult
from marti.agent_workflows.third_party.mcts_utils.ab_mcts.eval_result import EvalResult, EvalResultWithAns
from marti.verifiers.qwen.qwen_eval import qwen_reward_fn, qwen_reward_fn_format
from marti.agent_workflows.third_party.mcts_utils.ab_mcts.tasks.base import Task


class MathProblem(Task):
    # TODO add majority label
    def __init__(self, problem, label: Optional[str] = None) -> None:
        # self.demos = problem["train"]
        # self.tests = problem["test"]
        self.demos = None
        ground_truth = problem["label"]

        # if isinstance(ground_truth, str):
            # ground_truth = json.loads(ground_truth)
        # ground_truth = json.loads(ground_truth["ground_truth"])
        if not isinstance(ground_truth, list):
            if isinstance(ground_truth, dict):
                ground_truth = ground_truth["ground_truth"]
            ground_truth = [ground_truth]
        # print(f" ground")
        assert isinstance(ground_truth, list), f"ground truth must be list，ground truth is：{ground_truth} with type：{type(ground_truth)}"
        self.tests = [input_output for input_output in ground_truth]
        if self.demos is None:
            self.demos = self.tests
        self.prompt = problem["prompt"]
        # self.label = problem["extra_info"]["index"]   
        # self.label = problem['indice'] 
        self.label = problem['indice'] if label is None else label

        # self.label = label

    @classmethod
    def load_file(cls, json_path: Path | str) -> "MathProblem":
        prob_path = Path(json_path)
        if not prob_path.exists():
            raise RuntimeError(f"Code problem not found at {str(prob_path)}")
        data = prob_path.read_text()
        # print(f"the type of data is {type(data)}")
        # print(data)
        data = json.loads(data)
        print(f"the type of data is {type(data)}")
        print(data)
        # data["label"] = data["answer"]
        label = prob_path.stem
        label = label.split("-")[1]
        # print(f"the label is {label}")
        # data["indice"] = label

        problem = {
            'prompt': data["problem"],
            'label': {
                'ground_truth': data["answer"],
            },
            'meta_data': {},
            'indice': label,
        }


        return cls(problem=problem, label=label)
        # prob_path = Path(json_path)
        # if not prob_path.exists():
        #     raise RuntimeError(f"math problem not found at {str(prob_path)}")

        # return cls(problem=json.loads(prob_path.read_text()), label=prob_path.stem)
    @classmethod
    def load_data(cls, prompt, ground_truth, meta_data, indice) -> "MathProblem":
        problem = {
            'prompt': prompt,
            'label': {
                'ground_truth': ground_truth,
            },
            'meta_data': meta_data,
            'indice': indice,
        }
        return cls(problem=problem, label=indice)

    def generate_eval_results(
        self, llm_answer: GenerationResult, kind: Action
    ) -> Optional[List[EvalResult]]:
        # llm_answer_code = llm_answer.parse_python_code()
        # if llm_answer_code is None:
        #     return None
        generation = llm_answer.generation
        if kind == "transform":
            results = self.run_transform_on_demos(generation)
        else:
            raise NotImplementedError()

        return results

    def evaluate_on_test(
        self, llm_answer: GenerationResult
    ) -> Tuple[List[EvalResult], float]:
        # py_code = llm_answer.parse_python_code()
        # if py_code is None:
        #     return [], 0.0
        generation = llm_answer.generation
        eval_results = self.run_transform_on_tests_and_check(generation)
        score = (
            1.0
            if all(eval_result.get_score() == 1.0 for eval_result in eval_results)
            else 0.0
        )
        return eval_results, score

    def run_transform_on_demos(self, transform_code: str) -> List[EvalResult]:
        eval_results: List[EvalResultWithAns] = []
        if self.demos is None:
            return None
        for demo in self.demos:
            is_correct = qwen_reward_fn(transform_code, demo, task='math')
            eval_results.append(
                EvalResultWithAns(answer=is_correct, groundtruth=demo)
            )
        return eval_results

    def run_transform_on_tests_and_check(self, transform_code: str) -> List[EvalResult]:
        eval_results: List[EvalResultWithAns] = []
        for test in self.tests:
            is_correct = qwen_reward_fn(transform_code, test, task='math')
            eval_results.append(
                EvalResultWithAns(answer=is_correct, groundtruth=test)
            )
        return eval_results

    def run_transform_on_tests(self, transform_code: str) -> List[EvalResult]:
        """
        Does not check if it is correct on test
        """
        eval_results: List[EvalResultWithAns] = []
        for test in self.tests:
            is_correct = qwen_reward_fn(transform_code, test, task='math')
            eval_results.append(EvalResultWithAns(answer=is_correct, groundtruth=None))
        return eval_results

    def check_on_tests(self, preds: List[Grid]) -> bool:
        """True if correct on all tests"""
        for pred, test in zip(preds, self.tests):
            if pred != test:
                return False
        return True
