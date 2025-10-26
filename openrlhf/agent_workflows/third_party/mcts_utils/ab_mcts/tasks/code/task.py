import json
from pathlib import Path
from typing import List, Optional, Tuple

from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.data_types import Action, ARCProbData, Grid
from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.llm_generation_interface import GenerationResult
from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.eval_result import EvalResult, EvalResultWithAns
from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.tasks.base import Task

from openrlhf.verifiers.areal.code_reward import code_verify 
from openrlhf.utils.logging_utils import init_logger
logger = init_logger(__name__)

def postprocess_lcb_sample(sample, metadata=None):
    if sample is None:
        return None
    if isinstance(sample, str):
        sample = json.loads(sample)
    assert isinstance(sample, list), f"sample must be a list, but got {type(sample)}\n sample: {sample}"
    assert isinstance(sample[0], dict), f"sample must be a list of dicts, but got {type(sample[0])}, sample: {sample}"
    sample_inputs = [sample['input'] for sample in sample]
    sample_outputs = [sample['output'] for sample in sample]
    
    sample_dict = {
        'inputs': sample_inputs,
        'outputs': sample_outputs,
    }
    
    if sample[0].get("testtype") == "functional":
        # if metadata is None:
            # metadata = sample[0].get("metadata", {}) 
            # assert "metadata" in sample[0], f"metadata is required for functional testtype, but not found in sample: {sample}"
        assert metadata is not None, f"metadata is required for functional testtype"

        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        assert isinstance(metadata, dict), f"metadata must be a dict, but got {type(metadata)}\n metadata: {metadata}"
        fn_name = metadata.get("func_name", "")
        logger.info(f"Function name extracted from metadata: {fn_name}")
        assert fn_name != "", f"Function name is not found, check if your LCB data is preprocessed correctly: {metadata}"
        # Fill in the blank
        sample_dict['fn_name'] = fn_name
    
    return sample_dict

class CodeProblem(Task):
    def __init__(self, problem, label: Optional[str] = None) -> None:
        '''
        new_item = {
            "data_source": item["data_source"]"livecodebench"
            "prompt": item["prompt"],[{role:user, content:}]
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": item["label"]#[{input:xxx,outpu:xxx,},{input:xxx,outpu:xxx,}]
            },
            "extra_info": {
                "split": "train",
                "index": item["indice"],
                "reference": None,
                "difficulty": item["difficulty"]
            }
        }
        '''
        self.debug = False
        ground_truth = problem["label"]
        if isinstance(ground_truth, str):
            ground_truth = json.loads(ground_truth)

        assert isinstance(ground_truth, dict), f"ground truth must be dict, ground truth is with type: {type(ground_truth)}"
        if "inputs" in ground_truth:
            public_tests = ground_truth
            private_tests = None
        else:
            # logger.info(f"public&private tests have been loaded from {problem['indice']}")
            public_tests = ground_truth["public_tests"]# public_tests
            private_tests = ground_truth["private_tests"]# private_tests

        self.demos = public_tests # **FOR TRAINING**
        self.tests = private_tests
        self.prompt = problem["prompt"] 
        self.metadata = problem["metadata"]
        # change format of demos/test to [inputs_outputs]
        # self.demos = postprocess_lcb_sample(self.demos, self.metadata)
        # self.tests = postprocess_lcb_sample(self.tests, self.metadata)
        # logger.info(f"demos is {self.demos}")
        # logger.info(f"tests is {self.tests}")

    @classmethod
    def load_file(cls, json_path: Path | str) -> "CodeProblem":
        prob_path = Path(json_path)
        if not prob_path.exists():
            raise RuntimeError(f"Code problem not found at {str(prob_path)}")
        data = prob_path.read_text()
        data = json.loads(data)
        # print(f"the type of data is {type(data)}")
        data["label"] = data["reward_model"]
        label=prob_path.stem
        label = label.split("_")[1]
        data["indice"] = label
        return cls(problem=data, label=label)
    
    @classmethod
    def load_data(cls, prompt, ground_truth, meta_data) -> "CodeProblem":
        problem = {
            'prompt': prompt,
            'label': ground_truth,
            'metadata': meta_data,
        }
        return cls(problem=problem)

    def generate_eval_results(
        self, llm_answer: GenerationResult, kind: Action
    ) -> Optional[List[EvalResult]]:
        # llm_answer_code = llm_answer.parse_python_code()
        # if llm_answer_code is None:
        #     return None

        if kind == "transform":
            results = self.run_transform_on_demos(llm_answer.action_text)
        else:
            raise NotImplementedError()

        return results

    def evaluate_on_test(
        self, llm_answer: GenerationResult
    ) -> Tuple[List[EvalResult], float]:

        eval_results = self.run_transform_on_tests_and_check(llm_answer.action_text)
        score = (
            1.0
            if all(eval_result.get_score() == 1.0 for eval_result in eval_results)
            else 0.0
        )
        return eval_results, score

    def run_transform_on_demos(self, transform_code: str) -> List[EvalResult]:
        '''
        transform_code is model output
        '''
        eval_results: List[EvalResultWithAns] = []
        if self.demos is None:
            return None

        res, output = code_verify(generateds=transform_code, problems=self.demos, debug=self.debug, return_metadata=True)
        eval_results.append(
            EvalResultWithAns(answer=0 if any(x != True for x in res) else 1, groundtruth=self.demos["outputs"] )
        )
        return eval_results

    def run_transform_on_tests_and_check(self, transform_code: str) -> List[EvalResult]:
        '''
        transform_code is model output
        '''
        eval_results: List[EvalResultWithAns] = []
        if self.tests is None:
            return None

        res, output = code_verify(generateds=transform_code, problems=self.tests, debug=self.debug, return_metadata=True)
        
        # Check if the transform code passes all tests
        eval_results.append(
            EvalResultWithAns(answer=0 if any(x != True for x in res) else 1, groundtruth=self.tests["outputs"] )
        )
        return eval_results

    def run_transform_on_tests(self, transform_code: str) -> List[EvalResult]:
        """
        Does not check if it is correct on test
        """
        eval_results: List[EvalResultWithAns] = []
        if self.tests is None:
            return None

        res, output = code_verify(generateds=transform_code, problems=self.tests, debug=self.debug, return_metadata=True)
        eval_results.append(
            EvalResultWithAns(answer=0 if any(x != True for x in res) else 1, groundtruth=self.tests["outputs"] )
        )
        return eval_results

    def check_on_tests(self, preds: List[Grid]) -> bool:
        """True if correct on all tests"""
        for pred, test in zip(preds, self.tests):
            if pred != test["output"]:
                return False
        return True
