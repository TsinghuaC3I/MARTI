from abc import ABC
from typing import List, Optional, Tuple

from marti.agent_workflows.third_party.mcts_utils.ab_mcts.data_types import Action
from marti.agent_workflows.third_party.mcts_utils.ab_mcts.llm_generation_interface import GenerationResult
from marti.agent_workflows.third_party.mcts_utils.ab_mcts.eval_result import EvalResult


class Task(ABC):
    def generate_eval_results(
        self, llm_answer: GenerationResult, kind: Action
    ) -> Optional[List[EvalResult]]:
        raise NotImplementedError()

    def evaluate_on_test(
        self, llm_answer: GenerationResult
    ) -> Tuple[List[EvalResult], float]:
        raise NotImplementedError()
