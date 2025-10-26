import re
from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional

from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.model_base import GenerationRequest as BaseGenerationRequest
from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.model_base import GenerationResult as BaseGenerationResult
from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.model_base import Message as BaseMessage
from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.model_base import Model as BaseModel
from PIL import Image


class GenerationResult(BaseGenerationResult):
    def parse_python_code(self) -> Optional[str]:
        code = re.search(
            r"```python\n(.*)\n```", self.action_text, re.DOTALL | re.MULTILINE
        )
        if code is None:
            return None
        return code.group(1).replace("\t", " " * 4)

    def parse_score_block(self) -> Optional[str]:
        score = re.search(r"```score(.*)```", self.action_text, re.DOTALL | re.MULTILINE)
        if score is None:
            return None
        return score.group(1).strip()


class GenerationRequest(BaseGenerationRequest):
    def get_last_generation_result(self) -> GenerationResult:
        assert self.messages[-1].role == "user" and len(self.messages) >= 3
        assert self.messages[-2].role == "assistant"

        return GenerationResult(
            request=GenerationRequest(messages=self.messages[:-2]),
            generation=self.messages[-2].content,
        )


class Model(BaseModel):

    @abstractmethod
    def generate(
        self, requests: Image.Sequence[GenerationRequest]
    ) -> Iterable[GenerationResult]:
        raise NotImplementedError()


@dataclass
class Message(BaseMessage):
    content: str | List[str | Image.Image]
