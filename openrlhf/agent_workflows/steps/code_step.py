import json
from typing import Dict, Any, List, Optional

# from marti.worlds.tools.manager import ToolManager
from openrlhf.agent_workflows.tools.manager import ToolManager
from openrlhf.utils.logging_utils import init_logger# from openrlhf.verifiers.deepcoder.code_reward import rllm_reward_fn_code
from openrlhf.verifiers.areal.code_reward import code_verify

import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))


async def step(
    observation: List[str],
    action: str,
    tool_manager: Optional[ToolManager] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generic step function that works with any tools.
    Tool parser should return: [{"name": "tool_name", "args": "{...}"}, ...]
    """
    # metadata = json.loads(kwargs["metadata"])
    # final_reward = rllm_reward_fn_code(
    #     metadata["data_source"], action, kwargs["label"])
    final_reward = code_verify(kwargs["label"], action)

    return {
        "next_observation": observation + [action],
        "done": True,
        "extra_logs": {"tools_used": {}},
        "final_reward": final_reward
    }
