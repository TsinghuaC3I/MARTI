"""
Example workflow: MathChat-style multi-agent interaction
Pattern: Generator -> Coder -> Refiner -> Coder -> Refiner -> ...
"""
import os
from typing import Dict, List, Any, Optional
import json
import asyncio
from marti.utils.logging_utils import init_logger
from marti.verifiers.auto_verify import auto_verify
from marti.agent_workflows.utils import apply_template_with_tokenizer

logger = init_logger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))


async def workflow(
    prompt: str,
    label: str,
    agents: List[Dict[str, Any]],
    tool_manager,
    task: str,
    metadata: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    MathChat workflow: Generator -> Coder -> Refiner cycle

    Args:
        prompt: Initial problem prompt
        label: Expected answer/label
        agents: List of agent configurations
        tool_manager: Tool manager instance
        task: Task identifier
        metadata: Additional metadata
    """
    trajectory = [
        {
            "turn_id": 0,
            "agent_index": 0,
            "agent_name":  "agent0",
            "agent_role": "generator",
            "agent_input": "input example",
            "agent_output": "output example",
            "metadata": {}
        },
        # Add more turns
    ]
    rewards = [
        0
        # Add reward for each turn if exist
    ]

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "final_reward": rewards[-1]
    }
