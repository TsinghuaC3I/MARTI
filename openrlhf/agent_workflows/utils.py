from openrlhf.agent_workflows.tools.search import SearchToolExecutor
from openrlhf.agent_workflows.tools.sandbox import SandboxFusionExecutor

import os
import logging
import srsly
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))

def apply_template_with_tokenizer(tokenizer, prompt, tools=None):
    if tools is None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            tools=tools
        )

def assign_action_mask(turn):
    if "\n<|im_start|>user\n<tool_response>" in turn and "</tool_response><|im_end|>\n<|im_start|>assistant" in turn:
        return 0
    else:
        return 1

def register_mcp_tools(tool_manager):
    import asyncio
    asyncio.run(tool_manager.initialize())
    return tool_manager.openai_tools

def register_openai_tools(tools_config, tool_manager):
    """Register all configured tools."""
    tools = []
    # TODO: default configuration for deepcoder
    for tool_name, tool_cfg in tools_config.get("tools", {}).items():
        tool_type = tool_cfg.get("type")
        if tool_type == "search_r1":
            # Register search tool
            executor = SearchToolExecutor(
                base_url=tool_cfg["base_url"],
                topk=tool_cfg.get("topk", 3),
                timeout=tool_cfg.get("timeout", 15)
            )
            tool_manager.register_tool(tool_name, executor)

            schema = srsly.read_json(tool_cfg["schema_path"])
            tools.append(schema)
        elif tool_type == "sandbox_fusion":
            executor = SandboxFusionExecutor(
                base_url=tool_cfg["base_url"],
                timeout=tool_cfg.get("timeout", 30),
                language=tool_cfg.get("language", "python")
            )
            tool_manager.register_tool(tool_name, executor)

            schema = srsly.read_json(tool_cfg["schema_path"])
            tools.append(schema)
        else:
            logger.warning(
                f"Unknown tool type: {tool_type} for tool: {tool_name}")
    return tools

def print_tools(tools):
    logger.info(f"-------- Discovery {len(tools)} tools --------")
    for idx, tool in enumerate(tools):
        logger.info(f"Tool {idx}: {tool}")