import json
from typing import Dict, Any, List, Optional

from marti.worlds.tools.manager import ToolManager
from openrlhf.utils.logging_utils import init_logger
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))

async def step_with_tools(
    observation: List[str],
    action: str,
    tool_manager: Optional[ToolManager] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generic step function that works with any tools.
    Tool parser should return: [{"name": "tool_name", "args": "{...}"}, ...]
    """
    next_observation = observation + [action]
    done = False
    extra_logs = {"tools_used": {}}
    
    parser_results = tool_manager.tool_parser.parse_tools(action)

    if parser_results[0] == action:
        # No tool calls detected
        done = True
    else:
        tool_responses = []
        
        for parser_result in parser_results:
            tool_name = parser_result.get("name")
            
            if tool_name in ['<error>', '<empty>', '<parse_error>']:
                # Handle parsing errors
                tool_response = json.dumps(parser_results)
            elif tool_manager and tool_name in tool_manager.available_tools:
                # Execute registered tool
                try:
                    # logger.error(f"{parser_result}")
                    # Parse tool arguments
                    args_dict = json.loads(parser_result.get("args", "{}"))

                    if isinstance(args_dict, str):
                        args_dict = json.loads(args_dict)

                    # Track tool usage
                    if tool_name not in extra_logs["tools_used"]:
                        extra_logs["tools_used"][tool_name] = 0
                    extra_logs["tools_used"][tool_name] += 1

                    # Execute tool through manager
                    response, metadata = await tool_manager.execute_tool(
                        tool_name, args_dict, **kwargs
                    )
                    tool_response = response
                except Exception as e:
                    tool_response = f"Error executing {tool_name}: {str(e)}"
                    logger.error(f"Tool execution error: {e} - {type(parser_result.get('args'))} - {parser_result.get('args')}")
            else:
                # Unknown tool
                tool_response = f"Tool '{tool_name}' is not supported"
                
            tool_responses.append(tool_response)
        
        # Format tool responses
        tool_context = '\n------\n'.join(tool_responses)
        tool_context = f"\n<|im_start|>user\n<tool_response>\n{tool_context}\n</tool_response><|im_end|>\n<|im_start|>assistant"
        
        next_observation += [tool_context]

    return {
        "next_observation": next_observation,
        "done": done,
        "extra_logs": extra_logs
    }
