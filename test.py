import asyncio

from openrlhf.agent_workflows.utils import register_mcp_tools, register_openai_tools, print_tools, assign_action_mask
from openrlhf.agent_workflows.tools.manager import ToolManager
from openrlhf.agent_workflows.tools.mcp_manager import MCPManager

import socket

def get_my_ip():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

my_ip = get_my_ip()
print("My IP:", my_ip)

tools_config={
    "max_turns": 2,
    "num_workers": 4,
    "enable_metrics": True,
    "enable_rate_limiting": True,
    "tools": {
        "code_interpreter": {
            "type": "sandbox_fusion",
            "enable_rate_limiting": True,
            "rate_limit": 256,
            "timeout": 30,
            # "base_url": "http://101.6.64.188:10086/run_code",
            "base_url": f"http://{my_ip}:10086/run_code",
            "schema_path": "examples/schema/code.json"
        }
    }
}


def _init_tool_manager():
    print(f"Tool config is: {tools_config}")
    # self.args.get("tools_config", {})

    # assert self.packing_samples, "Only support packing samples"

    if tools_config.get("mcp_url", None) is not None:
        tool_manager = MCPManager(tools_config)
        tools = register_mcp_tools(tool_manager)
    else:
        tool_manager = ToolManager(tools_config)
        tools = register_openai_tools(tools_config, tool_manager)

    print(f"tools is: {tools}")
    print(f"tool maneger 初始化：{tool_manager}")
    
    tool_manager.set_tools(tools)
    print(f"有tools 的tool manager：{tool_manager}")
    print(f"tool manager de tool_executorsyounaxie:{tool_manager.tool_executors}")
    print_tools(tools)
    return tool_manager


async def main():
    # 1️⃣ 初始化 tool manager
    tool_manager = _init_tool_manager()

    # 2️⃣ 构造 code interpreter 输入
    coder_output = "nihao,qingyuedu dama ```python\nprint('hello, world')\n``` ,diama wanbi"
    coder_content = coder_output.split("```python")[-1].split("```")[0].strip()

    # 3️⃣ 调用 tool（关键：await 必须在 async 函数里）
    response_content, response_metadata = await tool_manager.execute_tool(
        "code_interpreter",
        {"code": coder_content},
        metadata={},   # 保留你原本的 metadata
    )

    # 4️⃣ 打印结果
    print("=== Tool Response Content ===")
    print(response_content)
    print("=== Tool Response Metadata ===")
    print(response_metadata)


if __name__ == "__main__":
    asyncio.run(main())
# status = response_metadata["status"]