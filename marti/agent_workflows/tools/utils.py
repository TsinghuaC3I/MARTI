
def convert_mcp_to_openai_tools(mcp_tools: list) -> list:
    openai_tools = []

    for tool in mcp_tools.tools:
        tool_schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {}
            }
        }

        input_schema = tool.inputSchema

        parameters = {
            "type": input_schema['type'],
            "properties": input_schema['properties'],
            "required": input_schema['required'],
            "additionalProperties": False
        }

        for prop in parameters["properties"].values():
            if "enum" in prop:
                prop["description"] = f"Optional: {', '.join(prop['enum'])}"

        tool_schema["function"]["parameters"] = parameters
        openai_tools.append(tool_schema)

    # print("\nconverte to openai tools success:", [openai_tools])
    return openai_tools
