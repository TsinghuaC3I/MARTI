"""Workflow: Multi-agent Search (MASearch)
Pattern: Prompt Engineer -> Planner -> Retriever (iterative JSON tool calls) -> Generator
"""
import os
from typing import Dict, List, Any, Optional
import asyncio
import re
import json
from marti.helpers.logging import init_logger
from marti.verifiers.auto_verify import auto_verify
from marti.worlds.workflows.utils import apply_template_with_tokenizer
from marti.worlds.steps.mcp_step import step_with_tools

logger = init_logger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))

# Pattern to extract JSON payload inside <tool_call>...</tool_call>
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*({.*?})\s*</tool_call>", re.DOTALL)

def extract_search_queries(text: str) -> List[str]:
    queries: List[str] = []
    for match in TOOL_CALL_PATTERN.finditer(text):
        raw_json = match.group(1)
        obj = None
        try:
            obj = json.loads(raw_json)
        except Exception as e:
            logger.warning(f"Failed to parse tool_call JSON: {e}. Raw: {raw_json}")
            continue
        if not isinstance(obj, dict):
            logger.warning(f"tool_call JSON is not a dict: {obj}")
            continue
        name = obj.get("name")
        if name != "search":
            logger.warning(f"tool_call JSON missing 'name'='search': {obj}")
            continue
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            logger.warning(f"tool_call 'arguments' is not a dict: {args}")
            continue
        ql = args.get("query_list")
        if not isinstance(ql, list):
            logger.warning(f"tool_call 'query_list' is not a list: {ql}")
            continue
        for q in ql:
            if isinstance(q, str):
                qs = q.strip()
                if qs:
                    queries.append(qs)
    return queries

# --- Main Workflow -----------------------------------------------------------
async def workflow(
    prompt: str,
    label: str,
    agents: List[Dict[str, Any]],
    tool_manager,
    task: str,
    metadata: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """MASearch workflow orchestrating 4 agents.

    Expected agents list order:
      0: Prompt Engineer
      1: Planner
      2: Retriever
      3: Generator
    """
    assert tool_manager is not None
    assert len(agents) >= 4

    # Identify agents
    prompt_engineer = agents[0]
    planner = agents[1]
    retriever = agents[2]
    generator = agents[3]

    # Prompts
    prompt_engineer_prompt = "You are a great prompt engineer, please write a clear prompt for the given question to the planner. Question: {question}"
    planner_prompt = "Task: Produce an execution plan to answer the prompt. You need to refactor the problem into atomic steps for the generator. If external knowledge is needed, wrap what to retrieve in <retrieval></retrieval> and intended usage in <target></target>. Question: {question}, Prompt: {prompt}"
    retriever_prompt = (
        "You are a retrieval-oriented answer generator. For any needed external info inside <retrieval></retrieval> in the plan, design focused queries. "
        "Emit EACH query in a <tool_call>{{\\n  \"name\": \"search\",\\n  \"arguments\": {{\\n    \"query_list\": [\"ONE FOCUSED QUERY STRING\"]\\n  }}\\n}}</tool_call>. "
        "Tool responses will appear between <tool_response> and </tool_response>. You don't need to answer the question, you only need to design the query. "
        "Question: {question} Plan: {plan}"
    )
    generator_prompt = "You are a great writer who is good at organizing different information. Based on the retrieved content and their corresponding use in <target> and </target>, generate the response to the question. Question: {question} Plan: {plan} Retrieved Information: {information}"

    trajectory = []

    # Prompt Engineer
    pe_input = apply_template_with_tokenizer(prompt_engineer["tokenizer"], prompt_engineer_prompt.format(question=prompt))
    pe_resp = await prompt_engineer["llm"].generate_async.remote(pe_input, prompt_engineer["sampling_params"])
    pe_output = pe_resp.outputs[0].text.strip()
    trajectory.append({
        "turn_id": 0,
        "agent_index": 0,
        "agent_name": prompt_engineer["agent_id"],
        "agent_role": prompt_engineer["agent_role"],
        "agent_input": pe_input,
        "agent_output": pe_output,
        "metadata": {}
    })

    # Planner
    planner_input = apply_template_with_tokenizer(planner["tokenizer"], planner_prompt.format(question=prompt, prompt=pe_output))
    planner_resp = await planner["llm"].generate_async.remote(planner_input, planner["sampling_params"])
    planner_output = planner_resp.outputs[0].text.strip()
    trajectory.append({
        "turn_id": 1,
        "agent_index": 1,
        "agent_name": planner["agent_id"],
        "agent_role": planner["agent_role"],
        "agent_input": planner_input,
        "agent_output": planner_output,
        "metadata": {}
    })

    # Retriever
    retriever_input = apply_template_with_tokenizer(
        retriever["tokenizer"],
        retriever_prompt.format(question=prompt, plan=planner_output)
    )
    retriever_resp = await retriever["llm"].generate_async.remote(
        retriever_input,
        retriever["sampling_params"]
    )
    retriever_output = retriever_resp.outputs[0].text
    search_queries = extract_search_queries(retriever_output)
    # Use only the first valid query, like MathChat's code block extraction
    search_content = search_queries[0] if search_queries else ""
    # Execute the tool (search)
    try:
        response_content, response_metadata = await tool_manager.execute_tool(
            "search", {"search": search_content}, metadata=metadata
        )
        status = response_metadata["status"]
    except Exception:
        response_content = "ERROR"
        status = "failed"
    # Format execution output for generator, similar to MathChat
    execution = search_content + f"\nExecution status: {status}\nSearch output: {response_content[:512]}"
    trajectory.append({
        "turn_id": 2,
        "agent_index": 2,
        "agent_name": retriever["agent_id"],
        "agent_role": retriever["agent_role"],
        "agent_input": retriever_input,
        "agent_output": retriever_output,
        "metadata": {
            "status": status,
            "response": response_content
        }
    })

    # Generator
    generator_input = apply_template_with_tokenizer(generator["tokenizer"], generator_prompt.format(question=prompt, plan=planner_output, information=execution))
    generator_resp = await generator["llm"].generate_async.remote(generator_input, generator["sampling_params"])
    generator_output = generator_resp.outputs[0].text
    trajectory.append({
        "turn_id": 3,
        "agent_index": 3,
        "agent_name": generator["agent_id"],
        "agent_role": generator["agent_role"],
        "agent_input": generator_input,
        "agent_output": generator_output,
        "metadata": {"original_prompt": prompt, "normalized_prompt": pe_output}
    })

    # Unified scoring for all outputs
    all_outputs = [pe_output, planner_output, retriever_output, generator_output]
    all_rewards = auto_verify(task, 1, all_outputs, [label] * len(all_outputs))
    for turn, reward in zip(trajectory, all_rewards):
        turn["agent_reward"] = reward

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "final_reward": all_rewards[-1]
    }