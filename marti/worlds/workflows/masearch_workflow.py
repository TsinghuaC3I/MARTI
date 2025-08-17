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
    """Extract list of search query strings from retriever LLM output.

    Expects blocks like:
      <tool_call>{\n  "name": "search",\n  "arguments": {\n    "query_list": ["ONE FOCUSED QUERY STRING"]\n  }\n}</tool_call>
    Returns all query strings found (order preserved). Robust to minor JSON formatting issues.
    """
    queries: List[str] = []
    for match in TOOL_CALL_PATTERN.finditer(text):
        raw_json = match.group(1)
        obj = None
        try:
            obj = json.loads(raw_json)
        except json.JSONDecodeError:
            # Attempt light repairs (remove trailing commas)
            repaired = re.sub(r",\s*([}\]])", r"\1", raw_json)
            try:
                obj = json.loads(repaired)
            except Exception:
                continue
        if not isinstance(obj, dict):
            continue
        if obj.get("name") != "search":
            continue
        args = obj.get("arguments", {})
        if isinstance(args, dict):
            ql = args.get("query_list")
            if isinstance(ql, list):
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
    assert tool_manager is not None, "tool_manager required"
    assert len(agents) >= 4, "Need four agents for MASearch"

    prompt_engineer = agents[0]
    planner = agents[1]
    retriever = agents[2]
    generator = agents[3]
    trajectory: List[Dict[str, Any]] = []

    # Prompts now sourced from agent chat_template definitions in config
    prompt_engineer_prompt = prompt_engineer.get("chat_template") or "Question: {question}"
    planner_prompt = planner.get("chat_template") or "Question: {question} Prompt: {prompt}"
    retriever_prompt = retriever.get("chat_template") or "Question: {question} Plan: {plan}"
    generator_prompt = generator.get("chat_template") or "Question: {question} Plan: {plan} Retrieved Information: {information}"

    trajectory: List[Dict[str, Any]] = []

    # ---- Prompt Engineer ----
    pe_input = apply_template_with_tokenizer(
        prompt_engineer["tokenizer"],
        prompt_engineer_prompt.format(question=prompt)
    )
    pe_resp = await prompt_engineer["llm"].generate_async.remote(
        pe_input,
        prompt_engineer["sampling_params"]
    )
    pe_text = pe_resp.outputs[0].text.strip()
    trajectory.append({
        "turn_id": 0,
        "agent_index": 0,
        "agent_name": prompt_engineer["agent_id"],
        "agent_role": prompt_engineer["agent_role"],
        "agent_input": pe_input,
        "agent_output": pe_text,
        "metadata": {}
    })

    # ---- Planner ----
    planner_input = apply_template_with_tokenizer(
        planner["tokenizer"],
        planner_prompt.format(question=prompt, prompt=pe_text)
    )
    planner_resp = await planner["llm"].generate_async.remote(
        planner_input,
        planner["sampling_params"]
    )
    planner_text = planner_resp.outputs[0].text.strip()
    trajectory.append({
        "turn_id": 1,
        "agent_index": 1,
        "agent_name": planner["agent_id"],
        "agent_role": planner["agent_role"],
        "agent_input": planner_input,
        "agent_output": planner_text,
        "metadata": {}
    })

    # ---- Retriever ----
    retriever_input = apply_template_with_tokenizer(
        retriever["tokenizer"],
        retriever_prompt.format(question=prompt, plan=planner_text)
    )
    retriever_resp = await retriever["llm"].generate_async.remote(
        retriever_input,
        retriever["sampling_params"]
    )
    retriever_output = retriever_resp.outputs[0].text
    # --- Extract search query content from retriever output ---
    search_queries = extract_search_queries(retriever_output)
    search_content = search_queries[0] if search_queries else ""
    if not search_content:
        logger.debug("No search query extracted from retriever output.")
    try:
        response_content, response_metadata = await tool_manager.execute_tool(
            "search", {"search": search_content}, metadata=metadata
        )
        status = response_metadata["status"]
    except Exception as e:
        response_content = f"ERROR"
        status = "failed"

    # status = response_metadata["status"]
    execution = search_content + \
        f"\nExecution status: {status}\nSearch output: {response_content}"
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
    },
    # Provide a neutral reward so processor doesn't fail on missing key
    "agent_reward": 0.0
    })

    # ---- Generator ----
    generator_input = apply_template_with_tokenizer(
        generator["tokenizer"],
        generator_prompt.format(question=prompt, plan=planner_text, information=execution)
    )
    gen_resp = await generator["llm"].generate_async.remote(
        generator_input,
        generator["sampling_params"]
    )
    gen_text = gen_resp.outputs[0].text
    trajectory.append({
        "turn_id": len(trajectory),
        "agent_index": 3,
        "agent_name": generator["agent_id"],
        "agent_role": generator["agent_role"],
        "agent_input": generator_input,
        "agent_output": gen_text,
        "metadata": {"original_prompt": prompt, "normalized_prompt": pe_text}
    })

    # ---- Verification (score subset: prompt engineer, planner, generator) ----
    outputs_for_scoring = [t for t in trajectory if t["agent_index"] in (0, 1, 3)]
    try:
        scores = auto_verify(
            task,
            1,
            [t["agent_output"] for t in outputs_for_scoring],
            [label] * len(outputs_for_scoring)
        )
    except Exception:
        scores = [0.0] * len(outputs_for_scoring)

    si = 0
    for t in trajectory:
        if t["agent_index"] in (0, 1, 3):
            t["agent_reward"] = scores[si]
            si += 1
        elif "agent_reward" not in t:
            # Ensure every turn has reward field
            t["agent_reward"] = 0.0

    return {
        "prompt": prompt,
        "normalized_prompt": pe_text,
        "label": label,
        "trajectory": trajectory,
        "final_reward": scores[-1] if scores else 0.0
    }