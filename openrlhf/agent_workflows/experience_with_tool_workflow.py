"""
Sequential experience workflow:
T0 (no memory) -> M0 -> T1 -> M1 -> ... -> TK (final)
- Only the LAST trajectory TK is evaluated to produce final_reward.
- No reward is computed nor injected for intermediate trajectories.
- Prompts no contain any reward info.
- Memory generation is integrated here (single-file workflow).
- For each sample, save a full JSON log (no printing) under LOG_DIR.
"""

import os
import re
import json
import time
from typing import Dict, List, Any, Optional

from openrlhf.utils.logging_utils import init_logger
from openrlhf.verifiers.auto_verify import auto_verify
from openrlhf.agent_workflows.utils import apply_template_with_tokenizer
from marti.worlds.steps.mcp_step import step_with_tools

logger = init_logger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))


# Memory & Solver (updated)

MEMORY_PROMPT_TEMPLATE = """You are presented with a problem, a section of an article or a solution attempt,
and a previous memory. Update the memory with general, reusable rules that could help answer the problem next time.

Guidelines:
- Do not restate or infer the final answer.
- Prefer checklists, guardrails, tool-choice heuristics, and validation steps.
- Keep it concise and generalizable; avoid task-specific entities, numbers, or names.
- Align with the search protocol that uses <think>, <tool_call>, <tool_response>, and <answer>.
- Focus on: when to search, how to craft queries, how to triage results, how to verify/stop, and how to format outputs.

<problem>
{prompt}
</problem>

<previous_memory>
{memory}
</previous_memory>

<section>
{section}
</section>

Updated memory (concise bullet points; avoid task-specific entities):
"""

SOLVER_PROMPT_TEMPLATE = """You are presented with a problem and a previous memory.
Use the memory only as heuristics; do not quote or restate it verbatim, and ignore it if irrelevant.

Follow this search protocol strictly:
1) Always begin with private reasoning inside <think>...</think>.
2) If your reasoning shows you lack knowledge or need verification, call the search engine inside <tool_call>...</tool_call>.
   The engine returns top results inside <tool_response>...</tool_response>. You may search multiple times.
3) After each <tool_response>, continue private reasoning in <think> to synthesize and decide next steps.
4) When you have enough information, output only the final answer inside <answer>...</answer>, with no extra text. For example, <answer> Beijing </answer>.

Constraints:
- Do not reveal the content of <think>, <tool_call>, or <tool_response> in your final output.
- Be concise and avoid step-by-step explanations in the final answer.
- Ignore the memory if it does not help.

<problem>
{prompt}
</problem>

<previous_memory>
{memory}
</previous_memory>

Your output:
"""

BEGIN_RE = re.compile(r"\bBEGIN[_ ]MEMORY\b", re.I)
END_RE = re.compile(r"\bEND[_ ]MEMORY\b", re.I)


def extract_inner_memory(text: str) -> str:
    """If BEGIN/END markers appear, keep the inner part; otherwise keep the whole text.
    Also normalize lines to "- ..." bullets and remove xml-ish tags and angle brackets.
    """
    m = re.search(r"BEGIN[_ ]MEMORY(.*?)END[_ ]MEMORY", text, re.S | re.I)
    inner = m.group(1) if m else text
    inner = re.sub(r"\s*</?[^>\s]+[^>]*>\s*", " ", inner)  # strip xml-ish
    inner = inner.replace("<", " less than ").replace(">", " greater than ")
    lines = []
    for ln in inner.splitlines():
        s = ln.strip()
        if not s:
            continue
        if BEGIN_RE.search(s) or END_RE.search(s):
            continue
        if not s.startswith("- "):
            s = "- " + s.lstrip("- ")
        lines.append(s)
    return "\n".join(lines).strip()


async def gen_one(agent_cfg: Dict[str, Any], raw_text: str) -> (str, str):
    """Sequential single-call generation.

    Returns:
    (output_text, templated_input_prompt)."""
    input_prompt = apply_template_with_tokenizer(agent_cfg["tokenizer"], raw_text)
    result = await agent_cfg["llm"].generate_async.remote(
        input_prompt, agent_cfg["sampling_params"]
    )
    return result.outputs[0].text, input_prompt

async def gen_loop(agent_cfg, tool_manager, raw_text, label, metadata, max_length):
    prompt = apply_template_with_tokenizer(agent_cfg["tokenizer"], raw_text, tool_manager.tools)
    result = await agent_cfg["llm"].get_trajectory.remote(tool_manager, agent_cfg["tokenizer"], max_length, prompt, label, agent_cfg["sampling_params"], metadata)
    return result

def safe_name(s: str) -> str:
    """Make a filesystem-safe filename stem."""
    s = str(s)
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:200] or "sample"

def assign_memory_rewards(
    traj: List[Dict[str, Any]],
    style: str = "propagate",
) -> None:
    """Post-hoc memory reward assignment in-place."""
    for i, rec in enumerate(traj):
        if rec.get("agent_role") != "memory":
            continue

        prev_r = traj[i-1]["agent_reward"]
        next_r = traj[i+1]["agent_reward"]

        if style == "propagate":
            rec["agent_reward"] = next_r
        else:  # 'delta'
            rec["agent_reward"] = 1.0 if (next_r - prev_r) > 0 else 0.0

def get_len(text, hf_tokenizer):
    return  len(hf_tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0])

async def workflow(
    prompt: str,
    label: str,
    agents: List[Dict[str, Any]],
    tool_manager,
    task: str,
    metadata: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """Sequential experience workflow with integrated memory.
    - agents[0]: solver
    - agents[1]: memory writer (optional if only one model just fallback to agents[0])
    """
    wf_args = kwargs.get("workflow_args", {}) or {}
    num_rounds: int = int(wf_args.get("num_rounds", 4))
    memory_reward_style = wf_args.get("memory_reward_style", "propagate")
    use_solver_output_as_section: bool = bool(wf_args.get("memory_section_from_solver", True))
    max_length = kwargs["max_length"]
    
    max_length_mem = wf_args.get("max_length_mem")
    max_length_agent = wf_args.get("max_length_agent")
    # log_path = wf_args.get("log_save_path")
    # # Directory to give per-sample JSON logs 
    # LOG_DIR = os.environ.get("WORKFLOW_JSON_DIR", log_path)

    solver = agents[0]
    memory_writer = agents[1] if len(agents) > 1 else agents[0]

    # base var
    trajectory: List[Dict[str, Any]] = []
    rounds_dump: List[Dict[str, Any]] = []   # for JSON log
    current_memory: str = ""                 # M_{-1} = empty
    last_solver_output: str = ""
    
    # item_id = None
    # if isinstance(metadata, dict):
    #     item_id = metadata.get("id") or metadata.get("indice") or metadata.get("index") or metadata.get("uid")
    # if not item_id:
    #     # fallback to hash of prompt
    #     item_id = f"hash_{abs(hash(prompt))}"
    # fname_stem = safe_name(item_id)
    # os.makedirs(LOG_DIR, exist_ok=True)
    # json_path = os.path.join(LOG_DIR, f"{fname_stem}.json")

    """
    max-length
    Prompt -> Multi-turn Rollout; max-length = max-length - max-length-mem
    Prompt + Multi-turn Rollout -> Memory; max-length = max-length
    Prompt + Memory -> Multi-turn Rollout; max-length = 
    """

    # Rounds: T0 -> M0 -> T1 -> ... -> TK 
    for r in range(num_rounds):
        # Solver turn: T_r 
        solver_input_raw = SOLVER_PROMPT_TEMPLATE.format(
            prompt=prompt,
            memory=current_memory if current_memory.strip() else "(empty)"
        )
        solver["sampling_params"].max_tokens = max_length - get_len(solver_input_raw, solver["tokenizer"]) - max_length_mem
        # solver_out, solver_input_templated = await gen_one(solver, solver_input_raw)
        solver_obs = await gen_loop(solver, tool_manager, solver_input_raw, label, metadata, max_length)

        observation = solver_obs["observation"]
        solver_input_templated = observation[0]
        solver_out = observation[1:]

        solver_reward = auto_verify(task, 1, [solver_out[-1]], [label])[0]

        trajectory.append({
            "turn_id": len(trajectory),
            "round": r,
            "agent_index": 0,
            "agent_name": solver.get("agent_id", "solver"),
            "agent_role": "solver",
            "agent_input": solver_input_templated,
            "agent_output": solver_out,
            "agent_reward": solver_reward,
            "metadata": {"memory_used": bool(current_memory.strip())}
        })

        round_rec: Dict[str, Any] = {
            "round": r,
            "solver": {
                "input_raw": solver_input_raw,
                "output": solver_out,
            }
        }

        # TODO: remove hard-coded!
        last_solver_output = "\n\n".join(solver_out).replace("<|im_end|>\n<|im_start|>assistant", "").replace("\n<|im_start|>user\n", "").replace("<|im_end|>", "")

        # If last round, skip memory creation
        if r == num_rounds - 1:
            rounds_dump.append(round_rec)
            break

        # Memory turn: M_r 
        section_text = last_solver_output if use_solver_output_as_section else ""
        mem_input_raw = MEMORY_PROMPT_TEMPLATE.format(
            prompt=prompt,
            memory=current_memory,
            section=section_text
        )
        memory_writer["sampling_params"].max_tokens = max_length - get_len(mem_input_raw, memory_writer["tokenizer"])
        mem_out, mem_input_templated = await gen_one(memory_writer, mem_input_raw)
        mem_inner = extract_inner_memory(mem_out)

        trajectory.append({
            "turn_id": len(trajectory),
            "round": r,
            "agent_index": 0,
            "agent_name": memory_writer.get("agent_id", "memory"),
            "agent_role": "memory",
            "agent_input": mem_input_templated,
            "agent_output": mem_out,
            "metadata": {"memory_inner": mem_inner}
        })

        round_rec["memory"] = {
            "input_raw": mem_input_raw,
            "raw_output": mem_out,
            "extracted": mem_inner,
        }
        rounds_dump.append(round_rec)

        # Inject memory into next round
        current_memory = mem_inner

    # Only evaluate the LAST trajectory TK 
    # verify_scores = auto_verify(task, 1, [last_solver_output], [label])
    # final_reward = float(verify_scores[0])
    final_reward = solver_reward

    #  Persist JSON log per sample 
    # dump_obj = {
    #     "id": item_id,
    #     "task": task,
    #     "num_rounds": num_rounds,
    #     "agent": {
    #         "solver": solver.get("agent_id", "solver"),
    #         "memory_writer": memory_writer.get("agent_id", "solver"),
    #     },
    #     "prompt": prompt,
    #     "label": label,
    #     "rounds": rounds_dump,
    #     "final_reward": final_reward,
    #     "meta": metadata if isinstance(metadata, dict) else {},
    # }

    # try:
    #     with open(json_path, "w", encoding="utf-8") as f:
    #         json.dump(dump_obj, f, ensure_ascii=False, indent=2)
    # except Exception as e:
    #     # Avoid crashing 
    #     logger.warning(f"Failed to write JSON log to {json_path}: {e}")

    assign_memory_rewards(trajectory, memory_reward_style)

    # Final Return 
    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory[:-1],
        "final_reward": final_reward
    }
