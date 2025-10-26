"""
Sequential experience workflow with optimized length control:
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

def get_token_length(text: str, tokenizer) -> int:
    """Calculate the token length of the given text using the tokenizer."""
    return len(tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0])

def calc_max_output_tokens(
    max_length: int,
    current_input_tokens: int,
    reserve_tokens: int = 0,
    min_output_tokens: int = 512,
    safety_margin: int = 100
) -> int:
    """
    Calculate maximum output tokens ensuring total conversation fits in max_length.
    
    Args:
        max_length: Total maximum length (input + output)
        current_input_tokens: Tokens used by current input
        reserve_tokens: Tokens to reserve for next stage
        min_output_tokens: Minimum tokens to allocate for output
        safety_margin: Safety buffer to prevent overflow
    
    Returns:
        Maximum tokens for output
    """
    available_tokens = max_length - current_input_tokens - reserve_tokens - safety_margin
    return max(min_output_tokens, available_tokens)

def estimate_memory_tokens(memory_text: str, tokenizer) -> int:
    """Estimate tokens needed for memory in next round's input."""
    if not memory_text.strip():
        return 0
    # Add some buffer for template formatting
    template_overhead = 50  
    return get_token_length(memory_text, tokenizer) + template_overhead

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
    Sequential experience workflow with integrated memory and optimized length control.
    - agents[0]: solver
    - agents[1]: memory writer (optional if only one model just fallback to agents[0])
    """
    wf_args = kwargs.get("workflow_args", {}) or {}
    num_rounds: int = int(wf_args.get("num_rounds", 4))
    memory_reward_style = wf_args.get("memory_reward_style", "propagate")
    use_solver_output_as_section: bool = bool(wf_args.get("memory_section_from_solver", True))
    
    # Length control parameters
    max_length = kwargs["max_length"]
    default_max_tokens = int(wf_args.get("default_max_tokens", 4096))
    min_solver_tokens = int(wf_args.get("min_solver_tokens", 512))
    min_memory_tokens = int(wf_args.get("min_memory_tokens", 256))
    safety_margin = int(wf_args.get("safety_margin", 100))
    
    # Dynamic adjustment thresholds
    prompt_ratio = get_token_length(prompt, agents[0]["tokenizer"]) / max_length
    
    # Adjust rounds based on prompt length
    if prompt_ratio > 0.75:
        logger.warning(f"Prompt uses {prompt_ratio:.1%} of max_length. Forcing single round.")
        num_rounds = 1
    elif prompt_ratio > 0.5:
        max_rounds = 2
        if num_rounds > max_rounds:
            logger.warning(f"Prompt uses {prompt_ratio:.1%} of max_length. Limiting to {max_rounds} rounds.")
            num_rounds = max_rounds
    elif prompt_ratio > 0.25:
        max_rounds = 3
        if num_rounds > max_rounds:
            logger.info(f"Prompt uses {prompt_ratio:.1%} of max_length. Limiting to {max_rounds} rounds.")
            num_rounds = max_rounds
    
    solver = agents[0]
    memory_writer = agents[1] if len(agents) > 1 else agents[0]

    # Base variables
    trajectory: List[Dict[str, Any]] = []
    rounds_dump: List[Dict[str, Any]] = []
    current_memory: str = ""
    last_solver_output: str = ""
    accumulated_memory_tokens = 0  # Track cumulative memory size
    
    # Main loop: T0 -> M0 -> T1 -> ... -> TK
    for r in range(num_rounds):
        is_last_round = (r == num_rounds - 1)
        skip_memory_this_round = False  # Initialize flag
        
        # ==== SOLVER TURN ====
        solver_input_raw = SOLVER_PROMPT_TEMPLATE.format(
            prompt=prompt,
            memory=current_memory if current_memory.strip() else "(empty)"
        )
        
        # Calculate solver input tokens
        solver_input_tokens = get_token_length(solver_input_raw, solver["tokenizer"])
        
        # Check if we have enough space to continue
        min_required_space = solver_input_tokens + min_solver_tokens + safety_margin
        if min_required_space > max_length:
            logger.error(f"Round {r}: Not enough space for solver. Required: {min_required_space}, Available: {max_length}")
            # Return what we have so far
            if trajectory:
                final_reward = trajectory[-1].get("agent_reward", 0.0)
            else:
                final_reward = 0.0
            break
        
        # Calculate reserve for next memory writer (if not last round)
        if not is_last_round:
            # Estimate memory writer's needs more accurately
            estimated_solver_output = min(
                solver_max_tokens if 'solver_max_tokens' in locals() else default_max_tokens // 2,
                2048
            )
            estimated_memory_input = (
                get_token_length(prompt, memory_writer["tokenizer"]) +
                accumulated_memory_tokens +
                estimated_solver_output +
                200  # Template overhead
            )
            
            # Calculate memory reserve more conservatively
            ideal_memory_reserve = min(default_max_tokens, estimated_memory_input + min_memory_tokens)
            available_for_reserve = max_length - solver_input_tokens - min_solver_tokens - safety_margin
            memory_writer_reserve = min(ideal_memory_reserve, max(0, available_for_reserve))
            
            # If we can't reserve enough for memory, skip memory generation
            if memory_writer_reserve < min_memory_tokens + 500:  # 500 for minimum viable memory input
                logger.warning(f"Round {r}: Insufficient space for memory generation. Continuing without memory update.")
                memory_writer_reserve = 0
                skip_memory_this_round = True
            else:
                skip_memory_this_round = False
        else:
            memory_writer_reserve = 0
            skip_memory_this_round = True
        
        # Calculate max tokens for solver
        solver_max_tokens = calc_max_output_tokens(
            max_length=max_length,
            current_input_tokens=solver_input_tokens,
            reserve_tokens=memory_writer_reserve,
            min_output_tokens=min_solver_tokens,
            safety_margin=safety_margin
        )
        
        # Ensure we don't exceed default_max_tokens
        solver_max_tokens = min(solver_max_tokens, default_max_tokens)
        
        # Update solver's sampling params
        solver["sampling_params"].max_tokens = solver_max_tokens
        
        # Generate solver output
        solver_obs = await gen_loop(
            solver, 
            tool_manager, 
            solver_input_raw, 
            label, 
            metadata, 
            max_length
        )

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
            "metadata": {
                "memory_used": bool(current_memory.strip()),
                "input_tokens": solver_input_tokens,
                "max_output_tokens": solver_max_tokens
            }
        })

        round_rec: Dict[str, Any] = {
            "round": r,
            "solver": {
                "input_raw": solver_input_raw,
                "output": solver_out,
                "input_tokens": solver_input_tokens,
                "max_output_tokens": solver_max_tokens
            }
        }

        # Extract solver output for memory section
        last_solver_output = "\n\n".join(solver_out).replace(
            "<|im_end|>\n<|im_start|>assistant", ""
        ).replace(
            "\n<|im_start|>user\n", ""
        ).replace(
            "<|im_end|>", ""
        )
        
        # Truncate solver output if it's too long for memory processing
        solver_output_tokens = get_token_length(last_solver_output, solver["tokenizer"])
        if solver_output_tokens > default_max_tokens // 2:
            # Keep only the most important parts (beginning and end)
            max_chars = len(last_solver_output) * (default_max_tokens // 2) // solver_output_tokens
            if max_chars > 1000:
                last_solver_output = last_solver_output[:max_chars//2] + "\n...\n" + last_solver_output[-max_chars//2:]
            logger.info(f"Truncated solver output from {solver_output_tokens} to ~{default_max_tokens//2} tokens for memory processing.")

        # If last round or skip memory, skip memory creation
        if is_last_round or skip_memory_this_round:
            rounds_dump.append(round_rec)
            if is_last_round:
                break
            else:
                continue  # Skip to next round without updating memory
        
        # ==== MEMORY WRITER TURN ====
        section_text = last_solver_output if use_solver_output_as_section else ""
        
        # Compress memory if it's getting too large
        if accumulated_memory_tokens > max_length // 6:  # If memory > 1/6 of max_length
            logger.info(f"Memory size ({accumulated_memory_tokens}) is large. Applying compression.")
            # Truncate current memory to keep only recent parts
            memory_lines = current_memory.split('\n')
            if len(memory_lines) > 10:
                current_memory = '\n'.join(memory_lines[-10:])  # Keep last 10 lines
                accumulated_memory_tokens = get_token_length(current_memory, memory_writer["tokenizer"])
        
        mem_input_raw = MEMORY_PROMPT_TEMPLATE.format(
            prompt=prompt,
            memory=current_memory,
            section=section_text
        )
        
        # Calculate memory writer input tokens
        mem_input_tokens = get_token_length(mem_input_raw, memory_writer["tokenizer"])
        
        # Check if memory input itself is too large
        if mem_input_tokens > max_length - min_memory_tokens - safety_margin:
            logger.warning(f"Round {r}: Memory input too large ({mem_input_tokens}). Skipping memory update.")
            rounds_dump.append(round_rec)
            continue
        
        # Calculate max tokens for memory writer
        # No need to reserve for next round as memory will be part of next solver's input
        mem_max_tokens = calc_max_output_tokens(
            max_length=max_length,
            current_input_tokens=mem_input_tokens,
            reserve_tokens=0,
            min_output_tokens=min_memory_tokens,
            safety_margin=safety_margin
        )
        
        # Ensure we don't exceed default_max_tokens
        mem_max_tokens = min(mem_max_tokens, default_max_tokens)
        
        # Also ensure memory doesn't grow too large for future rounds
        # Estimate maximum memory size that would fit in future rounds
        remaining_rounds = num_rounds - r - 1
        if remaining_rounds > 0:
            # Conservative estimate: memory shouldn't exceed 1/4 of max_length
            max_memory_size = max_length // 4
            estimated_current_memory_tokens = estimate_memory_tokens(
                current_memory, 
                memory_writer["tokenizer"]
            )
            mem_max_tokens = min(
                mem_max_tokens,
                max_memory_size - estimated_current_memory_tokens
            )
        
        # Update memory writer's sampling params
        memory_writer["sampling_params"].max_tokens = mem_max_tokens
        
        # Generate memory
        mem_out, mem_input_templated = await gen_one(memory_writer, mem_input_raw)
        mem_inner = extract_inner_memory(mem_out)
        
        # Update accumulated memory tokens
        accumulated_memory_tokens = get_token_length(mem_inner, memory_writer["tokenizer"])

        trajectory.append({
            "turn_id": len(trajectory),
            "round": r,
            "agent_index": 0,
            "agent_name": memory_writer.get("agent_id", "memory"),
            "agent_role": "memory",
            "agent_input": mem_input_templated,
            "agent_output": mem_out,
            "metadata": {
                "memory_inner": mem_inner,
                "input_tokens": mem_input_tokens,
                "max_output_tokens": mem_max_tokens,
                "memory_tokens": accumulated_memory_tokens
            }
        })

        round_rec["memory"] = {
            "input_raw": mem_input_raw,
            "raw_output": mem_out,
            "extracted": mem_inner,
            "input_tokens": mem_input_tokens,
            "max_output_tokens": mem_max_tokens,
            "accumulated_memory_tokens": accumulated_memory_tokens
        }
        rounds_dump.append(round_rec)

        # Update memory for next round
        current_memory = mem_inner
        
        # Check if we're approaching length limits
        if accumulated_memory_tokens > max_length // 3:
            logger.warning(
                f"Memory size ({accumulated_memory_tokens} tokens) exceeds 1/3 of max_length. "
                f"Consider truncating or summarizing memory."
            )

    # Only evaluate the LAST trajectory TK
    final_reward = solver_reward

    # Assign memory rewards
    assign_memory_rewards(trajectory, memory_reward_style)

    # Final Return
    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory[:-1] if len(trajectory) > 1 else trajectory,
        "final_reward": final_reward,
        "metadata": {
            "num_rounds": num_rounds,
            "max_length": max_length,
            "default_max_tokens": default_max_tokens,
            "final_memory_tokens": accumulated_memory_tokens
        }
    }