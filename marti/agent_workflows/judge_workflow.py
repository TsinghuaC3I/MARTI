# generator_judge_workflow.py
"""
A configurable, single-turn workflow where a 'generator' agent proposes an answer,
and a 'judge' agent scores it. Supports custom judge templates and scoring methods.
"""
import os
import re
from typing import Dict, List, Any, Optional, Callable
from marti.utils.logging_utils import init_logger
from marti.verifiers.auto_verify import auto_verify
from marti.agent_workflows.utils import apply_template_with_tokenizer

logger = init_logger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))


def _find_agent(name: str, agents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Helper to find an agent by name from the agent list."""
    return next((agent for agent in agents if agent["name"] == name), None)


def _parse_score_regex(text: str) -> float:
    """
    Default score parser: looks for pattern like "Score: 8/10" or "Score: 8".
    """
    # Look for "Score: X/Y" or "Score: X"
    match = re.search(
        r'Score:\s*(\d+(\.\d+)?)(?:\s*/\s*\d+)?', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    # Fallback: if no explicit score, return a neutral value.
    return 0.0


def _parse_score_keywords(text: str) -> float:
    """
    Keyword-based score parser for binary judgments.
    Example: "REVIEW_1_BETTER" -> 1.0, "REVIEW_2_BETTER" -> 0.0
    """
    response_text = text.upper().strip()
    if "REVIEW_1_BETTER" in response_text or "FIRST_BETTER" in response_text or "OPTION_1" in response_text:
        return 1.0
    elif "REVIEW_2_BETTER" in response_text or "SECOND_BETTER" in response_text or "OPTION_2" in response_text:
        return 0.0
    else:
        logger.warning(
            f"Judge output did not match expected keywords: '{response_text}'. Assigning default score 0.5.")
        return 0.5


def _parse_score_normalize(text: str) -> float:
    """
    Normalized score parser: extracts any number and normalizes to [0, 1].
    """
    # Look for any number in the text
    match = re.search(r'(\d+(?:\.\d+)?)', text)
    if match:
        score = float(match.group(1))
        # Assume score is out of 10 if > 1, otherwise assume already normalized
        if score > 1:
            return min(score / 10.0, 1.0)
        else:
            return min(score, 1.0)
    return 0.0


def _get_score_parser(parser_type: str) -> Callable[[str], float]:
    """Get the appropriate score parser function."""
    parsers = {
        "regex": _parse_score_regex,
        "keywords": _parse_score_keywords,
        "normalize": _parse_score_normalize,
    }
    return parsers.get(parser_type, _parse_score_regex)

async def workflow(
    prompt: str,
    label: str,
    agents: List[Dict[str, Any]],
    tool_manager=None,
    task: str = "review_group",
    metadata: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Configurable Generator-Judge workflow with custom templates and scoring methods.

    Workflow args:
    - judge_template: Custom judge prompt template with placeholders {prompt}, {generated_answer}, {label}
    - score_parser: Score parsing method ("regex", "keywords", "normalize")
    - judge_weight: Weight for combining judge score with rule-based reward (0.0-1.0)
    - label_separator: Separator for parsing structured labels (e.g., "||DIV REVIEW SCORE||")

    Returns trajectory format matching debate_workflow.py by default.
    """
    # Extract workflow configuration
    workflow_args = kwargs.get("workflow_args", {})

    generator_agent = agents[0]
    judge_agent = agents[1]

    score_parser_type = workflow_args.get("score_parser", "regex")
    judge_weight = workflow_args.get(
        "judge_weight", 1.0)  # Default: only judge score
    label_separator = workflow_args.get("label_separator", None)

    # Get score parser function
    score_parser = _get_score_parser(score_parser_type)

    # 2. Parse label if structured (e.g., "review||DIV REVIEW SCORE||score")
    if label_separator and label_separator in label:
        label_parts = label.split(label_separator)
        ground_truth_content = label_parts[0]
        ground_truth_score = label_parts[1] if len(label_parts) > 1 else label
    else:
        ground_truth_content = label
        ground_truth_score = score_parser(label)
        # ground_truth_score = label

    # 3. Generator produces the answer
    generator_input = apply_template_with_tokenizer(
        generator_agent["tokenizer"],
        prompt
    )
    generator_input_token_ids = generator_agent["tokenizer"](
        generator_input,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].tolist()
    
    generator_response = await generator_agent["llm"].generate_async.remote(
        prompt_ids=generator_input_token_ids,
        sampling_params=generator_agent["sampling_params"]
    )
    generated_answer = generator_response.outputs[0].text.strip()
    
    generator_output_token_ids = generator_response.outputs[0].token_ids
    generator_sequence_ids = generator_input_token_ids + generator_output_token_ids

    # Extract rollout_log_probs for generator
    generator_rollout_log_probs = None
    if generator_agent["sampling_params"].logprobs is not None:
        generator_rollout_log_probs = [0.0] * len(generator_input_token_ids)
        if hasattr(generated_answer.outputs[0], "logprobs") and generated_answer.outputs[0].logprobs is not None:
            for i, logprob_dict in enumerate(generated_answer.outputs[0].logprobs):
                if i < len(generator_output_token_ids) and generator_output_token_ids[i] in logprob_dict:
                    generator_rollout_log_probs.append(logprob_dict[generator_output_token_ids[i]].logprob)
                else:
                    generator_rollout_log_probs.append(0.0)
        else:
            generator_rollout_log_probs.extend([0.0] * len(generator_output_token_ids))

    # 4. Judge evaluates using custom template
    # Default judge template (backward compatible)
    judge_template = """Please evaluate the following answer based on the problem. Provide a clear rationale and a final score on a scale of 1 to 10 (e.g., "Score: 8/10").

[Problem]
{prompt}

[Generated Answer]
{generated_answer}
"""
    if workflow_args.get("judge_template", None):
        judge_template = workflow_args.get("judge_template")
    
    judge_input = judge_template.format(
        prompt=prompt.split("You are an expert reviewer. You should aim to provide the most reliable review results by conducting a thorough analysis of the paper. I will provide you with three different questions about the paper's background knowledge and the search results for each question.\n\nI will help you with the search.")[-1][:300],
        generated_answer=generated_answer.split("</think>")[-1],
        label=ground_truth_content,
        ground_truth=ground_truth_content,
        score=ground_truth_score
    )

    judge_input = apply_template_with_tokenizer(
        judge_agent["tokenizer"],
        judge_input
    )

    judge_input_token_ids = judge_agent["tokenizer"](
        judge_input,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].tolist()

    judge_response = await judge_agent["llm"].generate_async.remote(
        prompt_ids=judge_input_token_ids,
        sampling_params=judge_agent["sampling_params"]
    )
    judge_output = judge_response.outputs[0].text.strip()
    
    judge_output_token_ids = judge_response.outputs[0].token_ids
    judge_sequence_ids = judge_input_token_ids + judge_output_token_ids

    # Extract rollout_log_probs for judge
    judge_rollout_log_probs = None
    if judge_agent["sampling_params"].logprobs is not None:
        judge_rollout_log_probs = [0.0] * len(judge_input_token_ids)
        if hasattr(judge_response.outputs[0], "logprobs") and judge_response.outputs[0].logprobs is not None:
            for i, logprob_dict in enumerate(judge_response.outputs[0].logprobs):
                if i < len(judge_output_token_ids) and judge_output_token_ids[i] in logprob_dict:
                    judge_rollout_log_probs.append(logprob_dict[judge_output_token_ids[i]].logprob)
                else:
                    judge_rollout_log_probs.append(0.0)
        else:
            judge_rollout_log_probs.extend([0.0] * len(judge_output_token_ids))

    # 5. Extract score using configurable parser
    judge_score = score_parser(judge_output)

    # 6. Get rule-based reward if weight < 1.0
    if judge_weight < 1.0:
        rule_reward = auto_verify(task, 1, [generated_answer], [
                                  ground_truth_score])[0]
    else:
        rule_reward = 0.0

    # 7. Combine rewards
    # Generator gets judge_score, judge gets (1 - judge_score) for binary comparison tasks
    # For single evaluation tasks, both can get the same score
    generator_judge_reward = judge_score

    # Combine with rule-based reward
    generator_combined_reward = generator_judge_reward * \
        judge_weight + rule_reward * (1 - judge_weight)
    # Placeholder. The judge should not be updated.
    judge_combined_reward = 0.0

    # Clip rewards to [0, 1]
    generator_combined_reward = max(0.0, min(1.0, generator_combined_reward))
    # Placeholder. The judge should not be updated.
    judge_combined_reward = 0.0

    # 8. Return format based on configuration
    # New trajectory format matching debate_workflow.py
    trajectory = [
        # Generator trajectory
        {
            "turn_id": 0,
            "agent_index": 0,
            "agent_id": generator_agent["agent_id"],
            "agent_name": generator_agent["agent_id"],
            "agent_role": generator_agent["agent_role"],
            "agent_input": generator_input,
            "agent_output": generated_answer,
            "output_ids": generator_output_token_ids,
            "sequence_ids": generator_sequence_ids,
            "rollout_log_prob": generator_rollout_log_probs,
            "reward": generator_combined_reward,
            "metadata": {
                "judge_score": judge_score,
                "rule_reward": rule_reward,
                "generator_combined_reward": generator_combined_reward,
            },
        },
        # Judge trajectory
        {
            "turn_id": 1,
            "agent_index": 1,
            "agent_id": judge_agent["agent_id"],
            "agent_name": judge_agent["agent_id"],
            "agent_role": judge_agent["agent_role"],
            "agent_input": judge_input,
            "agent_output": judge_output,
            "output_ids": judge_output_token_ids,
            "sequence_ids": judge_sequence_ids,
            "rollout_log_prob": judge_rollout_log_probs,
            "reward": judge_combined_reward,
            "metadata": {},
        }
    ]

    # Final reward is the generator's combined reward
    final_reward = generator_combined_reward

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "final_reward": final_reward,
    }
