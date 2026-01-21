import os
import re
import string
import logging
from typing import Any, Optional, List
import numpy as np
import random
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "WARN"))

def question_scorer(model_answer: str, ground_truth: str) -> bool:
    r"""Scorer for the GAIA benchmark.
    https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/
    scorer.py

    Args:
        model_answer (str): The model answer.
        ground_truth (str): The ground truth answer.

    Returns:
        bool: The score of the model
    """

    def is_float(element: Any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    if is_float(ground_truth):
        logger.info(f"Evaluating {model_answer} as a number.")
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)

    elif any(char in ground_truth for char in [",", ";"]):
        logger.info(
            f"Evaluating {model_answer} as a comma separated list."
        )
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        if len(gt_elems) != len(ma_elems):
            # logger.warning(
            #     "Answer lists have different lengths, returning False.",
            #     UserWarning,
            # )
            return False

        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                ma_elem = normalize_str(ma_elem, remove_punct=False)
                gt_elem = normalize_str(gt_elem, remove_punct=False)
                comparisons.append(ma_elem == gt_elem)
        return all(comparisons)
    else:
        logger.info(f"Evaluating {model_answer} as a string.")
        ma_elem = normalize_str(model_answer)
        gt_elem = normalize_str(ground_truth)
        return ma_elem == gt_elem


def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        logger.error(
            f"String {number_str} cannot be normalized to number str."
        )
        return float("inf")


def split_string(
    s: str, char_list: Optional[List[str]] = None
) -> list[str]:
    r"""Split a string based on a list of characters.

    Args:
        s (str): The string to split.
        char_list (Optional[List[str]], optional): T
            he list of characters to split on.
            (default: :obj:`None`)
    """
    if char_list is None:
        char_list = [",", ";"]
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def normalize_str(input_str, remove_punct=True) -> str:
    r"""Normalize a string.

    Args:
        input_str: The input string to normalize.
        remove_punct: Whether to remove punctuation.

    Returns:
        str: The normalized string.
    """
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()

def gaia_em_reward_fn(solution_str, ground_truth):
    pred_answer = solution_str.split("<answer>")[-1].split("</answer>")[0].strip()
    return question_scorer(pred_answer, ground_truth)


def gaia_em_reward_fn_ttt(solutions, ground_truth=None):
    answers = [solution_str.split("<answer>")[-1].split("</answer>")[0].strip() for solution_str in solutions]
    
    counts = Counter(answers)
    majority_answer , _ = counts.most_common(1)[0]

    rewards = [question_scorer(answer, majority_answer) for answer in answers]

    return rewards
