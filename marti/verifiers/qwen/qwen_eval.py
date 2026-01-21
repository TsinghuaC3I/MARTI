import concurrent.futures
from typing import List
from collections import Counter
from marti.verifiers.qwen.qwen_math_parser import extract_answer
from marti.verifiers.qwen.math_grade import grade_answer
from marti.verifiers.qwen.grader import math_equal as qwen_math_equal

def qwen_reward_fn_format(generated_text, golden_answer, task="math"):
    model_answer = extract_answer(generated_text, task)
    accuracy = 1.0 if grade_answer(model_answer, golden_answer) else -0.5 #0.0
    if "boxed" not in generated_text:
        accuracy = -1.0
    return accuracy

def qwen_reward_fn(generated_text, golden_answer, task="math"):
    if isinstance(golden_answer, list):
        golden_answer = golden_answer[0]

    model_answer = extract_answer(generated_text, task)
    accuracy = 1.0 if grade_answer(model_answer, golden_answer) else 0.0 #-0.5 #0.0
    # if "boxed" not in generated_text:
    #     accuracy = -1.0
    return accuracy

def majority_vote(
    solutions: List[str],
    ground_truth: str,
    task="math"
):
    model_answers = [extract_answer(generated_text, task) for generated_text in solutions]
    model_answers = [answer for answer in model_answers if answer is not None]

    if len(model_answers) == 0:
        return 0.0

    counter = Counter(model_answers)
    
    majority_answer, _ = counter.most_common(1)[0]
    accuracy = 1.0 if grade_answer(majority_answer, ground_truth) else 0.0

    return accuracy

def test_time_train(
    solutions: List[str],
    ground_truth: str,
    task="math"):
    model_answers = [extract_answer(generated_text, task) for generated_text in solutions]
    counter = Counter([answer for answer in model_answers if answer is not None])
    
    majority_answer, majority_count = counter.most_common(1)[0]

    # if majority_count / len(solutions) > 0.0 and majority_count > 1:
    rewards = [float(grade_answer(majority_answer, model_answer)) for model_answer in model_answers]
    # else:
    #     rewards = [0.0] * len(solutions)

    assert len(rewards) == len(solutions), f"{len(rewards)} vs {len(solutions)}"
    
    return rewards


def qwen_math_equal_subprocess(prediction, reference, timeout_seconds=10):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(qwen_math_equal, prediction=prediction, reference=reference, timeout=False)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return False

def simplerl_reward_fn(generated_text, golden_answer):
    model_answer = extract_answer(generated_text, "math")
    accuracy = 1.0 if qwen_math_equal_subprocess(prediction=model_answer, reference=golden_answer) else -0.5
    if "boxed" not in generated_text:
        accuracy = -1.0
    return accuracy