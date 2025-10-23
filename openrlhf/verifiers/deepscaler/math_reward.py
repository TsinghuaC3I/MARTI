"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from collections import defaultdict
from collections import Counter
from typing import List, Union
import srsly
import numpy as np
import pandas as pd

from marti.verifiers.deepscaler.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, OAI_RM_MODEL
from marti.verifiers.deepscaler.reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from marti.verifiers.deepscaler.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd


class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(
                input.problem_type)

        problem = input.problem
        model_response = input.model_response

        # Extract solution.
        if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        model_answer = extract_answer(model_solution)
        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]

        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)

        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(
                model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                return RewardOutput(reward=self.config.correct_reward, is_correct=True)

        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

def get_majority_vote_answer(solutions: List[str], inputs=None, config=None, return_extracted=False):
    if inputs is None:
        inputs = [RewardInput(problem=solution, problem_type=RewardType.MATH, model_response=solution, ground_truth={"answer": "-1"})
                for solution in solutions]
    if config is None:
        config = RewardConfig()

    answers = []

    extracted_answers = []
    for inp in inputs:
        assert inp.problem_type == RewardType.MATH, (
            f"Invalid problem type: expected 'MATH', but got '{inp.problem_type}'"
        )

        model_response = inp.model_response
        if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            extracted_answers.append("None")
            continue

        model_answer = extract_answer(model_solution)
        if model_answer is None:
            extracted_answers.append("None")
            continue

        answers.append(model_answer)
        extracted_answers.append(model_answer)

    if not answers:
        return RewardOutput(
            reward=config.format_error_reward,
            is_correct=False
        )

    counter = Counter(answers)
    majority_answer, _ = counter.most_common(1)[0]
    if return_extracted:
        assert len(extracted_answers) == len(solutions), f"Number of answers is not equal to solutions, {len(extracted_answers)} vs {solutions}"
        return majority_answer, {
            "all_answers": extracted_answers,
            "all_rewards": [float(is_equal(majority_answer, answer)) for answer in extracted_answers]
        }
    else:
        return majority_answer

def is_equal(solution, ground_truth):
    is_correct = (
            grade_answer_mathd(solution, ground_truth) or
            grade_answer_sympy(solution, ground_truth)
        )
    return is_correct

def test_time_train(solutions, ground_truth=None):
    for solution_str in solutions:
        if "</think>" in solution_str and not solution_str.startswith("<think>"):
            solution_str = "<think>" + solution_str

    # we add <think> in the prompt, which is not in solution
    _, answers_and_rewards = get_majority_vote_answer(solutions=solutions, return_extracted=True)
    return answers_and_rewards["all_rewards"]

def majority_vote(
    solutions: List[str],
    ground_truth: str,
) -> RewardOutput:
    inputs = [RewardInput(problem=solution, problem_type=RewardType.MATH, model_response=solution, ground_truth={"answer": ground_truth})
                for solution in solutions]
    config = RewardConfig()

    majority_answer = get_majority_vote_answer(solutions=solutions, inputs=inputs, config=config)

    ground_truths = inputs[0].ground_truth.get("answer", None)
    if ground_truths is None:
        return RewardOutput(
            reward=config.unk_error_reward,
            is_correct=False
        )
    if isinstance(ground_truths, (str, float, int)):
        ground_truths = [ground_truths]

    processed_ground_truths = []
    for truth in ground_truths:
        truth_str = str(truth)
        if "\\boxed" in truth_str:
            processed_truth = extract_answer(truth_str)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth_str)

    if not processed_ground_truths:
        return RewardOutput(
            reward=config.unk_error_reward,
            is_correct=False
        )

    for ground_truth in processed_ground_truths:
        is_correct = (
            grade_answer_mathd(majority_answer, ground_truth) or
            grade_answer_sympy(majority_answer, ground_truth)
        )
        if is_correct:
            return RewardOutput(
                reward=config.correct_reward,
                is_correct=True
            )

    return RewardOutput(
        reward=config.incorrect_reward,
        is_correct=False
    )

def deepscaler_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm=False):
    # we add <think> in the prompt, which is not in solution
    if "</think>" in solution_str and not solution_str.startswith("<think>"):
        solution_str = "<think>" + solution_str

    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH,
                                model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response.is_correct


def test_case():
    reward = RewardMathFn(RewardConfig)
    input = RewardInput(problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.",
                        problem_type=RewardType.MATH, model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]})
    output = reward(input)
    print(output)
