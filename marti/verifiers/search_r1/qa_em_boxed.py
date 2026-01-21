import re
from marti.verifiers.search_r1.qa_em import count_answer_tags, em_check


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"boxed{\\text{(.*?)}"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em_boxed(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    # do_print = random.randint(1, 64) == 1

    # if do_print:
    #     print("--------------------------------")
    #     print(f"Golden answers: {ground_truth}")
    #     if answer is not None:
    #         print(f"Extracted answer is not None: {answer}")
    #     else:
    #         print("Extracted answer: None!")
    #     print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth):
            if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
                score = score / 4
                return score
            return score
        else:
            return format_score

if __name__ == "__main__":
    print(extract_solution("\nIt seems there was an issue with retrieving the information needed to find the latest champion's address. Let's proceed with the correct information about the latest Scottish Open champion, \"Elsie the Champion,\" who lives in \"Edinburgh, Scotland.\"\n\nTo format the answer:\n\\[\n\\boxed{\\text{Edinburgh, Scotland}}\n\\]<|im_end|>"))
    print(extract_solution("To determine if The New Hotel and Get That Venus are from the same country, we need to identify the countries associated with each entity.\n\n1. **Identify the Country of The New Hotel:**\n   - The New Hotel is a hotel chain. Without specific information about its location, we can't definitively state its country of origin. However, if we assume it has a global presence, we could still consider multiple countries. But for simplicity, let's assume it's in a country that doesn't require a specific mention of its country.\n   \n2. **Identify the Country of Get That Venus:**\n   - \"Get That Venus\" is a song by the British singer-songwriter James Blake. The song's title suggests that it is likely from a country where the Venus symbol is culturally significant, which could be a hint towards the artist's nationality or the cultural context, but not directly the country of origin.\n   \nSince we do not have specific information about the country of The New Hotel, we can't make a definitive comparison. However, if we consider the information we do have, the song \"Get That Venus\" is from the UK, and The New Hotel, being a hotel chain, could be found in various countries. For a definitive answer, we would need more specific information about The New Hotel's location.\n\nLet's assume for the sake of this exercise that The New Hotel is indeed from the UK (as it's a common hotel chain). Then we can compare the countries.\n\nAssuming both entities are from the UK:\n- The New Hotel: UK\n- Get That Venus: UK\n\nSince they are both from the UK, they are from the same country. Let's box the final answer:\n\n\\boxed{\\text{Yes}}<|im_end|>"))