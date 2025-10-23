from func_timeout import func_timeout, FunctionTimedOut
from marti.verifiers.qwen.qwen_math_parser import extract_answer
from marti.verifiers.qwen.math_grade import grade_answer

def qwen_reward_fn_timeout(
    generated_text: str,
    golden_answer: str,
    task: str = "math",
    timeout: float = 10.0,
) -> float:
    model_answer = extract_answer(generated_text, task)
    try:
        ok = func_timeout(timeout, grade_answer, args=(model_answer, golden_answer))
        return 1.0 if ok else 0.0
    except FunctionTimedOut:
        return -1.0