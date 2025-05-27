from tqdm import tqdm, trange
from marti.verifiers.qwen.qwen_eval import simplerl_reward_fn, qwen_reward_fn, qwen_reward_fn_format, test_time_train
from marti.verifiers.deepscaler.math_reward import deepscaler_reward_fn #, test_time_train
from marti.verifiers.deepscaler.math_reward import test_time_train as test_time_train_thinking

def auto_verify(task, batch_size, all_outputs, all_labels):
    all_outputs = [output.outputs[0].text for output in all_outputs]

    task2verify = {
        "math": qwen_reward_fn,
        "math_format": qwen_reward_fn_format,
        "simplerl_math": simplerl_reward_fn,
        "think": deepscaler_reward_fn,
        "ttt": test_time_train,
        "ttt_thinking": test_time_train_thinking,
    }
    assert task in task2verify, f"{task} not in {list(task2verify.keys())}"
    
    verify_fn = task2verify[task]
    if task in ["math", "simplerl_math", "think"]:
        rewards = [verify_fn(output, label)
                   for output, label in tqdm(zip(all_outputs, all_labels), desc="Rewarding", total=len(all_outputs))]
    elif task in ["ttt", "ttt_thinking"]:
        rewards = []
        n_prompts = len(all_outputs) // batch_size
        for prompt_idx in range(n_prompts):
            group_outputs = all_outputs[batch_size *
                                        prompt_idx:batch_size*(prompt_idx+1)]
            group_labels = all_labels[batch_size *
                                      prompt_idx:batch_size*(prompt_idx+1)]
            rewards.extend(verify_fn(group_outputs, group_labels))
    return rewards

# https://github.com/huggingface/open-r1/blob/af487204ca09005d12b4d9a48b4162a02e9b6a35/src/open_r1/rewards.py#L268
def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -1.0):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(solutions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            solutions: List of model solutions
        """

        rewards = []
        for completion in solutions:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward
