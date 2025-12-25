import math
from dataclasses import dataclass

from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.llm_generation_interface import GenerationRequest, GenerationResult
from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.eval_result import EvalResult
from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.tasks.base import Task
from openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.prompts.base import PromptTemplate
import csv
import json
import ray
import time
import dataclasses
from vllm import SamplingParams
from pathlib import Path
import datetime
import numpy as np
from typing import List, Optional
from openrlhf.utils.logging_utils import init_logger
logger = init_logger(__name__)

@dataclass
class NodeState:
    generation_result: GenerationResult
    eval_results: EvalResult

def is_power_of_two(n: int):
    return n > 0 and (n & (n - 1)) == 0

def get_private_score(task: Task, node_state: NodeState | None) -> float:
    if node_state is not None:
        eval_results, _score = task.evaluate_on_test(
            llm_answer=node_state.generation_result
        )
        if len(eval_results) == 0:
            private_score = 0
        else:
            private_score = sum(
                [eval_result.get_score() for eval_result in eval_results]
            ) / len(eval_results)
    else:
        private_score = 0
    return private_score

def process_node(node, task):
    """Helper function to process a single node and calculate scores."""
    if node.expand_idx < 0:
        return None

    node_idx = node.expand_idx
    public_score = node.score

    # calc private score
    node_state = node.state
    private_score = get_private_score(task, node_state)
    # private_score = public_score

    return node_idx, public_score, private_score

def apply_template_with_tokenizer(tokenizer, prompt, enable_thinking=False):
    if isinstance(prompt, str):
        message = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        message = prompt
    return tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

async def generate_fn_async(
    state: NodeState | None,
    action: str,
    task: Task,
    prompt_template: PromptTemplate,
    # model_name: str,    
    llm,
    sampling_params: dict,
    llm_log_dir: Path,
    tokenizer: None,
    enable_thinking: bool=False,
) -> tuple[NodeState, float]:
    start_time = time.time()

    # From root
    if state is None:
        messages = [{"role": "user", "content": prompt_template.initial_prompt()}]
    else:
        feedback_prompt = prompt_template.feedback_prompt(
            "transform",
            eval_results=state.eval_results,
            generation_result=state.generation_result,
        )
        messages = [
            {"role": "user", "content": feedback_prompt},
        ]
    
    messages = apply_template_with_tokenizer(tokenizer, messages, enable_thinking)

    assert isinstance(messages, str), f"messages must be str"
    assert isinstance(sampling_params, SamplingParams)

    # Tokenize the initial observation
    current_obs_tokens = tokenizer(messages, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ][0].tolist()

    if sampling_params.logprobs is not None:
        rollout_log_probs = [0.0] * len(current_obs_tokens)
    else:
        rollout_log_probs = None

    request_output = await llm.generate_async.remote(
        prompt_ids=current_obs_tokens,
        sampling_params=sampling_params,
    )
    action_tokens = request_output.outputs[0].token_ids
    action_text = request_output.outputs[0].text

    current_obs_tokens = (
        current_obs_tokens
        + action_tokens
    )

    # get rollout logprobs for enable_vllm_is_correct
    if sampling_params.logprobs is not None:
        # action tokens logprobs
        for i, logprob in enumerate(request_output.outputs[0].logprobs):
            rollout_log_probs.append(logprob[action_tokens[i]].logprob)
        # dummy logprobs for the env feedback tokens
        rollout_log_probs.extend([0.0] * (len(current_obs_tokens) - len(rollout_log_probs)))
    #logger.warning(f"the rollout_log_probs is: {rollout_log_probs}")
    result = GenerationResult(
        request=GenerationRequest(messages=messages), 
        action_text=action_text, 
        action_tokens=action_tokens, 
        rollout_log_probs=rollout_log_probs,
        agent_id=action,
        sequence_ids=current_obs_tokens
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
        :-3
    ]  # up to milliseconds

    log_txt = llm_log_dir / f"log_{timestamp}_{action}.txt"
    log_txt.write_text(
        json.dumps(
            {"agent_id": action, "result": dataclasses.asdict(result)},
            indent=4,
        )
    )  # save cost and result

    eval_results = task.generate_eval_results(llm_answer=result, kind="transform")
    if eval_results is None:
        score = 0.0
    else:
        score = sum([eval_result.get_score() for eval_result in eval_results]) / len(
            eval_results
        )

    return NodeState(
        generation_result=result, eval_results=eval_results
    ), score

def generate_fn(
    state: NodeState | None,
    task: Task,
    prompt_template: PromptTemplate,
    # model_name: str,    
    llm,
    sampling_params: dict,
    llm_log_dir: Path,
    tokenizer: None,
) -> tuple[NodeState, float]:
    # global total_cost, cost_by_model, time_by_model, node_times

    start_time = time.time()

    # From root
    if state is None:
        messages = [{"role": "user", "content": prompt_template.initial_prompt()}]
    else:
        feedback_prompt = prompt_template.feedback_prompt(
            "transform",
            eval_results=state.eval_results,
            generation_result=state.generation_result,
        )
        messages = [
            {"role": "user", "content": feedback_prompt},
        ]
    
    messages = apply_template_with_tokenizer(tokenizer, messages)



    # assert prompt_token_ids is not None
    assert isinstance(messages, str), f"messages must be str"
    assert isinstance(sampling_params, SamplingParams)
    answer = ray.get(llm.generate.remote(
        prompts=messages,
        sampling_params=sampling_params,
    ))

    generation = answer.outputs[0].text

    result = GenerationResult(
        request=GenerationRequest(messages=messages), generation=generation, rollout_log_probs=rollout_log_probs,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
        :-3
    ]  # up to milliseconds

    log_txt = llm_log_dir / f"log_{timestamp}_{action}.txt"
    log_txt.write_text(
        json.dumps(
            {"agent_id": action, "result": dataclasses.asdict(result)},
            indent=4,
        )
    )  # save cost and result

    eval_results = task.generate_eval_results(llm_answer=result, kind="transform")
    if eval_results is None:
        score = 0.0
    else:
        score = sum([eval_result.get_score() for eval_result in eval_results]) / len(
            eval_results
        )
    return NodeState(
        generation_result=result, eval_results=eval_results
    ), score

def get_coverage_and_passk(node_list, problem, workflow_args, checkpoint_path=None, prompt_id=0) -> List[float]:
    n_jobs = workflow_args.get("eval_n_jobs", 8)
    # n_jobs = min(8, os.cpu_count())
    top_k = workflow_args.get("top_k", 1)
    proc_ret_path = Path(checkpoint_path.as_posix().replace(".pkl", "_proc_result.json"))
    # Parallel processing of nodes
    # results = Parallel(n_jobs=n_jobs, prefer="processes")(
    #     delayed(process_node)(node, code_problem)
    #     for node in tqdm(valid_nodes, desc=f"Processing {prompt_id}")
    # )
    results = [
        process_node(node, problem)
        for node in node_list
    ]

    # Filter out None results (from nodes with expand_idx < 0, though already filtered)
    results = [r for r in results if r is not None]

    # Sort results by node_idx (the first element of each tuple in results)
    # This ensures that node_idx_list, public_scores, and private_scores are ordered by node_idx
    results.sort(key=lambda x: x[0])

    # Unpack sorted results
    node_idx_list = []
    public_scores = []
    private_scores = []
    for result in results:
        node_idx, public_score, private_score = (
            result  # No need to check for None here, already filtered
        )
        node_idx_list.append(node_idx)
        public_scores.append(public_score)
        private_scores.append(private_score)
    assert len(node_idx_list) == len(public_scores) == len(private_scores)
    # assert public_scores != private_scores, "public_scores and private_scores should not be the same"
    passkk_reward = 0
    for node_id in range(len(node_idx_list)):
        passkk_reward = max(private_scores[node_id], passkk_reward)

    # transform NumPy array
    public_arr = np.array(public_scores)
    private_arr = np.array(private_scores)

    # get top_k index_list 
    # topk_idx = np.argsort(-public_arr)[:top_k]
    # # np.lexsort((-idx, -window_reward), axis=0)
    # logger.info(f"Prompt {prompt_id} - Top-{top_k} indices: {topk_idx.tolist()}, public_scores: {public_arr[topk_idx].tolist()}")

    # get top_k index_list 
    # topk_idx = np.argsort(-public_arr)[:top_k]
    idx = np.arange(len(public_arr))
    # sort_idx = np.lexsort((-public_arr, -idx))
    topk_idx = np.lexsort((-idx, -public_arr))[:top_k]
    logger.info(f"Prompt {prompt_id} - Top-{top_k} indices: {topk_idx.tolist()}, public_scores: {public_arr[topk_idx].tolist()}")

    # select private score
    selected_private = private_arr[topk_idx]

    # get best pass@k
    # best_final_reward = selected_private.max()
    pass1_reward = selected_private.max()
    
    # avg_private_score: public_score == 1, private_score is average value
    perfect_public_mask = public_arr == 1.0
    if np.any(perfect_public_mask):
        avg_private_score = np.mean(private_arr[perfect_public_mask])
    else:
        avg_private_score = pass1_reward


    # save node scores info 
    proc_ret = {
        "node_idx_list": node_idx_list,
        "public_scores": public_scores,
        "private_scores": private_scores,
    }
    with open(proc_ret_path, "w") as f:
        json.dump(proc_ret, f)
    
    # Save CSV files for public and private scores to top-level directory
    # checkpoint_path structure: workflow_save_path / prompt_id / checkpoints / checkpoint_latest.pkl
    # We want to save CSVs to: workflow_save_path (go up 3 levels from checkpoint file)
    if checkpoint_path:
        csv_dir = checkpoint_path.parent.parent.parent  # Go from checkpoint_latest.pkl -> checkpoints -> prompt_id -> workflow_save_path
    else:
        csv_dir = Path(".")
    public_csv_path = csv_dir / "all_public_scores.csv"
    private_csv_path = csv_dir / "all_private_scores.csv"
    
    # Read existing data to avoid duplicates
    existing_public = set()
    existing_private = set()
    
    if public_csv_path.exists():
        with open(public_csv_path, "r", newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    existing_public.add((int(row[0]), int(row[1])))  # (prompt_id, node_idx)
    
    if private_csv_path.exists():
        with open(private_csv_path, "r", newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    existing_private.add((int(row[0]), int(row[1])))
    
    # Prepare data to write (only new entries)
    new_public_data = []
    new_private_data = []
    
    for node_idx, public_score, private_score in zip(node_idx_list, public_scores, private_scores):
        key = (prompt_id, node_idx)
        if key not in existing_public:
            new_public_data.append([prompt_id, node_idx, public_score])
        if key not in existing_private:
            new_private_data.append([prompt_id, node_idx, private_score])
    
    # Write new data
    write_public_header = not public_csv_path.exists()
    write_private_header = not private_csv_path.exists()
    
    if new_public_data:
        with open(public_csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            if write_public_header:
                writer.writerow(["prompt_id", "node_idx", "public_score"])
            writer.writerows(new_public_data)
        logger.info(f"Prompt {prompt_id} - Added {len(new_public_data)} new entries to public scores CSV")
    else:
        logger.info(f"Prompt {prompt_id} - No new public scores to add (all already exist)")
    
    if new_private_data:
        with open(private_csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            if write_private_header:
                writer.writerow(["prompt_id", "node_idx", "private_score"])
            writer.writerows(new_private_data)
        logger.info(f"Prompt {prompt_id} - Added {len(new_private_data)} new entries to private scores CSV")
    else:
        logger.info(f"Prompt {prompt_id} - No new private scores to add (all already exist)")
    
    logger.info(f"Prompt {prompt_id} - Saved {len(node_idx_list)} nodes to CSV files: {public_csv_path} and {private_csv_path}")
    
    # logger.info(f"the coverage_final_reward of {prompt_id}-th prompt: {coverage_final_reward}")
    # logger.info(f"the best_final_reward of {prompt_id}-th prompt: {best_final_reward}")
    return [pass1_reward, passkk_reward, avg_private_score]
