"""
Example workflow: MathChat-style multi-agent interaction
Pattern: Generator -> Coder -> Refiner -> Coder -> Refiner -> ...
"""

import os
import random
from typing import Dict, List, Any, Optional
import json
import asyncio
from marti.utils.logging_utils import init_logger
import ray
import sys
import time
import dataclasses
from vllm import SamplingParams

from marti.agent_workflows.third_party.mcts_utils.ab_mcts.llm_generation_interface import GenerationRequest, GenerationResult
from marti.agent_workflows.third_party.mcts_utils.ab_mcts.prompts.base import PromptTemplate
from marti.agent_workflows.third_party.mcts_utils.code_prompt import CodePrompt
from marti.agent_workflows.third_party.mcts_utils.math_prompt import MathPrompt
from marti.agent_workflows.third_party.mcts_utils.utils import NodeState, is_power_of_two, get_private_score, process_node, generate_fn, generate_fn_async, get_coverage_and_passk
from marti.agent_workflows.third_party.mcts_utils.ab_mcts.prompts.prompt_configs import PromptConfig
from marti.agent_workflows.third_party.mcts_utils.ab_mcts.tasks.code.task import CodeProblem
from marti.agent_workflows.third_party.mcts_utils.ab_mcts.tasks.math.task import MathProblem
from marti.trainer.ppo_utils.multi_agent_experience_maker import MultiAgentExperience
from marti.trainer.ppo_utils.experience_maker import Experience
from pathlib import Path
import treequest as tq
import pickle
from tqdm import tqdm
import json
from functools import partial
import importlib
from joblib import Parallel, delayed
import numpy as np

logger = init_logger(__name__)

sys.setrecursionlimit(
    20000
)  # Example: Increase limit to 20000.  Choose a sensible value.

use_local_llm_generate = False

async def workflow(
    workflow_args, 
    prompt: str,
    label: str,
    agents: List[Dict[str, Any]],
    metadata: str = None,
    is_eval=False,
    prompt_id=0, 
    task: str="code",
    **kwargs,
):
    eval_only = os.environ.get("eval_only", "0") == "1"
    start_time = time.time()
    if task == "code":
        problem_cls = CodeProblem
        prompt_cls = CodePrompt
        if is_eval:
            # print(f"Running in evaluation mode for {prompt_id}-th prompt")
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            assert isinstance(metadata, dict), f"metadata must be dict, but got {type(metadata)}"
            assert "public_test_cases" in metadata and "private_test_cases" in metadata, f"Evaluation requires public and private tests in metadata. \nmetadata: {metadata}"
            label = {
                "public_tests": metadata["public_test_cases"],
                "private_tests": metadata["private_test_cases"],
            }
            
    elif task == "math":
        problem_cls = MathProblem
        prompt_cls = MathPrompt

    code_problem = problem_cls.load_data(prompt, label, metadata)
    prompt_template = prompt_cls(prompt_config=PromptConfig(), problem=code_problem)

    assert isinstance(agents, list), f"agents must be list"
    for idx, agent in enumerate(agents):
        if not isinstance(agent, dict):
            agents[idx] = json.loads(agent)
    assert isinstance(agents[0], dict), f"element of agents must be dict: {agent}"


    # Algo
    algo_config = workflow_args["algo"]

    try:
        # sync generate_fn & algo class
        algo_cls = getattr(tq, algo_config["class_name"])
        gen_fn = generate_fn
        is_async = False
    except AttributeError:
        # async generate_fn & algo class
        module_name = algo_config.get("module", "openrlhf.agent_workflows.third_party.mcts_utils.ab_mcts.algo.AsyncABMCTSA")
        class_name = algo_config["class_name"]
        is_async = True
        module = importlib.import_module(module_name)
        algo_cls = getattr(module, class_name)
        gen_fn = generate_fn_async
    algo: tq.Algorithm = algo_cls(**algo_config["params"])
    
    save_dir = workflow_args["save_path"]
    save_dir = Path(save_dir)
    save_dir = save_dir / f"{prompt_id}"
    logger.info(f"the final save dir is {save_dir}")
    for subdir in ["llm_logs", "costs", "checkpoints"]:
        if not (save_dir / subdir).exists():
            (save_dir / subdir).mkdir(parents=True, exist_ok=True)

    llm_log_dir = save_dir / "llm_logs"
    checkpoint_path = save_dir / "checkpoints" / "checkpoint_latest.pkl"

    generate_fns = {
        agent["agent_id"]: partial(
            gen_fn,
            task=code_problem,
            llm=agent['llm'],
            sampling_params=agent['sampling_params'],
            llm_log_dir=llm_log_dir,
            prompt_template=prompt_template,
            tokenizer=agent["tokenizer"],
            enable_thinking=agent["enable_thinking"]
        )
        for agent in agents
    }
    agents_dict = {agent["agent_id"]: agent for agent in agents}
    
    # ==================== Checkpoint Loading Logic ====================
    # In eval mode, try to load checkpoint for resume capability
    checkpoint_loaded = False
    if eval_only and is_eval and checkpoint_path.exists():
        try:
            with open(checkpoint_path, "rb") as f:
                search_tree = pickle.load(f)
            checkpoint_loaded = True
            loaded_nodes = len(algo.get_state_score_pairs(search_tree))
            logger.info(f"✅ Successfully loaded checkpoint with {loaded_nodes} existing nodes")
        except Exception as e:
            logger.warning(f"❌ Failed to load checkpoint: {e}")
            search_tree = algo.init_tree()
    else:
        if eval_only and is_eval:
            logger.info("No checkpoint found, starting fresh evaluation")
        search_tree = algo.init_tree()

    # Calculate initial nodes and target nodes
    initial_num_nodes = len(algo.get_state_score_pairs(search_tree))
    max_num_nodes = workflow_args["max_num_nodes"] if not is_eval else workflow_args["eval_max_num_nodes"]
    nodes_to_expand = max_num_nodes - initial_num_nodes

    logger.info(f"📊 Node Statistics:")
    logger.info(f"  - Initial nodes: {initial_num_nodes}")
    logger.info(f"  - Target nodes: {max_num_nodes}")
    logger.info(f"  - Nodes to expand: {nodes_to_expand}")
    
    if nodes_to_expand <= 0:
        logger.warning(f"⚠️  Already reached target nodes ({initial_num_nodes} >= {max_num_nodes}), skipping expansion")

    # ==================== MCTS Search Loop with Periodic Saving ====================
    # Get checkpoint save interval from config (default: every 2 nodes)
    checkpoint_interval = workflow_args.get("checkpoint_save_interval", 2)
    for i in tqdm(range(nodes_to_expand), desc=f"MCTS Process (Prompt {prompt_id})"):
        node_start_time = time.time()
        if is_async:
            search_tree = await algo.step(search_tree, generate_fns)
        else:
            search_tree = algo.step(search_tree, generate_fns)

        current_nodes = len(algo.get_state_score_pairs(search_tree))
        
        # Periodic checkpoint saving in eval mode
        if eval_only and is_eval and (i + 1) % checkpoint_interval == 0:
            try:
                temp_checkpoint_path = save_dir / "checkpoints" / f"checkpoint_temp.pkl"
                with open(temp_checkpoint_path, "wb") as f:
                    pickle.dump(search_tree, f)
                # Atomic rename to avoid corrupted checkpoints
                temp_checkpoint_path.rename(checkpoint_path)
                logger.info(f"💾 Checkpoint saved at {current_nodes} nodes (iteration {i+1}/{nodes_to_expand})")
            except Exception as e:
                logger.warning(f"Failed to save periodic checkpoint: {e}")

    # ==================== Final Checkpoint Save ====================
    if is_eval:
        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(search_tree, f)
            final_nodes = len(algo.get_state_score_pairs(search_tree))
            logger.info(f"✅ Final checkpoint saved with {final_nodes} nodes")
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")

    # node：Node(state=state, score=score, parent=parent, expand_idx=self.size - 1), 
    # state: NodeState(generation_result=result, eval_results=eval_results, model_name=model_name)
    valid_nodes = [
        node for node in search_tree.tree.get_nodes() if node.expand_idx >= 0
    ]
    if is_eval:
        final_reward = get_coverage_and_passk(valid_nodes, code_problem, workflow_args, checkpoint_path, prompt_id)
    else:
        final_reward = [0, 0, 0]
    # valid_nodes.sort(key=lambda node: node.state.model_name)

    node_records = []
    for node_id, node in enumerate(valid_nodes):
        node_state = node.state
        # agent_name = node_state.model_name

        # 获取生成内容
        gen_result: GenerationResult = node_state.generation_result
        #logger.warning(f"the gen_result is: {gen_result}")
        rollout_log_prob = getattr(gen_result, "rollout_log_probs", None)
        agent_id = getattr(gen_result, "agent_id", None)
        # print(f"the agent id is: {agent_id} with type {type(agent_id)}")
        node_record = {
            "turn_id": node.expand_idx,
            "node_id": node_id,
            "agent_id": agent_id,
            "agent_role": agents_dict[agent_id].get("agent_role", "unknown"),
            "agent_input": gen_result.request.messages,
            "agent_output": gen_result.action_text,
            "output_ids": gen_result.action_tokens,
            "sequence_ids": gen_result.sequence_ids,
            "reward": node.score,
            "rollout_log_prob": rollout_log_prob,
            "metadata": {
                "expand_idx": node.expand_idx,
                # "parent": getattr(node.parent, "expand_idx", None),
                # "depth": getattr(node, "depth", None),
                "timestamp": time.time(),
            },
        }
        node_records.append(node_record)

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": node_records,
        "final_reward": final_reward,
    }