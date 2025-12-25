import os
import argparse
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

import ray
from ray.util.placement_group import placement_group

from openrlhf.trainer.ray import create_vllm_engines
from openrlhf.trainer.ray.launcher import (
    RayActorGroup,
    ReferenceModelActor,
    RewardModelActor,
)
from openrlhf.trainer.ray.ppo_actor import PolicyModelActor
from openrlhf.trainer.ray.ppo_critic import CriticModelActor
from openrlhf.utils import get_strategy
import json

from openrlhf.datasets.utils import blending_datasets
from openrlhf.datasets import PromptDataset
from openrlhf.utils.utils import get_tokenizer

def get_seed(base_seed, num_workers=4):
    """    
    Args:
        base_seed: 基础种子值
        num_workers: 工作进程数量
    Returns:
        新的种子值
    """
    return base_seed + num_workers * 10


def calculate_max_steps(args, strategy):
    """提前计算训练的max_steps
    Args:
        args: 训练参数
        strategy: DeepSpeed策略
    Returns:
        max_steps: 最大训练步数
    """   
    train_data = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        dataset_split=args.prompt_split,
    )
    
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    
    first_pretrain = args.agents[0][list(args.agents[0].keys())[0]]["pretrain"]
    tokenizer = get_tokenizer(first_pretrain, None, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    
    prompts_dataset = PromptDataset(train_data, tokenizer, strategy, input_template=args.input_template)
    
    max_steps = (
        len(prompts_dataset)
        * args.n_samples_per_prompt
        // args.train_batch_size
        * args.num_episodes
        * args.max_epochs
    )
    return max_steps

def init_agents_sequential(agent_configs, args, strategy, unified_pg_list, global_seed, max_steps):
    agent_list = []
    for i, (agent_id, agent_config) in enumerate(agent_configs):
        # We copy the first agent to others under shared_agents settings
        if args.shared_agents and i > 0:
            agent = agent_list[0]
        else:
            agent = init_agent(
                agent_id=agent_id,
                agent_config=agent_config,
                global_config=args,
                strategy=strategy,
                pg_id=i,
                unified_pg_list=unified_pg_list,
                global_seed=global_seed,
                max_steps=max_steps
            )
        agent_list.append(agent)
    return agent_list

def init_agents_parallel(agent_configs, args, strategy, unified_pg_list, global_seed, max_steps):
    import concurrent.futures
    print(f"Initializing {len(agent_configs)} agents in parallel...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(agent_configs)) as executor:
        futures = [
            executor.submit(
                init_agent,
                agent_id=agent_id,
                agent_config=agent_config,
                global_config=args,
                strategy=strategy,
                pg_id=idx,
                unified_pg_list=unified_pg_list,
                global_seed=global_seed,
                max_steps=max_steps
            )
            for idx, (agent_id, agent_config) in enumerate(agent_configs)
        ]

        try:
            agents = [f.result() for f in futures]
        except Exception as e:
            print(f"Failed to initialize agents in parallel: {e}")
            print("Falling back to sequential initialization...")
            return init_agents_sequential(agent_configs, args, strategy, unified_pg_list, global_seed, max_steps)
        if args.shared_agents:
            first_agent = agents[0]
            agents = [first_agent] * len(agent_configs)
        print(f"Successfully initialized {len(agents)} agents in parallel")
        return agents 

def train(args):
    # initialize ray if not initialized
    if not ray.is_initialized():
        import logging
        dashboard_host="0.0.0.0"
        dashboard_port=8265
        ray.init(runtime_env={
            "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            logging_level=logging.DEBUG,
            logging_format="%(asctime)s %(levelname)s %(message)s",
            dashboard_host="0.0.0.0",
            dashboard_port=8265,
            include_dashboard=True, 
            )

    # Normalize global vllm_num_engines: convert None to 0
    if args.vllm_num_engines is None:
        args.vllm_num_engines = 0

    # configure strategy
    strategy = get_strategy(args)
    strategy.print(args)

    # Record global seed and set different seed for each agent
    global_seed = args.seed

    # Prepare config for each agent
    agent_configs = []
    for agent_dict in args.agents:
        for agent_id, agent_config in agent_dict.items():
            agent_configs.append([agent_id, agent_config])

    # prepare placement_group for agents
    unified_pg_list = []
    for agent_id, agent_config in agent_configs:
        # Normalize vllm_num_engines: convert None to 0
        if agent_config.get("vllm_num_engines") is None:
            agent_config["vllm_num_engines"] = 0

        if agent_config.get("is_tuning", False):
            # if colocated, create placement group for actor and ref model explicitly
            pg = None
            if agent_config.get("colocate_actor_ref", False) or agent_config.get("colocate_all_models", False):
                assert (
                    agent_config["actor_num_nodes"] == agent_config["ref_num_nodes"] 
                    and agent_config["actor_num_gpus_per_node"] == agent_config["ref_num_gpus_per_node"]
                ), "num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

                bundles = [{"GPU": 1, "CPU": 1} for _ in range(
                    agent_config["actor_num_nodes"] * agent_config["actor_num_gpus_per_node"])]
                pg = placement_group(bundles, strategy="PACK")
                ray.get(pg.ready())
            unified_pg_list.append(pg)

    # init actor&critic scheduler
    strategy.print("Calculating max_steps for training...")
    max_steps = calculate_max_steps(args, strategy)
    strategy.print(f"Calculated max_steps: {max_steps}")

    # init multi agent
    agent_list = init_agents_parallel(agent_configs, args, strategy, unified_pg_list, global_seed, max_steps) if args.parallel_loading else init_agents_sequential(agent_configs, args, strategy, unified_pg_list, global_seed, max_steps)
    num_agents = len(agent_list)

    # Training loop using MultiAgent_PPOTrainer
    from openrlhf.trainer.multi_agent_ppotrainer import MultiAgent_PPOTrainer

    # Collect pretrain paths for all agents
    agents_pretrain = []
    for agent_dict in args.agents:
        for agent_id, agent_config in agent_dict.items():
            agents_pretrain.append(agent_config["pretrain"])
    
    # Add agents_pretrain to args for MultiAgent_PPOTrainer to use
    args.agents_pretrain = agents_pretrain
    
    # Prepare generate kwargs
    generate_kwargs = {
        "do_sample": True,
        "max_new_tokens": args.generate_max_len,
        "max_length": args.max_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    
    ppo_trainer = MultiAgent_PPOTrainer.remote(
        args.agents_pretrain[0],
        strategy,
        agent_list,  # Pass the entire agent list
        prompt_max_len=args.prompt_max_len,
        dataloader_pin_memory=True,  # Explicitly set default value
        prompt_split=args.prompt_split,
        eval_split=args.eval_split,
        **generate_kwargs,
    )

    # Train all agents together
    ray.get(ppo_trainer.fit.remote())

    # save model
    for agent in agent_list:
        agent.save_actor_and_critic_model()
        # ray.get(agent.actor_model_group.async_save_model())
        # if args.critic_pretrain and args.save_value_network:
        #     ray.get(agent.critic_model_group.async_save_model())

def init_agent(agent_id, agent_config, global_config, strategy, pg_id, unified_pg_list, global_seed, max_steps):
    print("Create agent for", agent_id, agent_config)

    print(agent_id, agent_config)
    # Create tokenizer
    tokenizer = get_tokenizer(
        agent_config["pretrain"], None, "left", strategy, use_fast=not agent_config["disable_fast_tokenizer"])

    max_len = agent_config["max_len"] if agent_config["max_len"] else agent_config["prompt_max_len"] + \
        max(agent_config["generate_max_len"], agent_config["eval_generate_max_len"])
    generate_kwargs = {
        "do_sample": True,
        "max_new_tokens": agent_config["generate_max_len"],
        "eval_max_new_tokens": agent_config["eval_generate_max_len"],
        "max_length": max_len,
        "temperature": agent_config["temperature"],
        "eval_temperature": agent_config["eval_temperature"] if agent_config["eval_temperature"] is not None else agent_config["temperature"],
        "top_p": agent_config["top_p"],
        "top_k": agent_config.get("top_k", -1),
        # "pad_token_id": tokenizer.pad_token_id,
        # "eos_token_id": tokenizer.eos_token_id,
    }

    # Initialize pg to None first
    pg = None
    if agent_config["is_tuning"]:
        pg = unified_pg_list[pg_id]

    vllm_engines = None
    if agent_config["is_tuning"] or (not agent_config["is_tuning"] and agent_config["pretrain"] is not None):
        # init vLLM engine for text generation
        if agent_config["vllm_num_engines"] is not None and agent_config["vllm_num_engines"] > 0:
            if agent_config["colocate_all_models"]:
                assert (
                    agent_config["actor_num_nodes"] * agent_config["actor_num_gpus_per_node"]
                    == agent_config["vllm_num_engines"] * agent_config["vllm_tensor_parallel_size"]
                ), (
                    f"actor_num_nodes * actor_num_gpus_per_node must be equal to "
                    f"vllm_num_engines * vllm_tensor_parallel_size, got {agent_config['actor_num_nodes'] * agent_config['actor_num_gpus_per_node']} "
                    f"and {agent_config['vllm_num_engines'] * agent_config['vllm_tensor_parallel_size']}"
                )

            if global_config.workflow_func_path:
                from openrlhf.trainer.ray.vllm_engine_async import LLMRayActorAsync as LLMRayActor
            else:
                from openrlhf.trainer.ray.vllm_engine import LLMRayActor

            vllm_engines = create_vllm_engines(
                agent_config["vllm_num_engines"],
                agent_config["vllm_tensor_parallel_size"],
                agent_config["pretrain"],
                get_seed(global_seed, agent_config["vllm_num_engines"]),
                agent_config.get("full_determinism", False),
                agent_config["enable_prefix_caching"],
                agent_config["enforce_eager"],
                max_len,
                pg if agent_config["colocate_all_models"] else None,
                agent_config["vllm_gpu_memory_utilization"],
                agent_config["vllm_enable_sleep"],
                LLMRayActor,
                "processed_logprobs" if args.enable_vllm_is_correction else None,
                # agent_config.get("logprobs_mode", None),
                agent_config.get("agent_func_path", None),
            )

    actor_model = None
    critic_model = None
    ref_model = None
    reward_models = None

    if agent_config["is_tuning"]:
        actor_model = RayActorGroup(
            agent_config["actor_num_nodes"],
            agent_config["actor_num_gpus_per_node"],
            PolicyModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            # duplicate_actors=agent_config["ring_attn_size"] * agent_config["ds_tensor_parallel_size"],
        )

        ref_model = RayActorGroup(
            agent_config["ref_num_nodes"],
            agent_config["ref_num_gpus_per_node"],
            ReferenceModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            # duplicate_actors=agent_config["ring_attn_size"] * agent_config["ds_tensor_parallel_size"],
        )

        if not agent_config["colocate_all_models"]:
            pg = None

        if agent_config["critic_pretrain"] and agent_config["colocate_critic_reward"]:
            assert (
                agent_config["critic_num_nodes"] == agent_config["reward_num_nodes"]
                and agent_config["critic_num_gpus_per_node"] == agent_config["reward_num_gpus_per_node"]
            ), "num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

            bundles = [{"GPU": 1, "CPU": 1} for _ in range(
                agent_config["critic_num_nodes"] * agent_config["critic_num_gpus_per_node"])]
            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())

        if agent_config["critic_pretrain"]:
            critic_model = RayActorGroup(
                agent_config["critic_num_nodes"],
                agent_config["critic_num_gpus_per_node"],
                CriticModelActor,
                pg=pg,
                num_gpus_per_actor=0.2 if pg else 1,
                # duplicate_actors=agent_config["ring_attn_size"] * agent_config["ds_tensor_parallel_size"],
            )

        # multiple reward models
        if agent_config["reward_pretrain"] is not None:
            reward_pretrains = agent_config["reward_pretrain"].split(",")
            reward_models = []
            for _ in reward_pretrains:
                reward_models.append(
                    RayActorGroup(
                        agent_config["reward_num_nodes"],
                        agent_config["reward_num_gpus_per_node"],
                        RewardModelActor,
                        pg=pg,
                        num_gpus_per_actor=0.2 if pg else 1,
                        # duplicate_actors=agent_config["ring_attn_size"] * agent_config["ds_tensor_parallel_size"],
                    )
                )

        if ref_model is not None:
            # init reference/reward/actor model
            refs = ref_model.async_init_model_from_pretrained(strategy, agent_config["pretrain"])
            ray.get(refs)

        refs = actor_model.async_init_model_from_pretrained(
            strategy, agent_config["pretrain"], max_steps, vllm_engines, actor_ckpt_path=agent_config["ckpt_path"])
        ray.get(refs)

        if agent_config["reward_pretrain"] is not None:
            for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
                refs = reward_model.async_init_model_from_pretrained(
                    strategy, reward_pretrain)
            ray.get(refs)

        if agent_config["critic_pretrain"]:
            # critic scheduler initialization depends on max_step, so we have to init critic after actor
            refs = critic_model.async_init_model_from_pretrained(
                strategy, agent_config["critic_pretrain"], max_steps, critic_ckpt_path=agent_config["ckpt_path"])
            ray.get(refs)

    return Agent(
        agent_id=agent_id,
        agent_config=agent_config,
        vllm_engines=vllm_engines,
        actor_model_group=actor_model,
        critic_model_group=critic_model,
        ref_model_group=ref_model,
        tokenizer=tokenizer,
        generate_kwargs=generate_kwargs,
        is_reasoning_model=agent_config["is_reasoning_model"]
    )

@dataclass
class Agent:
    # One agent includes actor model (many workers), critic model, reference model, and vllm engines for PPO training
    # The agent can start training after making trajectory samples by itself or with other agents
    def __init__(self, 
                agent_id,
                agent_config, 
                vllm_engines: List,
                actor_model_group: Optional[RayActorGroup] = None,
                critic_model_group: Optional[RayActorGroup] = None,
                ref_model_group: Optional[RayActorGroup] = None,
                tokenizer = None,
                generate_kwargs = None,
                is_reasoning_model = False):
        self.agent_id = agent_id
        self.agent_config = agent_config
        self.vllm_engines = vllm_engines
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.ref_model_group = ref_model_group
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.is_reasoning_model = is_reasoning_model

    def get_metadata(self):
        return {
            "agent_id": self.agent_id,
            "agent_role": self.agent_config["role"],
            "pretrain": self.agent_config["pretrain"],
            "llms": self.vllm_engines,
            "tokenizer": self.tokenizer,
            "generate_kwargs": self.generate_kwargs,
            "is_reasoning_model": self.is_reasoning_model
        }

    def save_actor_and_critic_model(self):
        if self.agent_config["is_tuning"]:
            ray.get(self.actor_model_group.async_save_model())
            if self.agent_config["critic_pretrain"] and self.agent_config["save_value_network"]:
                ray.get(self.critic_model_group.async_save_model())

            # save tokenizer
            self.tokenizer.save_pretrained(os.path.join(self.agent_config["save_path"], self.agent_id))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Multi agent papameters
    parser.add_argument("--agents", type=json.loads, nargs="+", required=True,
                      help="List of agent configurations, each containing agent_id and config")
    parser.add_argument("--default_agent", type=json.loads, default={},
                      help="Default configuration for all agents")
    parser.add_argument("--shared_agents", action="store_true", default=False,
                      help="Whether to share the same agent across all positions")
    parser.add_argument("--parallel_loading", action="store_true", default=False,
                      help="Whether to initialize agents in parallel")

    # Workflow parameters for multi-agent
    parser.add_argument("--workflow_args", type=json.loads, default={}, help="Workflow arguments (dict)")
    parser.add_argument("--workflow_func_path", type=str, default=None, help="Workflow function script path")
    parser.add_argument("--processor_func_path", type=str, default=None, help="reward shaping processor script path")
    parser.add_argument("--evaluate_mcts_methods", type=str, default="agent", help="Workflow function script path")

    # tool args
    parser.add_argument("--tools_config", type=json.loads, default={}, help="tool arguments (dict)")

    # reward alloc args
    parser.add_argument("--reward_alloc", type=json.loads, default={"name": "margin", "alpha": 0.5, "beta": 0.5, "use_ttrl": False}, help="reward_alloc arguments (dict)")

    # Ray and vLLM
    parser.add_argument("--ref_num_nodes", type=int, default=1, help="number of nodes for reference")
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reference")
    parser.add_argument("--reward_num_nodes", type=int, default=1, help="number of nodes for reward model")
    parser.add_argument(
        "--reward_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reward model"
    )
    parser.add_argument(
        "--colocate_actor_ref",
        action="store_true",
        default=False,
        help="whether to colocate reference and actor model, if true, they will share same gpus.",
    )

    parser.add_argument("--actor_num_nodes", type=int, default=1, help="number of nodes for actor")
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=8, help="number of gpus per node for actor")
    parser.add_argument("--critic_num_nodes", type=int, default=1, help="number of nodes for critic")
    parser.add_argument("--critic_num_gpus_per_node", type=int, default=8, help="number of gpus per node for critic")
    parser.add_argument(
        "--colocate_critic_reward",
        action="store_true",
        default=False,
        help="whether to colocate critic and reward model, if true, they will share same gpus.",
    )
    parser.add_argument(
        "--colocate_all_models",
        action="store_true",
        default=False,
        help="whether to colocate all models (including vLLM engines), if true, they will share same gpus.",
    )
    parser.add_argument("--truncate_prompt", action="store_true", default=False, help="whether to truncate prompt, if true, the input prompt surpassing the max prompt len will be truncated")

    # vLLM for text generation
    parser.add_argument(
        "--vllm_num_engines", type=int, default=None, help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument("--vllm_sync_backend", type=str, default="nccl", help="DeepSpeed -> vLLM weight sync backend")
    parser.add_argument("--vllm_sync_with_ray", action="store_true", default=False)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true", default=False, help="Disable CUDA graph in vLLM")
    parser.add_argument(
        "--vllm_enable_sleep",
        action="store_true",
        default=False,
        help="Enable sleep mode for vLLM when using --colocate_all_models",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.95,
        help="vLLM gpu_memory_utilization",
    )

    # Your Efficient RL Framework Secretly Brings You Off-Policy RL Training: https://fengyao.notion.site/off-policy-rl
    parser.add_argument("--enable_vllm_is_correction", action="store_true", default=False)
    parser.add_argument("--vllm_is_truncated_threshold", type=float, default=2)
    parser.add_argument("--vllm_is_level", type=str, default="token", choices=["token", "sequence", "geometric"], 
                        help="IS weighting level: token, sequence, or geometric mean")
    parser.add_argument("--vllm_is_mode", type=str, default="truncate", choices=["truncate", "mask"],
                        help="IS correction mode: truncate (TIS) or mask (MIS)")
    parser.add_argument("--vllm_is_truncated_threshold_lower", type=float, default=None,
                        help="Lower threshold for masked IS (MIS), defaults to 1/upper_threshold")
    parser.add_argument("--vllm_is_veto_threshold", type=float, default=1e-4,
                        help="Veto threshold for catastrophic token ratios, set to None to disable")

    # Async training using ray
    parser.add_argument("--async_train", action="store_true", default=False, help="Enable async training")

    # Checkpoints
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo_ray")
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument(
        "--use_ds_universal_ckpt", action="store_true", help="Use deepspeed universal checkpoint", default=False
    )

    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    ## Make EMA as an optional feature
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--ema_beta", type=float, default=0.992, help="EMA beta coefficient")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation (e.g., eager, flash_attention_2, flash_attention_3, kernels-community/vllm-flash-attn3)",
    )
    parser.add_argument("--use_liger_kernel", action="store_true", default=False, help="Enable Liger Kernel")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument(
        "--deepspeed_enable_sleep",
        action="store_true",
        default=False,
        help="Enable sleep mode for deepspeed when using --colocate_all_models",
    )
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=1, help="DeepSpeed tensor parallel size")

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # dynamic batch size
    parser.add_argument("--use_dynamic_batch", action="store_true", default=False)
    parser.add_argument("--rollout_max_tokens_per_gpu", type=int, default=None)
    parser.add_argument("--train_max_tokens_per_gpu", type=int, default=16192)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # PPO
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=1024, help="Batch size for make experience")
    parser.add_argument(
        "--vllm_generate_batch_size", type=int, default=None, help="Batch size for vLLM generating samples"
    )
    parser.add_argument("--micro_rollout_batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--eval_generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--eps_clip_low_high", type=float, nargs=2, default=None, help="PPO-clip low and high")
    parser.add_argument("--dual_clip", type=float, default=None, help="Dual-clip PPO")
    parser.add_argument("--value_clip", type=float, default=0.5, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=1, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--kl_horizon", type=int, default=10000)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument("--policy_loss_type", type=str, default="ppo", choices=["ppo", "gspo"])
    parser.add_argument(
        "--kl_estimator",
        type=str,
        default="k1",
        choices=["k1", "k2", "k3"],
        help=(
            "In GRPO, k3 is utilized as the loss function, while k2, when used as the loss, is nearly equivalent to k1."
        ),
    )
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument(
        "--entropy_loss_coef",
        type=float,
        default=None,
        help="Entropy loss coef, set to 0 means only enable entropy logs",
    )
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")

    # Reinforce/GRPO, etc
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo, reinforce_baseline, group_norm, dr_grpo",
    )
    parser.add_argument("--use_kl_loss", action="store_true", default=False, help="whether to use KL loss from GRPO")
    parser.add_argument(
        "--no_advantage_std_norm",
        action="store_true",
        default=False,
        help="disable dividing by std for advantages while keeping mean normalization",
    )
    parser.add_argument(
        "--overlong_buffer_len", type=float, default=None, help="reward with optional overlong penalty"
    )
    parser.add_argument("--overlong_penalty_factor", type=float, default=1, help="overlong penalty factor")

    # Context Parallel
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API (HTTP)")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)
    parser.add_argument("--agent_func_path", type=str, default=None, help="Agent script path")

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default=None,
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Path to the evaluation dataset")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--eval_temperature", type=float, default=0.6, help="Temperature for evaluation")
    parser.add_argument(
        "--eval_n_samples_per_prompt", type=int, default=4, help="Number of samples per prompt for evaluation"
    )

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--label_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--metadata_key", type=str, default="extra_info", help="JSON dataset metadata key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # Dynamic filtering
    parser.add_argument("--dynamic_filtering", action="store_true", default=False, help="Enable dynamic filtering")
    parser.add_argument(
        "--dynamic_filtering_reward_range", nargs=2, default=(0, 1), type=float, help="Dynamic filtering rewards range"
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # performance tuning
    parser.add_argument("--perf", action="store_true", default=False)

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    parser.add_argument("--dynamic_filtering_for_agents", action="store_true", default=False, help="Enable dynamic filtering for agents")

    # eval config
    parser.add_argument("--eval_before_training", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    # parser.add_argument("--eval_save_path", type=str, default=None, help="Path to save evaluation results")
    parser.add_argument("--verify_task", type=str, default="math")
    parser.add_argument("--verify_task_eval", type=str, default="math")
    args = parser.parse_args()
    # Set vLLM generate_batch_size to rollout_batch_size if not specified
    if not args.vllm_generate_batch_size:
        args.vllm_generate_batch_size = args.rollout_batch_size
    # Validate arguments
    if args.eps_clip_low_high is None:
        args.eps_clip_low_high = (args.eps_clip, args.eps_clip)

    if args.agent_func_path:
        args.remote_rm_url = "agent"

    # Validate evaluation settings
    if args.eval_dataset:
        assert args.remote_rm_url or args.workflow_func_path, \
            f"`eval_dataset` is only supported with `remote_rm_url` for agent."

    assert (
        args.n_samples_per_prompt * args.rollout_batch_size // args.micro_rollout_batch_size
        >= args.actor_num_nodes * args.actor_num_gpus_per_node // args.ring_attn_size // args.ds_tensor_parallel_size
    ), "The number of sample batches must be greater than or equal to the effective number of actor processes."

    # Multi-agent specific validations
    for agent_dict in args.agents:
        for agent_id, agent_config in agent_dict.items():
            # Set default parameters from default_agent
            for key, value in args.default_agent.items():
                if key not in agent_config:
                    agent_config[key] = value
            for key, value in vars(args).items():
                if key in ["default_agent", "workflow_args"]:
                    continue
                if key not in agent_config:
                    agent_config[key] = value

            # Validate critic pretrain settings
            if agent_config["advantage_estimator"] not in ["gae"]:
                agent_config["critic_pretrain"] = None
            elif agent_config["critic_pretrain"] is None:
                if not agent_config["remote_rm_url"] and not agent_config["agent_func_path"]:
                    agent_config["critic_pretrain"] = agent_config.get("reward_pretrain", "").split(",")[0]
                else:
                    agent_config["critic_pretrain"] = agent_config.get("pretrain")

            # Validate advantage estimator settings
            if agent_config.get("advantage_estimator") in ["rloo", "reinforce_baseline", "group_norm"]:
                assert agent_config.get("n_samples_per_prompt", 1) > 1 or args.workflow_args.get("max_num_nodes", -1) > 1, \
                    f"{agent_config.get('advantage_estimator')} requires n_samples_per_prompt > 1"

            # Validate remote_rm_url
            if agent_config.get("remote_rm_url"):
                agent_config["remote_rm_url"] = agent_config["remote_rm_url"].split(",")

            # Validate input template
            if agent_config.get("input_template") and "{}" not in agent_config.get("input_template"):
                print(f"[Warning] {{}} not in input_template for agent {agent_id}, set to None")
                agent_config["input_template"] = None

            if agent_config.get("input_template") and "\\n" in agent_config.get("input_template"):
                print(
                    f"[Warning] input_template for agent {agent_id} contains \\n characters instead of newline. "
                    "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
                )

            # Validate ring attention settings
            if agent_config.get("ring_attn_size", 1) > 1:
                # if not agent_config.get("packing_samples", False):
                if not agent_config.get("packing_samples", False):
                    print(f"[Warning] ring_attn_size > 1 requires packing_samples for agent {agent_id}.")
                    agent_config["packing_samples"] = True

            # Validate dynamic batch settings
            if agent_config.get("use_dynamic_batch", False):
                if not agent_config.get("packing_samples", False):
                    print(f"[Warning] Please enable packing_samples to accelerate when use_dynamic_batch is enabled for agent {agent_id}.")
                    agent_config["packing_samples"] = True
                if agent_config.get("rollout_max_tokens_per_gpu") is None:
                    print(f"[Warning] Set rollout_max_tokens_per_gpu to train_max_tokens_per_gpu for agent {agent_id}.")
                    agent_config["rollout_max_tokens_per_gpu"] = agent_config.get("train_max_tokens_per_gpu", 16192)

            # Validate packing samples settings
            if agent_config.get("packing_samples", False):
                if "flash_attention" not in agent_config.get("attn_implementation", ""):
                    print(
                        f"[Warning] Please use attn_implementation with flash_attention to accelerate when packing_samples is enabled for agent {agent_id}."
                    )
                    agent_config["attn_implementation"] = "flash_attention_2"
                assert agent_config.get("vllm_num_engines", 0) > 0, f"Only support `packing_samples` with vLLM for agent {agent_id}."

            # Validate vLLM settings
            if agent_config.get("vllm_enable_sleep", False) and not agent_config.get("colocate_all_models", False):
                print(f"Set vllm_enable_sleep to False when colocate_all_models is disabled for agent {agent_id}.")
                agent_config["vllm_enable_sleep"] = False

            # Validate colocate settings
            if agent_config.get("colocate_all_models", False) and agent_config.get("async_train", False):
                print(f"[Warning] Using colocate_all_models in async RLHF only colocates DeepSpeed models for agent {agent_id}.")

            # Validate async training settings
            if agent_config.get("async_train", False):
                assert not agent_config.get("vllm_enable_sleep", False), \
                    f"Async RLHF is not supported with vllm_enable_sleep for agent {agent_id}."

            # Validate KL loss settings
            if agent_config.get("use_kl_loss", False):
                if agent_config.get("kl_estimator") not in ["k2", "k3"]:
                    print(f"Recommend setting kl_estimator to 'k2' or 'k3' when using KL as a loss for agent {agent_id}")
            else:
                if agent_config.get("kl_estimator") not in ["k1"]:
                    print(f"Recommend setting kl_estimator to 'k1' when not using KL as a loss for agent {agent_id}.")

            # Set vLLM generate batch size
            if not agent_config.get("vllm_generate_batch_size"):
                agent_config["vllm_generate_batch_size"] = agent_config.get("rollout_batch_size", 1024)

            # Validate dynamic filtering settings
            if agent_config.get("dynamic_filtering", False):
                reward_range = agent_config.get("dynamic_filtering_reward_range", (0, 1))
                assert reward_range[0] < reward_range[1], \
                    f"dynamic_filtering_reward_range[0] must be less than reward_range[1] for agent {agent_id}"
                assert agent_config.get("remote_rm_url") or agent_config.get("agent_func_path"), \
                    f"remote_rm_url or agent_func_path must be specified when using dynamic filtering for agent {agent_id}"
                assert agent_config.get("n_samples_per_prompt", 1) > 1, \
                    f"n_samples_per_prompt must be greater than 1 when using dynamic filtering for agent {agent_id}"

    # ModelScope settings
    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub
        patch_hub()

    train(args)
