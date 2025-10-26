import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import timedelta
from typing import Any, List, Tuple, Union

import ray
import torch
from openrlhf.agent_workflows.workflow_wrapper import MultiAgentWrapper
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions
from openrlhf.utils.utils import remove_pad_token, zero_pad_sequences
from openrlhf.trainer.ppo_utils.experience_maker import Experience, to, pin_memory, SamplesGenerator
from openrlhf.trainer.ppo_utils.multi_agent_experience_maker import MultiAgentExperience
from vllm import SamplingParams
#from openrlhf.worlds.base_world import Samples

logger = init_logger(__name__)

def update_samples_with_rewards(rewards_info, samples_list):
    """Process rewards info and update samples with rewards, scores and extra logs.

    Args:
        rewards_info: List of reward information dictionaries
        samples_list: List of Experience objects to update
    """
    # Process rewards and scores
    samples_len = [len(sample.sequences) for sample in samples_list]

    rewards_list = torch.cat([torch.as_tensor(info["rewards"]) for info in rewards_info], dim=0).split(samples_len)
    if "scores" in rewards_info[0]:
        scores_list = torch.cat([torch.as_tensor(info["scores"]) for info in rewards_info], dim=0).split(samples_len)
    else:
        scores_list = rewards_list

    # Process extra_logs if present
    if "extra_logs" in rewards_info[0]:
        # Merge all extra_logs tensors first
        merged_logs = {
            key: torch.cat(
                [torch.as_tensor(logs[key]) for logs in [info["extra_logs"] for info in rewards_info]], dim=0
            ).split(samples_len)
            for key in rewards_info[0]["extra_logs"].keys()
        }

    # Update samples with rewards, scores and extra logs
    for i, samples in enumerate(samples_list):
        samples.rewards = rewards_list[i]
        samples.scores = scores_list[i]
        samples.info["score"] = scores_list[i]
        samples.info["reward"] = rewards_list[i]
        if "extra_logs" in rewards_info[0]:
            for key, values in merged_logs.items():
                samples.info[key] = values[i]

    return samples_list


class MultiAgentSamplesGenerator(SamplesGenerator):
    def __init__(self, agents, strategy, prompt_max_len, credit_model=None, *args, **kwargs):
        self.strategy = strategy
        self.args = strategy.args
        self.agents = agents
        #logger.warning(f"agents: {self.agents}")
        # self.tokenizer_list = tokenizer_list
        self.prompt_max_len = prompt_max_len
        self.credit_model = credit_model
        # TODO 用于适配单独的credit model
        self.credit_tokenizer=None
        """
        agents: List[Dict[str, Any]]
            [{
                "agent_id": unique agent id
                "agent_role": agent role (generator/refiner/verifier/coder/...)
                "pretrain": path to pretrain models
                "llms": a list of vllm engines
                "tokenizer": hf tokenizer
                "generate_kwargs": generate kwargs, which is different from vllm.SamplingParams
                "is_reasoning_model": reasoning model with <think> tags or not
            }]
        """
        logger.warning(f"args: {self.args}")
        self.workflow_args = getattr(self.args, "workflow_args", {})

        self.num_agents = len(self.agents)
        self.num_vllms = len(self.agents[0]["llms"])

    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def get_rank_agent(self, rank, world_size, is_eval=False):
        """
        Get the first llm for async request
        """
        # credit_model
        rank_credit_model = None
        if isinstance(self.credit_model, list) and self.credit_model:
            num_rm_engines = len(self.credit_model)
            if num_rm_engines <= world_size:
                rank_credit_model = self.credit_model[rank % num_rm_engines]
            else:
                rank_credit_model = self.credit_model[rank::world_size][0]
        else:
            rank_credit_model = self.credit_model

        rank_agents = [{} for _ in range(self.num_agents)]
        for aid, agent in enumerate(self.agents):
            agent_llms = agent["llms"]
            if len(agent_llms) <= world_size:
                llms = [agent_llms[rank % len(agent_llms)]]
            else:
                llms = agent_llms[rank::world_size]
            #有问题：这里还得再调 参数找不到 暂时先固定住了
            generate_kwargs = agent.get("generate_kwargs") or {}
            sampling_params = SamplingParams(
                temperature=generate_kwargs.get(
                    "eval_temperature" if is_eval else "temperature", 1.0),
                top_p=generate_kwargs.get("top_p", 1.0),
                top_k=generate_kwargs.get("top_k", -1),
                max_tokens=generate_kwargs.get("eval_max_new_tokens" if is_eval else "max_new_tokens", 1024),
                min_tokens=generate_kwargs.get("min_new_tokens", 16),
                skip_special_tokens=generate_kwargs.get(
                    "skip_special_tokens", False),
                include_stop_str_in_output=True,
                logprobs=1 if self.args.enable_vllm_is_correction else None,
                )
            print(f"[Get rank agent] Sampling params is: \n{sampling_params}")
            agent_dict = {
                "llm": llms[0],
                "credit_model": rank_credit_model,
                "credit_tokenizer": self.credit_tokenizer,
                "sampling_params": sampling_params
            }
            for use_key in ["agent_id", "agent_role", "tokenizer", "is_reasoning_model"]:
                agent_dict[use_key] = deepcopy(agent[use_key])

            rank_agents[aid] = agent_dict
        return rank_agents

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, all_metadatas=None, **generate_kwargs) -> List[Experience]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        # vLLM wakeup when vllm_enable_sleep
        logger.warning(f"self.strategy.args.vllm_enable_sleep {self.strategy.args.vllm_enable_sleep}")

        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            for agent in self.agents:
                batch_vllm_engine_call(agent["llms"], "wake_up")
            print("成功wakeup", flush=True)
            logger.warning(f"成功wakeup")

        rollout_samples = self._generate_vllm(all_prompts, all_labels, all_metadatas, **generate_kwargs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            for agent in self.agents:
                batch_vllm_engine_call(agent["llms"], "sleep")
            print("成功sleep", flush=True)
            logger.warning(f"成功sleep")



        return rollout_samples

    @torch.no_grad()
    def _generate_vllm(self, all_prompts, all_labels, all_metadatas=None, **kwargs):
        args = self.strategy.args
        # Set return_list to False, and then we only get one llm for async request

        # Set up sampling parameters
        # sampling_params = SamplingParams(
        #     temperature=kwargs.get("temperature", 1.0),
        #     top_p=kwargs.get("top_p", 1.0),
        #     top_k=kwargs.get("top_k", -1),
        #     max_tokens=kwargs.get("max_new_tokens", 1024),
        #     min_tokens=kwargs.get("min_new_tokens", 1),
        #     skip_special_tokens=kwargs.get("skip_special_tokens", False),
        #     include_stop_str_in_output=True,
        #     logprobs=1 if self.strategy.args.enable_vllm_is_correction else None,
        # )
        # TODO check is eval的使用方式，优化
        is_eval = kwargs.get("is_eval", False)
        rank_agents_list = [self.get_rank_agent(
            rank=rank,
            world_size=self.num_vllms,
            is_eval=is_eval) for rank in range(self.num_vllms)]
        print(f"llll-0000000000")

        temperature = kwargs.get("temperature", 1.0)
        max_response_length = kwargs.get("max_new_tokens", 1024)
        truncate_length = self.prompt_max_len + max_response_length
        n_samples_per_prompt = kwargs.get("n_samples_per_prompt", args.n_samples_per_prompt)
        all_prompts = sum(
            [[prompt] * n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum(
            [[label] * n_samples_per_prompt for label in all_labels], [])
        all_metadatas = sum(
            [[metadata] * n_samples_per_prompt for metadata in all_metadatas], [])
        print(f"llll-111111111")
        all_trajectories = self.distribute_prompts(
            all_prompts,
            all_labels,
            all_metadatas,
            rank_agents_list,
            is_eval
        )
        print(f"llll-22222222")

        rollout_samples = self.prepare_samples(all_trajectories, truncate_length, is_eval)
        assert isinstance(rollout_samples, list), f"rollout samples should be list"
        assert isinstance(rollout_samples[0], Experience), f"rollout samples [0] should be experience"
        return rollout_samples

    def distribute_prompts(self, all_prompts, all_labels, all_metadatas, rank_agents_list, is_eval=False):
        # Start timing for the entire distribute_prompts phase
        distribute_start_time = time.time()
        # Distribute requests to engines and collect responses to outputs
        refs = []
        all_wrappers = []
        batch_size = (len(all_prompts) + len(rank_agents_list) - 1) // len(rank_agents_list)

        # Time the wrapper creation and request submission phase
        wrapper_creation_start = time.time()
        expected_lens = []
        for i, agent_list in enumerate(rank_agents_list):
            prompts = all_prompts[i * batch_size : (i + 1) * batch_size]
            labels = all_labels[i * batch_size : (i + 1) * batch_size]
            metadatas = all_metadatas[i * batch_size : (i + 1) * batch_size]
            expected_lens.append(len(prompts))
            multi_agent_wrapper = MultiAgentWrapper.remote(
                agents=agent_list,
                workflow_args=self.workflow_args,
                workflow_func_path=self.args.workflow_func_path
            )
            ref = multi_agent_wrapper.add_requests.remote(
                # tool_manager=self.tool_manager,
                prompts=prompts,
                labels=labels,
                metadatas=metadatas,
                is_eval=is_eval,
                # max_length=self.total_max_len,
            )
            refs.append(ref)
            all_wrappers.append(multi_agent_wrapper)

        print(f"llll-33333333")
        ray.get(refs)
        workflow_execution_time = time.time() - wrapper_creation_start

        # Time the result collection phase
        result_collection_start = time.time()
        all_output_refs = []

        for expected_len, wrapper in zip(expected_lens, all_wrappers):
            all_output_refs.append(wrapper.get_responses.remote(expected_len=expected_len))
        all_trajectories = ray.get(all_output_refs)
        result_collection_time = time.time() - result_collection_start
        
        # Calculate total time
        total_distribute_time = time.time() - distribute_start_time
        
        # Log comprehensive timing information
        logger = init_logger(__name__)
        logger.info(f"MultiAgentWrapper distribute_prompts timing - "
                   f"Total prompts: {len(prompts)}, "
                   f"Workflow execution: {workflow_execution_time:.2f}s, "
                   f"Result collection: {result_collection_time:.2f}s, "
                   f"Total distribute time: {total_distribute_time:.2f}s")

        all_trajectories = sum(all_trajectories, [])
        assert len(all_trajectories) == len(
            all_prompts), f"{len(all_trajectories)} vs {len(all_prompts)}"
        return all_trajectories

    def prepare_samples(self, all_trajectories, truncate_length=4096, is_eval=False):
        """
        Convert collected trajectories to a list of Experience objects.
        Args:
            all_trajectories: list of dicts containing prompt, label, trajectory, final_reward
            tokenizer: tokenizer for token id to text conversion (optional)
            max_length: truncate length
        Returns:
            List[Experience]
        """
        samples_list = []

        for traj_data in all_trajectories:
            prompt = traj_data["prompt"]
            label = traj_data["label"]
            trajectory = traj_data["trajectory"]

            # 对 trajectory 中的每个 step 分别处理
            for step in trajectory:
                sequence_ids = step["sequence_ids"]
                output_ids = step["output_ids"]

                # Create tensors
                sequences = torch.tensor(sequence_ids)
                attention_mask = torch.tensor([1] * len(sequences))
    
                # Create action mask based on tokenized action_ranges
                action_mask = torch.zeros_like(attention_mask)
                action_mask[-len(output_ids):] = 1

                # Apply length limit
                sequences = sequences[:truncate_length].to("cpu")
                attention_mask = attention_mask[:truncate_length].to("cpu")
                action_mask = action_mask[1:truncate_length].to("cpu")
                #logger.warning(f"step: {step}")
                if step["rollout_log_prob"] is not None:
                    rollout_log_probs = torch.tensor(step["rollout_log_prob"][1:truncate_length]).to("cpu")
                else:
                    rollout_log_probs = None

                # Calculate response length (distance between first and last 1)
                ones_indices = torch.where(action_mask)[0]
                response_length = (ones_indices[-1] - ones_indices[0] + 1).item() if len(ones_indices) else 0
                total_length = attention_mask.float().sum()
                is_clipped = total_length >= truncate_length

                info = {
                    "response_length": torch.tensor([response_length]),
                    "total_length": torch.tensor([total_length]),
                    "response_clip_ratio": torch.tensor([is_clipped], dtype=torch.float),
                    "agent_role": step.get("agent_role", "unknown"),
                    # "node_id": step["metadata"].get("expand_id", None),
                    "agent_id": step.get("agent_id", None),
                    "reward": step.get("reward", None),
                    # "timestamp": step.get("metadata", {}).get("timestamp", None),
                }

                rollout_sample = Experience(
                    sequences=sequences.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                    action_mask=action_mask.unsqueeze(0),
                    rollout_log_probs=(
                        rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None
                    ),
                    prompts=[prompt],
                    labels=[label],
                    rewards=torch.tensor([step["reward"]]) if not is_eval else traj_data["final_reward"],
                    scores=torch.tensor([step["reward"]]) if not is_eval else traj_data["final_reward"],
                    info=info,
                )
                samples_list.append(rollout_sample)

        return samples_list