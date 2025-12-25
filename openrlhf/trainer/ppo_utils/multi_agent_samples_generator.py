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
from openrlhf.agent_workflows.utils import register_mcp_tools, register_openai_tools, print_tools, assign_action_mask
from openrlhf.agent_workflows.tools.manager import ToolManager
from openrlhf.agent_workflows.tools.mcp_manager import MCPManager
#from openrlhf.worlds.base_world import Samples

logger = init_logger(__name__)

class MultiAgentSamplesGenerator(SamplesGenerator):
    def __init__(self, agents, strategy, prompt_max_len, credit_model=None, *args, **kwargs):
        self.strategy = strategy
        self.args = strategy.args
        self.agents = agents
        # self.tokenizer_list = tokenizer_list
        self.prompt_max_len = prompt_max_len
        self.generate_max_len = self.args.generate_max_len
        self.credit_model = credit_model
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
        self.total_max_len = self.args.max_len if self.args.max_len is not None else self.prompt_max_len + self.generate_max_len
        self.packing_samples = getattr(self.args, "packing_samples", False)
        self._init_tool_manager()
        self._init_processor()

    def _init_processor(self):
        """
        We convert collected workflow trajectories into training samples for each agent with the given processor
        """
        if self.args.processor_func_path is None:
            # self.processor = processor
            self.processor = None
        elif self.args.processor_func_path.endswith(".py"):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "processor", self.args.processor_func_path)
            processor_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(processor_module)
            self.processor = processor_module.processor
        else:
            raise ValueError("Processor path must be a Python file")

    def _init_tool_manager(self):
        self.tools_config = getattr(self.args, "tools_config", {})
        # print(f"Tool config is: {self.tools_config}")
        assert self.packing_samples, "Only support packing samples"
        # assert getattr(self.args, "packing_samples", True), "Only support packing samples"

        if self.tools_config.get("mcp_url", None) is not None:
            self.tool_manager = MCPManager(self.tools_config)
            self.tools = register_mcp_tools(self.tool_manager)
        else:
            self.tool_manager = ToolManager(self.tools_config)
            self.tools = register_openai_tools(self.tools_config, self.tool_manager)

        self.tool_manager.set_tools(self.tools)
        print_tools(self.tools)

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
                truncate_prompt_tokens=self.args.prompt_max_len if self.args.truncate_prompt else None
                )
            print(f"[Get rank agent] Sampling params is: \n{sampling_params}")
            agent_dict = {
                "llm": llms[0],
                "credit_model": rank_credit_model,
                "credit_tokenizer": self.credit_tokenizer,
                "sampling_params": sampling_params,
                "enable_thinking": agent.get("is_reasoning_model", False),
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
            # print("[MultiAgentSamplesGenerator generate_samples] wakeup succeed", flush=True)

        rollout_samples = self._generate_vllm(all_prompts, all_labels, all_metadatas, **generate_kwargs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            for agent in self.agents:
                batch_vllm_engine_call(agent["llms"], "sleep")
            # print("[MultiAgentSamplesGenerator generate_samples] sleep succeed", flush=True)
        return rollout_samples

    @torch.no_grad()
    def _generate_vllm(self, all_prompts, all_labels, all_metadatas=None, **kwargs):
        args = self.strategy.args
        # Set return_list to False, and then we only get one llm for async request

        is_eval = kwargs.get("is_eval", False)
        rank_agents_list = [self.get_rank_agent(
            rank=rank,
            world_size=self.num_vllms,
            is_eval=is_eval) for rank in range(self.num_vllms)]

        # temperature = kwargs.get("temperature", 1.0)
        max_response_length = kwargs.get("max_new_tokens", 1024)
        truncate_length = self.prompt_max_len + max_response_length
        n_samples_per_prompt = kwargs.get("n_samples_per_prompt", args.n_samples_per_prompt)
        all_prompts = sum(
            [[prompt] * n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum(
            [[label] * n_samples_per_prompt for label in all_labels], [])
        all_metadatas = sum(
            [[metadata] * n_samples_per_prompt for metadata in all_metadatas], [])
        # use different distributed prompts in eval and train stage
        evaluate_mcts_methods = kwargs.get("evaluate_mcts_methods", "")
        # distribute_prompts_fn = self.distribute_prompts if not is_eval or evaluate_mcts_methods != "agent" else self.distribute_prompts_single_agent
        # distribute_prompts_fn = self.distribute_prompts_single_agent if is_eval and evaluate_mcts_methods == "agent" else self.distribute_prompts
        all_trajectories = self.distribute_prompts(
            all_prompts,
            all_labels,
            all_metadatas,
            rank_agents_list,
            is_eval
        )
        # if self.processor: 
        if self.processor:
            all_trajectories = self.processor(all_trajectories, self.num_agents, self.args)

        rollout_samples = self.prepare_samples(all_trajectories, truncate_length, is_eval)
        # assert isinstance(rollout_samples, list), f"rollout samples should be list"
        # assert isinstance(rollout_samples[0], Experience), f"rollout samples [0] should be experience"
        return rollout_samples

    def distribute_prompts(self, all_prompts, all_labels, all_metadatas, rank_agents_list, is_eval=False):
        # Start timing for the entire distribute_prompts phase
        distribute_start_time = time.time()
        # Distribute requests to engines and collect responses to outputs
        refs = []
        all_wrappers = []
        batch_size = (len(all_prompts) + len(rank_agents_list) - 1) // len(rank_agents_list)

        # Time the wrapper creation and request submission phase
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
                tool_manager=self.tool_manager,
                prompts=prompts,
                labels=labels,
                metadatas=metadatas,
                is_eval=is_eval,
                max_length=self.total_max_len,
            )
            refs.append(ref)
            all_wrappers.append(multi_agent_wrapper)

        ray.get(refs)

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
        logger.info(f"[MultiAgentWrapper distribute_prompts timing] Total distribute time: {total_distribute_time:.2f}s")

        all_trajectories = sum(all_trajectories, [])
        assert len(all_trajectories) == len(
            all_prompts), f"{len(all_trajectories)} vs {len(all_prompts)}"
        return all_trajectories

    def prepare_samples(self, all_trajectories, truncate_length=4096, is_eval=False):
        """
        Convert collected trajectories (single-level or nested list) to Experience objects.

        Supports:
            - list[dict]
            - list[list[dict]]

        Returns:
            - list[Experience] or list[list[Experience]] (matches input structure)
        """
        def _process_list_trajectories(all_trajectories):
            """deal all_trajectory of single agent"""
            samples = []
            for traj_data in all_trajectories:

                prompt = traj_data["prompt"]
                label = traj_data["label"]
                trajectory = traj_data["trajectory"]

                for step in trajectory:
                    sequence_ids = step["sequence_ids"]
                    output_ids = step["output_ids"]

                    # Create tensors
                    sequences = torch.tensor(sequence_ids)
                    attention_mask = torch.tensor([1] * len(sequences))
                    action_mask = torch.zeros_like(attention_mask)
                    action_mask[-len(output_ids):] = 1
                    if step.get("action_mask", None):

                        traj_mask = torch.tensor(
                            step["action_mask"],
                            dtype=action_mask.dtype,
                            device=action_mask.device,
                        )

                        assert traj_mask.numel() == len(output_ids), (
                            f"traj_mask len {traj_mask.numel()} != output_ids len {len(output_ids)} with sequence len {len(sequences)}"
                        )

                        action_mask[-len(output_ids):] = traj_mask

                        # seq_len = len(sequences)
                        # traj_mask = torch.tensor(step.get("action_mask", None), dtype=attention_mask.dtype)
                        # assert traj_mask.numel() == len(output_ids), f"output action mask len should be equal with output_ids, but get: traj_mask.numel() vs len(output_ids)"
                        # if traj_mask.numel() != seq_len:
                        #     prompt_len = seq_len - traj_mask.numel()
                        #     action_mask = torch.cat(
                        #         [
                        #             torch.zeros(prompt_len, dtype=traj_mask.dtype),
                        #             traj_mask,
                        #         ],
                        #         dim=0,
                        #     )
                        # else:
                        #     action_mask = traj_mask
                    # Apply length limit
                    sequences = sequences[:truncate_length].to("cpu")
                    attention_mask = attention_mask[:truncate_length].to("cpu")
                    action_mask = action_mask[1:truncate_length].to("cpu")

                    rollout_log_probs = None
                    if step.get("rollout_log_prob") is not None:
                        rollout_log_probs = torch.tensor(step["rollout_log_prob"][1:truncate_length]).to("cpu")

                    # Calculate response length
                    ones_indices = torch.where(action_mask)[0]
                    # response_length = (ones_indices[-1] - ones_indices[0] + 1).item() if len(ones_indices) else 0
                    response_length = len(output_ids)
                    total_length = attention_mask.float().sum()
                    is_clipped = total_length >= truncate_length

                    info = {
                        "response_length": torch.tensor([response_length]),
                        "total_length": torch.tensor([total_length]),
                        "response_clip_ratio": torch.tensor([is_clipped], dtype=torch.float),
                        # "agent_role": step.get("agent_role", "unknown"),
                        # "agent_id": step.get("agent_id", None),
                        "agent_id": torch.tensor([float(step.get("agent_id", 0))], dtype=torch.long),
                        "reward": torch.tensor([float(step.get("reward", 0.0))], dtype=torch.float),
                    }

                    samples.append(
                        Experience(
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
                    )
            return samples

        if not all_trajectories:
            return []

        # check list[list[dict]]
        if isinstance(all_trajectories[0], list):
            return [_process_list_trajectories(traj_group) for traj_group in all_trajectories]
        else:
            return _process_list_trajectories(all_trajectories)