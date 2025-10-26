# marti/marti/worlds/tool_world.py
import srsly
import random
from typing import List, Union
import numpy as np
from omegaconf import OmegaConf
import ray
import torch
from vllm import SamplingParams

from openrlhf.utils.logging_utils import init_logger
from marti.worlds.base_world import Samples, BaseWorld
from marti.worlds.tools.manager import ToolManager
from marti.worlds.tools.mcp_manager import MCPManager
from marti.worlds.tools.search import SearchToolExecutor
from marti.worlds.tools.sandbox import SandboxFusionExecutor
from openrlhf.verifiers.auto_verify import auto_verify

import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))

def assign_action_mask(turn):
    if "\n<|im_start|>user\n<tool_response>" in turn and "</tool_response><|im_end|>\n<|im_start|>assistant" in turn:
        return 0
    else:
        return 1


def register_mcp_tools(tool_manager):
    import asyncio
    asyncio.run(tool_manager.initialize())
    return tool_manager.openai_tools

def register_openai_tools(tools_config, tool_manager):
    """Register all configured tools."""
    tools = []
    # TODO: default configuration for deepcoder
    for tool_name, tool_cfg in tools_config.get("tools", {}).items():
        tool_type = tool_cfg.get("type")
        if tool_type == "search_r1":
            # Register search tool
            executor = SearchToolExecutor(
                base_url=tool_cfg["base_url"],
                topk=tool_cfg.get("topk", 3),
                timeout=tool_cfg.get("timeout", 15)
            )
            tool_manager.register_tool(tool_name, executor)

            schema = srsly.read_json(tool_cfg["schema_path"])
            tools.append(schema)
        elif tool_type == "sandbox_fusion":
            executor = SandboxFusionExecutor(
                base_url=tool_cfg["base_url"],
                timeout=tool_cfg.get("timeout", 30),
                language=tool_cfg.get("language", "python")
            )
            tool_manager.register_tool(tool_name, executor)

            schema = srsly.read_json(tool_cfg["schema_path"])
            tools.append(schema)
        else:
            logger.warning(
                f"Unknown tool type: {tool_type} for tool: {tool_name}")
    return tools

def print_tools(tools):
    logger.info(f"-------- Discovery {len(tools)} tools --------")
    for idx, tool in enumerate(tools):
        logger.info(f"Tool {idx}: {tool}")


class ToolWorld(BaseWorld):
    """
    Generate samples by tool-calling
    """

    def __init__(self, strategy, agents, *args, **kwargs):
        super().__init__(strategy, agents, *args, **kwargs)

        self.chat_template = self.args.get("chat_template", "")
        self.tools_config = self.args.get("tools_config", {})

        assert self.packing_samples, "Only support packing samples"

        if self.tools_config.get("mcp_url", None) is not None:
            self.tool_manager = MCPManager(self.tools_config)
            self.tools = register_mcp_tools(self.tool_manager)
        else:
            self.tool_manager = ToolManager(self.tools_config)
            self.tools = register_openai_tools(self.tools_config, self.tool_manager)

        print_tools(self.tools)

    def distribute_prompts(self, task, prompts, labels, metadata, llms, sampling_params, is_repeat=False):
        if is_repeat:
            all_prompts = [prompts for _ in llms]
            all_labels = [labels for _ in llms]
            all_metadata = [metadata for _ in llms]
        else:
            if len(prompts) < len(llms):
                raise ValueError("Number of prompts must be more than llms")
            chunk_size = (len(prompts) + len(llms) - 1) // len(llms)
            all_prompts = [
                prompts[i*chunk_size: (i+1)*chunk_size] for i in range(len(llms))]
            all_labels = [
                labels[i*chunk_size: (i+1)*chunk_size] for i in range(len(llms))]
            all_metadata = [
                metadata[i*chunk_size: (i+1)*chunk_size] for i in range(len(llms))]

        refs = []

        for per_llm_prompts, per_llm_labels, per_llm_metadata, llm in zip(all_prompts, all_labels, all_metadata, llms):
            refs.append(
                llm.add_requests.remote(
                    tool_manager=self.tool_manager,
                    sampling_params=sampling_params,
                    prompts=self.process_prompts(
                        per_llm_prompts),  # add generation prompt
                    labels=self.process_labels(per_llm_labels),  # split labels
                    max_length=self.total_max_len,
                    task=task,
                    hf_tokenizer=self.shared_tokenizer,
                    metadata=per_llm_metadata,  # metadata for each prompt
                )
            )
        ray.get(refs)

        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote())
        all_results = ray.get(all_output_refs)

        if is_repeat:
            all_results = [self.group_prompts(
                results, is_repeat) for results in all_results]
            assert len(all_results) == len(
                llms), f"{len(all_results)} vs {len(llms)}"
            for results in all_results:
                assert len(results) == len(
                    prompts), f"{len(results)} vs {len(prompts)}"
            return all_results  # List[List[str]]
        else:
            all_results = sum(all_results, [])
            assert len(all_results) == len(
                prompts), f"{len(all_results)} vs {len(prompts)}"
            return self.group_prompts(all_results, is_repeat)

    def group_prompts(self, all_outputs, is_eval=False):
        prompt_groups = {}
        for output in all_outputs:
            prompt = output["prompt"]
            prompt_groups.setdefault(prompt, []).append(output)

        if self.args.reward_alloc.use_ttrl and not is_eval:
            for prompt, outputs in prompt_groups.items():
                try:
                    solutions = [output["observation"][-1]
                                 for output in outputs]
                except (KeyError, IndexError, TypeError):
                    raise ValueError(
                        f"Invalid 'observation' field in outputs for prompt: {prompt}")

                rewards = auto_verify(
                    self.args.verify_task + "_ttt", len(solutions), solutions, solutions)
                assert len(rewards) == len(
                    outputs), "Mismatch between rewards and outputs"
                for reward, output in zip(rewards, outputs):
                    output["reward"] = reward

        # Reorder outputs to keep same prompts together
        grouped_outputs = []
        for prompt in prompt_groups.keys():
            grouped_outputs.extend(prompt_groups[prompt])
        return grouped_outputs

    def process_prompts(self, prompts, tokenize=False):
        return [self.shared_tokenizer.apply_chat_template(
            [{"role": "user", "content": self.chat_template.format(
                question=prompt)}],
            add_generation_prompt=True,
            tools=self.tools,
            tokenize=tokenize,
            # Support Qwen3 models and also compatible with other models
            enable_thinking=self.args.enable_thinking,
        ) for prompt in prompts]

    def process_labels(self, labels):
        if self.args.allow_label_list:
            assert self.args.separate_label_list is not None
            labels = [label.split(self.args.separate_label_list)
                      for label in labels]
        return labels

    @torch.no_grad()
    def evaluate_samples(self, eval_data):
        args = self.strategy.args
        sampling_params = SamplingParams(
            temperature=self.shared_generate_kwargs.get(
                "eval_temperature", 0.6),
            top_p=self.shared_generate_kwargs.get("top_p", 1.0),
            top_k=self.shared_generate_kwargs.get("top_k", -1),
            max_tokens=self.shared_generate_kwargs.get("eval_max_new_tokens", 1024),
            min_tokens=self.shared_generate_kwargs.get("min_new_tokens", 16),
            skip_special_tokens=self.shared_generate_kwargs.get(
                "skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        all_prompts, all_labels, all_metadata = eval_data["prompt"], eval_data["label"], eval_data["metadata"]

        # TODO: avioding generate outputs with all llms, which is time-cost
        if self.args.eval_workers > 0:
            llms = self.shared_llms[:self.args.eval_workers]
        else:
            llms = self.shared_llms

        all_results = self.distribute_prompts(args.verify_task_eval, all_prompts, all_labels, all_metadata, llms, sampling_params, is_repeat=True)

        # concate the multi-turn generated content into a full string of trajectory
        all_accuracies = [[output["reward"]
                           for output in outputs] for outputs in all_results]
        accuracy = np.mean([np.mean(acc) for acc in all_accuracies])
        print(f"[Evaluate Sample] Accuracy is: {accuracy}")
        return {"accuracy": accuracy, "metadata": all_results}

    @torch.no_grad()
    def generate_samples(self, all_prompts: Union[List[str], dict], rank=0, world_size=8) -> List[Samples]:
        """
        Generate samples and return a list of Samples.
        """
        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.shared_llms) <= world_size:
            llms = [self.shared_llms[rank % len(self.shared_llms)]]
        else:
            llms = self.shared_llms[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=self.shared_generate_kwargs.get("temperature", 1.0),
            top_p=self.shared_generate_kwargs.get("top_p", 1.0),
            top_k=self.shared_generate_kwargs.get("top_k", -1),
            max_tokens=self.shared_generate_kwargs.get("max_new_tokens", 1024),
            min_tokens=self.shared_generate_kwargs.get("min_new_tokens", 1),
            skip_special_tokens=self.shared_generate_kwargs.get(
                "skip_special_tokens", False),
            include_stop_str_in_output=True,
            logprobs=1 if args.enable_vllm_is_correction else None,
            # We need to add loss on eos token, reference: https://github.com/OpenRLHF/OpenRLHF/issues/627
        )
        print(f"[Tool world generate samples] sampling_params is: {sampling_params}")

        all_prompts, all_labels, all_metadata = all_prompts["prompt"], all_prompts["label"], all_prompts["metadata"]
        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum(
            [[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum(
            [[label] * args.n_samples_per_prompt for label in all_labels], [])
        all_metadata = sum(
            [[metadata] * args.n_samples_per_prompt for metadata in all_metadata], [])

        # Distribute requests to engines and collect responses to outputs
        all_results = self.distribute_prompts(args.verify_task, all_prompts, all_labels, all_metadata, llms, sampling_params, is_repeat=False)

        all_prompts, all_outputs, all_rewards = [], [], []
        for result in all_results:
            all_prompts.append(result["prompt"])
            all_outputs.append(result["observation"])
            all_rewards.append(result["reward"])

        all_prompt_token_ids = [
            self.tokenize_fn(
                self.shared_tokenizer,
                prompts,
                self.total_max_len,
                padding=False)["input_ids"]
            for prompts in all_prompts]

        if args.enable_vllm_is_correction:
            all_rollout_log_probs = [torch.tensor(r["rollout_log_probs"], device="cpu") for r in all_results]
            all_output_token_ids = [r["action_ids"] for r in all_results]
        else:
            all_output_token_ids = [
                self.tokenize_fn(
                    self.shared_tokenizer,
                    [output for output in outputs if len(output) != 0],
                    self.total_max_len,
                    padding=False)["input_ids"]
                for outputs in all_outputs]

        all_action_mask = [
            [assign_action_mask(output)
             for output in outputs if len(output) != 0]
            for outputs in all_outputs]

        # Add eos token avoiding truncation loss masking
        # eos_token_id = self.shared_tokenizer.eos_token_id
        # for outputs in all_outputs:
        #     outputs[-1] += [eos_token_id]

        # for i in random.sample(list(range(len(all_prompts))), k=2):
        #     print(f"Question {i+1}:", all_prompts[i])
        #     print(f"Answer {i+1}:", all_outputs[i])
        #     print(f"Labels {i+1}:", all_labels[i])
        #     print(f"Reward {i+1}:", all_rewards[i])
            # print(f"Prompt Token Ids {i+1}:", all_prompt_token_ids[i])
            # print(f"Output Token Ids {i+1}:", all_output_token_ids[i])
            # print("\n\n")
        # dynamic filtering
        # -------- Dynamic Filtering --------
        if args.dynamic_filtering: # filter_samples_by_reward/dynamic_filtering
            number_of_samples = len(all_prompts)
            filtered_prompts, filtered_outputs, filtered_rewards, filtered_rollout_log_probs = [], [], [], []
            filtered_prompt_toks, filtered_output_toks, filtered_action_masks = [], [], []
            num_filtered_prompts_low = 0
            num_filtered_prompts_high = 0

            for i in range(0, len(all_prompts), args.n_samples_per_prompt):
                batch_prompts = all_prompts[i : i + args.n_samples_per_prompt]
                batch_outputs = all_outputs[i : i + args.n_samples_per_prompt]
                batch_rewards = all_rewards[i : i + args.n_samples_per_prompt]
                batch_p_toks = all_prompt_token_ids[i : i + args.n_samples_per_prompt]
                batch_o_toks = all_output_token_ids[i : i + args.n_samples_per_prompt]
                batch_masks = all_action_mask[i : i + args.n_samples_per_prompt]
                batch_rollout_log_probs = all_rollout_log_probs[i: i + args.n_samples_per_prompt] if args.enable_vllm_is_correction else None

                if len(batch_rewards) < args.n_samples_per_prompt:
                    continue

                avg_reward = sum(batch_rewards) / len(batch_rewards)
                min_reward, max_reward = args.dynamic_filtering_reward_low, args.dynamic_filtering_reward_high

                if min_reward + 1e-6 < avg_reward < max_reward - 1e-6:
                    filtered_prompts.extend(batch_prompts)
                    filtered_outputs.extend(batch_outputs)
                    filtered_rewards.extend(batch_rewards)
                    filtered_prompt_toks.extend(batch_p_toks)
                    filtered_output_toks.extend(batch_o_toks)
                    filtered_action_masks.extend(batch_masks)
                    filtered_rollout_log_probs.extend(batch_rollout_log_probs) if args.enable_vllm_is_correction else None
                elif avg_reward <= min_reward + 1e-6:
                    num_filtered_prompts_low += len(batch_prompts)
                elif avg_reward >= max_reward - 1e-6:
                    num_filtered_prompts_high += len(batch_prompts)

            pass_rate_per_worker = len(filtered_prompts) / number_of_samples * 100
            low_rate_per_worker = num_filtered_prompts_low / number_of_samples * 100
            high_rate_per_worker = num_filtered_prompts_high / number_of_samples * 100
            print(
                f"[World dynamic filtering] Dynamic filtering pass rate: {pass_rate_per_worker:.2f}% vs low_reward rate: {low_rate_per_worker:.2f}% vs high_reward rate: {high_rate_per_worker:.2f}%; ({len(filtered_prompts)}/{number_of_samples})", flush=True
            )

            all_prompts, all_outputs, all_rewards, all_rollout_log_probs = (
                filtered_prompts,
                filtered_outputs,
                filtered_rewards,
                filtered_rollout_log_probs,
            )
            all_prompt_token_ids, all_output_token_ids, all_action_mask = (
                filtered_prompt_toks,
                filtered_output_toks,
                filtered_action_masks,
            )

        samples_list = []
        if len(all_outputs) < args.micro_rollout_batch_size or len(all_outputs) < args.n_samples_per_prompt:
            return {"sample": samples_list, "prompts": all_prompts, "outputs": all_outputs, "labels": all_rewards}
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            batch_outputs = all_outputs[i: i +
                                        self.strategy.args.micro_rollout_batch_size]
            batch_output_token_ids = all_output_token_ids[i: i +
                                                          self.strategy.args.micro_rollout_batch_size]
            batch_action_mask = all_action_mask[i: i +
                                                self.strategy.args.micro_rollout_batch_size]
            batch_prompt_token_ids = all_prompt_token_ids[i: i +
                                                          self.strategy.args.micro_rollout_batch_size]

            batch_pred_labels = all_rewards[i: i +
                                            self.strategy.args.micro_rollout_batch_size]
            batch_pred_labels = torch.tensor(
                batch_pred_labels, device="cpu", dtype=torch.float)
            if args.enable_vllm_is_correction:
                batch_rollout_log_probs = all_rollout_log_probs[i: i + self.strategy.args.micro_rollout_batch_size]
            else:
                batch_rollout_log_probs = None

            sequences = []
            packed_seq_lens = []
            attention_mask = []
            action_mask = []
            num_actions = []
            for j, outputs in enumerate(batch_outputs):
                prompt_token_ids = batch_prompt_token_ids[j]
                output_token_ids = sum(batch_output_token_ids[j], [])
                current_action_mask = [[action]*len(output) for action, output in zip(
                    batch_action_mask[j], batch_output_token_ids[j])]

                input_len = len(prompt_token_ids)
                output_len = len(output_token_ids)
                packed_seq_lens.append(input_len + output_len)
                sequences.extend(prompt_token_ids + output_token_ids)
                # https://github.com/OpenRLHF/OpenRLHF/issues/587
                attention_mask.extend([j + 1] * (input_len + output_len))
                flat_action_mask = [
                    v for sublist in current_action_mask for v in sublist]
                action_mask.extend(flat_action_mask)
                # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                # num_actions.append(max(1, sum(current_action_mask)))
                num_actions.append(max(1, output_len))

            rollout_log_probs = None
            if args.enable_vllm_is_correction:
                rollout_log_probs = torch.cat(batch_rollout_log_probs, dim=0).unsqueeze(0).to("cpu")

            sequences = torch.tensor(sequences, device="cpu").unsqueeze(0)
            attention_mask = torch.tensor(
                attention_mask, device="cpu").unsqueeze(0)
            action_mask = torch.tensor(action_mask, device="cpu").unsqueeze(0)

            response_length = torch.tensor(
                num_actions, device="cpu", dtype=torch.float)
            total_length = torch.tensor(
                packed_seq_lens, device="cpu", dtype=torch.float)
            samples_list.append(
                Samples(
                    sequences=sequences,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    num_actions=num_actions,
                    packed_seq_lens=packed_seq_lens,
                    response_length=response_length,
                    total_length=total_length,
                    labels=batch_pred_labels,
                    rollout_log_probs=rollout_log_probs,
                )
            )

        return {"sample": samples_list, "prompts": all_prompts, "outputs": all_outputs, "labels": all_rewards}
