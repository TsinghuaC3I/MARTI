from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from copy import deepcopy

from marti.models.model_utils import process_sequences
from openrlhf.utils.logging_utils import init_logger
from vllm import SamplingParams

from marti.worlds.base_world import BaseWorld, Samples
from openrlhf.verifiers.auto_reward_alloc import MultiAgentRewardAllocation

from marti.worlds.third_party.base import BaseMultiAgentGame
from marti.worlds.third_party.chain import MultiAgentChain
from marti.worlds.third_party.debate import MultiAgentDebate
from marti.worlds.third_party.mixture import MultiAgentMixture

from marti.agents.multi_agent import MAGraph, get_kwargs

logger = init_logger(__name__)

import importlib.util
import sys
from pathlib import Path
import uuid

def load_module_from_path(py_path: str):
    path = Path(py_path).resolve()
    mod_name = f"_dyn_{path.stem}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module

class MultiAgentWorld(BaseWorld):
    """
    Generate samples by agent/multi-agent interactions
    And then pass samples into experience maker of each actor worker
    """

    def __init__(self,
                 strategy,
                 agents,
                 *args, 
                 **kwargs):
        super().__init__(strategy, agents, *args, **kwargs)

        self.workflow_args = self.args.get("workflow_args", {})
        print("workflow args", self.workflow_args)
        self.num_agents = len(self.agents)

        if self.args.workflow_version == "custom":
            m = load_module_from_path(self.args.processor_func_path)
            self.multi_agent_reward   = m.MultiAgentReward
            self.multi_agent_workflow_class = m.MultiAgentWorkflow
            self.reward_alloc = self.multi_agent_reward(
                verify=self.args.verify_task,
                **self.args.reward_alloc)
            self.reward_alloc_eval = self.multi_agent_reward(verify=self.args.verify_task_eval)
        else:
            self.reward_alloc = MultiAgentRewardAllocation(
                verify=self.args.verify_task,
                **self.args.reward_alloc)
            self.reward_alloc_eval = MultiAgentRewardAllocation(verify=self.args.verify_task_eval)

    def _build_multi_world(self, world_agents, sampling_params):
        self.name2workflow = {
            "multi-agents-debate": MultiAgentDebate,
            "chain-of-agents": MultiAgentChain,
            "mixture-of-agents": MultiAgentMixture
        }
        if self.args.workflow_version == "old":
            self.workflow_class = self.name2workflow[self.args.agent_workflow]
            group_game: BaseMultiAgentGame = self.workflow_class(
                agent_list=world_agents,
                sampling_params=sampling_params,
                **self.workflow_args)
            group_args = {}
        elif self.args.workflow_version == "new":
            self.workflow_class = MAGraph
            kwargs = get_kwargs(self.args)
            group_game = MAGraph(
                agents=self.agents,
                agent_ids=kwargs['agent_ids'],
                agent_roles=kwargs['agent_roles'],
                agent_workflow=self.args.agent_workflow,
                prompt=kwargs['prompt'],
                spatial_adj_mats=kwargs['spatial_adj_mats'],
                temporal_adj_mats=kwargs['temporal_adj_mats'],
                sampling_params=sampling_params,
                **self.workflow_args
            )
            group_args = {"num_rounds": self.args.workflow_args.get('num_rounds')}
        elif self.args.workflow_version == "custom":
            group_game = self.multi_agent_workflow_class(
                agent_list=world_agents,
                sampling_params=sampling_params,
                **self.workflow_args)
            group_args = {}
        return group_game, group_args

    @torch.no_grad()
    def evaluate_samples(self, eval_data: Union[List[str], dict]):
        all_prompts, all_labels, all_indices = eval_data[
            "prompt"], eval_data["label"], eval_data["indice"]

        sampling_params = SamplingParams(
            temperature=self.shared_generate_kwargs.get("eval_temperature", 0.6),
            top_p=self.shared_generate_kwargs.get("top_p", 1.0),
            top_k=self.shared_generate_kwargs.get("top_k", -1),
            max_tokens=self.shared_generate_kwargs.get("max_new_tokens", 1024),
            min_tokens=self.shared_generate_kwargs.get("min_new_tokens", 16),
            skip_special_tokens=self.shared_generate_kwargs.get("skip_special_tokens", False),
            truncate_prompt_tokens=self.args.prompt_max_len if self.args.truncate_prompt else None
        )

        group_game, group_args = self._build_multi_world(self.agents, sampling_params)

        group_game.run(all_prompts, labels=all_labels, **group_args)
        group_history = group_game.get_history()

        _, outcome_rewards = self.reward_alloc_eval.run(
            group_history, all_labels)
        accuracy = np.mean(outcome_rewards)

        metadata = []
        for indice, prompt, label, debate in zip(all_indices, all_prompts, all_labels, group_history):
            metadata.append({
                "indice": str(indice),
                "prompt": prompt,
                "label": str(label),
                "history": debate
            })

        return {"accuracy": float(accuracy), "metadata": metadata}

    def get_rank_agents(self, rank, world_size):
        rank_agents = [{} for _ in range(self.num_agents)]
        for aid, agent in enumerate(self.agents):
            agent_llms = agent["llms"]
            
            if len(agent_llms) <= world_size:
                llms = [agent_llms[rank % len(agent_llms)]]
            else:
                llms = agent_llms[rank::world_size]

            rank_agents[aid]["llms"] = llms
            rank_agents[aid]["tokenizer"] = agent["tokenizer"]
            rank_agents[aid]["is_reasoning_model"] = agent["is_reasoning_model"]
        return rank_agents

    def tokenize_fn_with_tok(self, messages, tokenizer=None):
        tokenizer = self.shared_tokenizer if tokenizer is None else tokenizer
        # For inputs
        if isinstance(messages, list):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            prompt_max_len = self.args.prompt_max_len
        # For outputs
        elif isinstance(messages, str):
            prompt = messages
            prompt_max_len = self.args.generate_max_len
        else:
            raise NotImplementedError

        return self.tokenize_fn(tokenizer, prompt, prompt_max_len, padding=False)["input_ids"]

    def process_agent_data(self, agent_data):
        if isinstance(agent_data, dict):
            return {
                "prompt": [{"role": "user", "content": agent_data["user"]}],
                "output": agent_data["assistant"],
                "reward": agent_data["reward"],
                "agent_id": agent_data["agent_id"],
                "turn_id": agent_data["turn_id"]
            }
        elif isinstance(agent_data, list):
            return [self.process_agent_data(turn_data) for turn_data in agent_data]

    def run_game(self, rank_agents, all_prompts, all_labels):
        sampling_params = SamplingParams(
                            temperature=self.shared_generate_kwargs.get("temperature", 1.0),
                            top_p=self.shared_generate_kwargs.get("top_p", 1.0),
                            top_k=self.shared_generate_kwargs.get("top_k", -1),
                            max_tokens=self.shared_generate_kwargs.get("max_new_tokens", 1024),
                            min_tokens=self.shared_generate_kwargs.get("min_new_tokens", 16),
                            skip_special_tokens=self.shared_generate_kwargs.get("skip_special_tokens", False),
                            include_stop_str_in_output=True,
                            truncate_prompt_tokens=self.args.prompt_max_len if self.args.truncate_prompt else None)

        group_game, group_args = self._build_multi_world(rank_agents, sampling_params)
        group_game.run(all_prompts, labels=all_labels, **group_args)
        group_history = group_game.get_history()

        local_rewards, outcome_rewards = self.reward_alloc.run(
            group_history, all_labels, self.args.n_samples_per_prompt)

        def alloc_reward(agent_data, agent_reward):
            if isinstance(agent_data, dict):
                agent_data["reward"] = agent_reward
            elif isinstance(agent_data, list):
                for turn_data, turn_reward in zip(agent_data, agent_reward):
                    turn_data["reward"] = turn_reward

        for history, rewards in zip(group_history, local_rewards):
            for agent_data, agent_reward in zip(history, rewards):
                alloc_reward(agent_data, agent_reward)

        all_meta_data = []
        for prob_idx, history in enumerate(group_history):
            prob_meta_data = []
            for agent_data in history:
                prob_meta_data.append(self.process_agent_data(agent_data))
            all_meta_data.append(prob_meta_data)
        
        return all_meta_data, outcome_rewards

    @torch.no_grad()
    def generate_samples(self, all_prompts: Union[List[str], dict], rank=0, world_size=8):
        """
        Generate samples and return a list of Samples.
        """
        args = self.strategy.args
        rank_agents = self.get_rank_agents(rank, world_size)

        all_prompts, all_labels = all_prompts["prompt"], all_prompts["label"]
        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum(
            [[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum(
            [[label] * args.n_samples_per_prompt for label in all_labels], [])

        all_meta_data, outcome_rewards = self.run_game(
            rank_agents, all_prompts, all_labels)

        if self.args.agent_workflow in ["chain-of-agents", "mixture-of-agents"]:
            samples_list = []
            for i in range(0, len(all_meta_data), args.micro_rollout_batch_size):
                batch_meta_data = all_meta_data[i: i +
                                                args.micro_rollout_batch_size]
                batch_labels = outcome_rewards[i: i +
                                            args.micro_rollout_batch_size]

                # Create full trajectory data for credit assisgnment (mainly chain workflow)
                batch_prompts = [self.tokenize_fn_with_tok(agent_data[0]["prompt"]) for agent_data in batch_meta_data]

                batch_output_ids = [[self.tokenize_fn_with_tok(agent["output"]) for agent in agent_data] for agent_data in batch_meta_data]
                # We concatenate all outputs of the first agent from each turn
                batch_outputs = [sum([agent for agent in agent_data], []) for agent_data in batch_output_ids]
                
                batch_action_mask = [sum([[idx] * len(agent) for idx, agent in enumerate(agent_data)], []) for agent_data in batch_meta_data]
                
                batch_agent_actions = [[len(agent) for agent in agent_data] for agent_data in batch_meta_data]

                # Prepare samples for full trajectory
                batch_samples = self.prepare_samples(
                    batch_prompts, batch_outputs, batch_labels,
                    None if self.args.credit_model is None else batch_action_mask,
                    None if self.args.credit_model is None else batch_agent_actions, self.shared_tokenizer)

                # Create agents samples as additional info of group samples
                # TODO: support multiple turn for MoA
                info = [None for _ in range(self.num_agents)]
                for agent_id in range(self.num_agents):
                    agent_tokenizer = self.agents[agent_id]["tokenizer"]
                    agent_prompts = [self.tokenize_fn_with_tok(agent_data[agent_id]["prompt"], agent_tokenizer) for agent_data in batch_meta_data]
                    agent_outputs = [self.tokenize_fn_with_tok(agent_data[agent_id]["output"], agent_tokenizer) for agent_data in batch_meta_data]
                    agent_labels = [agent_data[agent_id]["reward"] for agent_data in batch_meta_data]

                    agent_samples = self.prepare_samples(
                        agent_prompts, agent_outputs, agent_labels, tokenizer=agent_tokenizer)
                    info[agent_id] = agent_samples
                
                batch_samples.info = info
                samples_list.append(batch_samples)

        elif self.args.agent_workflow == "multi-agents-debate":
            agent_results = [{"prompts": [], "outputs": [], "labels": []} for _ in range(self.num_agents)]
            # num_problems, num_turns, num_agents
            for agent_data in all_meta_data:
                for agent_id, agent in enumerate(agent_data):
                    agent_tokenizer = self.agents[agent_id]["tokenizer"]
                    for turn in agent:
                        agent_results[agent_id]["prompts"].append(self.tokenize_fn_with_tok(turn["prompt"], agent_tokenizer))
                        agent_results[agent_id]["outputs"].append(self.tokenize_fn_with_tok(turn["output"], agent_tokenizer))
                        agent_results[agent_id]["labels"].append(turn["reward"])

            samples_list = [[] for _ in range(self.num_agents)]
            for i in range(0, len(agent_results[0]["prompts"]), args.micro_rollout_batch_size):
                for agent_id in range(self.num_agents):
                    prompts = agent_results[agent_id]["prompts"][i: i + args.micro_rollout_batch_size]
                    outputs = agent_results[agent_id]["outputs"][i: i + args.micro_rollout_batch_size]
                    labels = agent_results[agent_id]["labels"][i: i + args.micro_rollout_batch_size]

                    agent_tokenizer = self.agents[agent_id]["tokenizer"]
                    samples_list[agent_id].append(self.prepare_samples(
                        prompts=prompts,
                        outputs=outputs,
                        pred_labels=labels,
                        tokenizer=agent_tokenizer
                    ))

        return {"sample": samples_list}

    def prepare_samples(self,
                        prompts,
                        outputs,
                        pred_labels,
                        action_mask_list=None,
                        num_agent_actions=None,
                        tokenizer=None):
        pred_labels = torch.tensor(
            pred_labels, device="cpu", dtype=torch.float)
        if not self.packing_samples:
            assert action_mask_list is None, "Customized action_mask (eg., agent mask in prime) is not support while packing_samples is False"
            # NOTE: concat all outputs to following format:
            #
            # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
            # | token token token token token | token token [EOS] [PAD] |
            # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
            # |<---------- prompt ----------->|<-------- answer ------->|
            max_input_len, max_output_len = 0, 0
            for prompt, output in zip(prompts, outputs):
                max_input_len = max(max_input_len, len(prompt))
                max_output_len = max(max_output_len, len(output))

            pad_token_id, eos_token_id = tokenizer.pad_token_id, tokenizer.eos_token_id
            sequences = []
            for prompt, output in zip(prompts, outputs):
                # left padding input
                input_len = len(prompt)
                input_ids = [pad_token_id] * \
                    (max_input_len - input_len) + list(prompt)

                # right padding output
                output_len = len(output)
                output_ids = list(
                    output) + [pad_token_id] * (max_output_len - output_len)

                if output_ids[output_len - 1] != eos_token_id:
                    output_ids[min(output_len, len(
                        output_ids) - 1)] = eos_token_id

                # concat input and output
                sequences.append(input_ids + output_ids)

            sequences = torch.tensor(sequences)
            sequences, attention_mask, action_mask = process_sequences(
                sequences, max_input_len, eos_token_id, pad_token_id
            )
            sequences = sequences.to("cpu")
            attention_mask = attention_mask.to("cpu")
            action_mask = action_mask.to("cpu")
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                labels=pred_labels,
            )
        else:
            # NOTE: concat all outputs to following format:
            #
            # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
            # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
            pad_token_id, eos_token_id = tokenizer.pad_token_id, tokenizer.eos_token_id
            sequences = []
            packed_seq_lens = []
            attention_mask = []
            num_actions = []
            for i, output in enumerate(outputs):
                prompt = prompts[i]
                input_len = len(prompt)
                output_len = len(output)
                packed_seq_lens.append(input_len + output_len)
                sequences.extend(prompt + list(output))
                attention_mask.extend([i + 1] * (input_len + output_len))

                # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                # num_actions.append(max(1, sum(current_action_mask)))
                num_actions.append(max(1, output_len))
            sequences = torch.tensor(sequences, device="cpu").unsqueeze(0)
            attention_mask = torch.tensor(
                attention_mask, device="cpu").unsqueeze(0)
            response_length = torch.tensor(
                num_actions, device="cpu", dtype=torch.float)
            total_length = torch.tensor(
                packed_seq_lens, device="cpu", dtype=torch.float)

            if action_mask_list is not None:
                action_mask = sum(action_mask_list, [])
                assert len(action_mask) == sum(
                    num_actions), f"action_mask ({len(action_mask)}) and num_actions ({sum(num_actions)}) should have the same length"
                # TODO: action_mask should be a int tensor not bool tensor
                action_mask = torch.tensor(
                    action_mask, device="cpu", dtype=torch.int).unsqueeze(0)
            else:
                action_mask = None

            # TODO: number of agents in each sample should keep consistent, so we save num_agent_actions in list rather than torch.tensor

            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=num_actions,
                packed_seq_lens=packed_seq_lens,
                response_length=response_length,
                total_length=total_length,
                num_agent_actions=num_agent_actions,
                labels=pred_labels,
            )

        return samples
