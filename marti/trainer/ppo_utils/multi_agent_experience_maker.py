import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import timedelta
from typing import Any, List, Tuple, Union

import ray
import torch

from marti.models.utils import compute_approx_kl, compute_reward, masked_mean
from marti.trainer.ray.launcher import RayActorGroup
from marti.utils.logging_utils import init_logger
from marti.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions
from marti.utils.utils import remove_pad_token, zero_pad_sequences
from marti.trainer.ppo_utils.experience_maker import Experience, RemoteExperienceMaker

logger = init_logger(__name__)

class MultiAgentExperience(Experience):
    def __init__(self, *args, **kwargs):
        super.__init__(*args, **kwargs)


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

class MultiAgent_RemoteExperienceMaker(RemoteExperienceMaker):
    def __init__(
        self,
        agent_list,
        kl_controller,
        strategy=None,
        # tokenizer_list=None,
        remote_reward_model=None,
        **kwargs,
    ):
        #super().__init__()

        self.agent_list = agent_list
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.advantage_estimator = strategy.args.advantage_estimator
        self.args = strategy.args

        # remote_rm_url indicates that the remote reward model is agent enviroment, remote http server or custom reward func
        self.remote_rm_url = self.args.remote_rm_url
        self.remote_reward_model = remote_reward_model
        # self.tokenizer_list = tokenizer_list
        self.groups = None
        
        self.critic_model_group = kwargs.get('critic_model_group', None)
        self.reward_model_group = kwargs.get('reward_model_group', None)

    def split_rollout_samples(self, rollout_samples):
        for i, sample in enumerate(rollout_samples):
            sample.index = [i]

        samples_list = []
        if self.args.use_dynamic_batch:
            assert False, f"mcst don't support use_dynamic_batch"
        else:
            batch_size = self.args.micro_rollout_batch_size
            assert self.args.micro_rollout_batch_size == 1, f"micro_rollout_batch_size must be 1."
            for i in range(0, len(rollout_samples), batch_size):
                concat_samples = rollout_samples[i : i + batch_size]
                samples_list.extend(concat_samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples_list: List[Experience]) -> List[Experience]:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.

        Modified to support multiple agents: each sample may have an agent_id indicating
        which (actor_model_group, initial_model_group) to use.
        """
        start_time = time.time()
        logger.info(f"🚀 Starting experience making with {sum([len(s.sequences) for s in samples_list])} samples")

        args = self.strategy.args
        device = "cpu"

        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list]

        agent_ids = []
        for s in samples_list:
            if "agent_id" in s.info:
                agent_ids.append(int(s.info["agent_id"]))
            elif "agent_index" in s.info:
                agent_ids.append(int(s.info["agent_index"]))
            else:
                agent_ids.append(0)

        # The rewards are already filled in the samples_list, such as the agent's environment rewards
        if samples_list[0].rewards is not None:
            pass
        elif self.remote_rm_url:
            queries_list = sum(
                [
                    self.tokenizer.batch_decode(remove_pad_token(seq, attention_mask), skip_special_tokens=False)
                    for seq, attention_mask in zip(sequences_list, attention_mask_list)
                ],
                [],
            )
            prompts_list = sum([s.prompts for s in samples_list], [])
            labels_list = sum([s.labels for s in samples_list], [])
            # Keep the remote call asynchronous
            r_refs = self.remote_reward_model.get_rewards.remote(queries_list, prompts_list, labels_list)
        else:
            # Batch call reward model
            r_refs = self.reward_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                attention_mask=attention_mask_list,
                pad_sequence=[True] * len(samples_list),
            )

        # groups: agent_id -> list of indices in samples_list
        groups = {}
        for idx, aid in enumerate(agent_ids):
            groups.setdefault(aid, []).append(idx)
        self.groups = groups
        actor_action_log_probs_ref_per_agent = {}
        base_action_log_probs_ref_per_agent = {}
      
        # all agent forward all samples
        for aid in groups.keys():
            try:
                actor_group, initial_group = self.agent_list[aid].actor_model_group, self.agent_list[aid].ref_model_group
            except Exception as e:
                raise ValueError(f"agent id {aid} not found in self.agent_list") from e
            # if actor_group is None:
            #     continue
            if actor_group is not None:
                actor_action_log_probs_ref_per_agent[aid] = actor_group.async_run_method_batch(
                    method_name="forward",
                    sequences=sequences_list,
                    action_mask=action_mask_list,
                    attention_mask=attention_mask_list,
                )

                # Sync to avoid GPU OOM when colocate models
                if args.colocate_all_models or args.colocate_actor_ref: 
                    ray.get(actor_action_log_probs_ref_per_agent[aid])
                    ray.get(actor_group.async_run_method(method_name="empty_cache"))
            else:
                actor_action_log_probs_ref_per_agent[aid] = ray.put([[None]] * len(samples_list))

            if initial_group is not None:
                base_action_log_probs_ref_per_agent[aid] = initial_group.async_run_method_batch(
                    method_name="forward",
                    sequences=sequences_list,
                    action_mask=action_mask_list,
                    attention_mask=attention_mask_list,
                )
                if args.colocate_all_models or args.colocate_actor_ref:
                    ray.get(base_action_log_probs_ref_per_agent[aid])
                    ray.get(initial_group.async_run_method(method_name="empty_cache"))
            else:
                base_action_log_probs_ref_per_agent[aid] = ray.put([[None]] * len(samples_list))

        # Batch call critic model
        if self.agent_list[0].critic_model_group is not None:
            if args.colocate_critic_reward and not self.remote_rm_url:
                ray.get(r_refs)
                ray.get(self.agent_list[0].reward_model_group.async_run_method(method_name="empty_cache"))

            value_ref = self.agent_list[0].critic_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )
            if args.colocate_all_models or args.colocate_critic_reward:
                ray.get(value_ref)
                ray.get(self.critic_model_group.async_run_method(method_name="empty_cache"))
        else:
            value_ref = ray.put([[None]] * (len(samples_list) * args.ring_attn_size * args.ds_tensor_parallel_size))

        duplicate_factor = args.ring_attn_size * args.ds_tensor_parallel_size

        ray.get(value_ref)
        value_list = sum(ray.get(value_ref)[::duplicate_factor], [])

        action_log_probs_list = [None] * len(samples_list)
        base_action_log_probs_list = [None] * len(samples_list)

        # get correct log probs for different agents
        actor_outputs_per_agent = {}
        base_outputs_per_agent = {}
        
        for aid in groups.keys():
            actor_raw = ray.get(actor_action_log_probs_ref_per_agent[aid])
            actor_expanded = sum(actor_raw[::duplicate_factor], [])
            if len(actor_expanded) != len(samples_list):
                raise RuntimeError(
                    f"Actor returned {len(actor_expanded)} outputs for agent {aid}, expected {len(samples_list)}"
                )
            actor_outputs_per_agent[aid] = actor_expanded

            base_raw = ray.get(base_action_log_probs_ref_per_agent[aid])
            base_expanded = sum(base_raw[::duplicate_factor], [])
            if len(base_expanded) != len(samples_list):
                raise RuntimeError(
                    f"Initial model returned {len(base_expanded)} outputs for agent {aid}, expected {len(samples_list)}"
                )
            base_outputs_per_agent[aid] = base_expanded
        
        for sample_idx, aid in enumerate(agent_ids):
            action_log_probs_list[sample_idx] = actor_outputs_per_agent[aid][sample_idx]
            base_action_log_probs_list[sample_idx] = base_outputs_per_agent[aid][sample_idx]

        # assert None not in action_log_probs_list, "Some action logprobs missing after per-agent calls"
        # assert None not in base_action_log_probs_list, "Some base action logprobs missing after per-agent calls"
        #logger.warning(f"sample_list_111: {samples_list}")
        if samples_list[0].rewards is not None:
            pass
        elif self.remote_rm_url:
            # Get rewards info from remote model
            rewards_info = ray.get(r_refs)
            update_samples_with_rewards(rewards_info, samples_list)
        else:
            rewards_list = sum(ray.get(r_refs)[::duplicate_factor], [])
            for i, samples in enumerate(samples_list):
                samples.rewards = rewards_list[i]
                samples.info["reward"] = rewards_list[i]

        assert (
            len(samples_list) == len(action_log_probs_list) == len(base_action_log_probs_list) == len(value_list)
        ), f"len(samples_list): {len(samples_list)}, len(action_log_probs_list): {len(action_log_probs_list)}, len(base_action_log_probs_list): {len(base_action_log_probs_list)}, len(value_list): {len(value_list)}"

        for i, (samples, action_log_probs, base_action_log_probs, value) in enumerate(
            zip(samples_list, action_log_probs_list, base_action_log_probs_list, value_list)
        ):
            if action_log_probs is None:
                # Create a zero tensor with the same shape as action_mask instead of a float
                kl = torch.zeros_like(samples.action_mask, dtype=torch.float32, device=device)
                kl_mean = 0.0
            elif (base_action_log_probs is not None) and (not args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.strategy.args.kl_estimator,
                )
                kl_mean = masked_mean(kl, samples.action_mask, dim=-1)
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)
                kl_mean = masked_mean(kl, samples.action_mask, dim=-1)

            if not args.use_kl_loss:
                base_action_log_probs = None

            # Update experience with new information
            samples.action_log_probs = action_log_probs
            samples.base_action_log_probs = base_action_log_probs
            samples.values = value
            samples.kl = kl
            samples.info["kl"] = kl_mean

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Experience making completed in {time_str}")
        return samples_list


    def get_agent_experiences(self, samples_list: List[Experience]) -> List[List[Experience]]:
        """
        use self.groups to extra Experience list of agents

        Args:
            samples_list (List[Experience]): all samples 
        Returns:
            List[List[Experience]]: list of different agent experience_list
        """
        if not hasattr(self, "groups") or self.groups is None:
            raise ValueError("self.groups has not been inited!!")

        agent_experience_list = []
        for aid in sorted(self.groups.keys()):
            indices = self.groups[aid]
            agent_samples = [samples_list[i] for i in indices]
            agent_experience_list.append(agent_samples)

        return agent_experience_list