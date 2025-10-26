import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import timedelta
from typing import Any, List, Tuple, Union

import ray
import torch

from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions
from openrlhf.utils.utils import remove_pad_token, zero_pad_sequences
from openrlhf.trainer.ppo_utils.experience_maker import Experience, RemoteExperienceMaker

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
        # tokenizer=None,
        tokenizer_list=None,# 需要对应不同agent 的tokenizer
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
        self.tokenizer_list = tokenizer_list
        self.groups = None
        
        # 初始化可能需要的模型组（用于共享的 critic 和 reward model）
        self.critic_model_group = kwargs.get('critic_model_group', None)
        self.reward_model_group = kwargs.get('reward_model_group', None)

    def split_rollout_samples(self, rollout_samples):
        for i, sample in enumerate(rollout_samples):
            sample.index = [i]

        samples_list = []
        if self.args.use_dynamic_batch:
            assert False, f"当前不支持动态batch"
        # TODO 检查多智能体是否支持use_dynamic_batch，大概率不支持
            # total_lengths = [int(s.info["total_length"].item()) for s in rollout_samples]
            # effective_actor_num = (
            #     self.args.actor_num_nodes
            #     * self.args.actor_num_gpus_per_node
            #     // self.args.ring_attn_size
            #     // self.args.ds_tensor_parallel_size
            # )
            # minimum_batch_num = get_minimum_num_micro_batch_size(
            #     total_lengths,
            #     self.args.rollout_max_tokens_per_gpu,
            #     self.args.ring_attn_size,
            #     self.args.ds_tensor_parallel_size,
            # )
            # minimum_batch_num = minimum_batch_num // effective_actor_num * effective_actor_num
            # num_batch = max(minimum_batch_num, effective_actor_num)
            # batch_indexes = get_seqlen_balanced_partitions(total_lengths, num_batch, False)
            # for micro_index in batch_indexes:
            #     micro_batch = [rollout_samples[idx] for idx in micro_index]
            #     concat_samples = Experience.concat_experiences(micro_batch, self.tokenizer.pad_token_id)
            #     samples_list.append(concat_samples)
        else:
            batch_size = self.args.micro_rollout_batch_size
            assert self.args.micro_rollout_batch_size == 1, f"目前无法处理不为1的情况，可能涉及不同agent exprience的concat，会报错"
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

        # ---- 提取样本和 agent_id（**新增**） ----
        # sequences_list / masks 按 samples_list 顺序存放（与原逻辑一致）
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list]

        # 尝试读取 agent id：支持 "agent_id" 或 "agent_index"，否则默认 0
        agent_ids = []
        for s in samples_list:
            if "agent_id" in s.info:
                agent_ids.append(int(s.info["agent_id"]))
            elif "agent_index" in s.info:
                agent_ids.append(int(s.info["agent_index"]))
            else:
                agent_ids.append(0)

        # 确保 rewards 已经填好（跟原逻辑一致）
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

        # ---- 分组：按 agent_id 收集对应样本的索引与 inputs （**新增**） ----
        # groups: agent_id -> list of indices in samples_list
        groups = {}
        for idx, aid in enumerate(agent_ids):
            groups.setdefault(aid, []).append(idx)
        self.groups = groups
        # 为每个 agent 分别发起 actor/initial 的 batch 请求，保存 refs（**新增**）
        actor_action_log_probs_ref_per_agent = {}
        base_action_log_probs_ref_per_agent = {}
        # 如果 critic 是 shared 的，按原逻辑统一调用（否则可按 agent 同样分组）
        # 这里保持 critic/shared reward 的原始调用（仅 actor/initial 按 agent 分组）
        # 先构造 per-agent 输入并调用异步接口
        for aid, indices in groups.items():
            # 获取对应 agent 的 model groups
            try:
                actor_group, initial_group = self.agent_list[aid].actor_model_group, self.agent_list[aid].ref_model_group
            except Exception as e:
                raise ValueError(f"agent id {aid} not found in self.agent_list") from e

            # 提取该 agent 对应的输入子集（按原 samples_list 的顺序）
            sub_sequences = [sequences_list[i] for i in indices]
            sub_action_mask = [action_mask_list[i] for i in indices]
            sub_attention_mask = [attention_mask_list[i] for i in indices]

            # 调用 actor 的 batch forward（异步）
            actor_action_log_probs_ref_per_agent[aid] = actor_group.async_run_method_batch(
                method_name="forward",
                sequences=sub_sequences,
                action_mask=sub_action_mask,
                attention_mask=sub_attention_mask,
            )

            # Sync to avoid GPU OOM when colocate models
            if args.colocate_all_models or args.colocate_actor_ref: 
                ray.get(actor_action_log_probs_ref_per_agent[aid])  # TODO check
                ray.get(actor_group.async_run_method(method_name="empty_cache"))

            # 调用该 agent 的 initial model（若存在）
            if initial_group is not None:
                base_action_log_probs_ref_per_agent[aid] = initial_group.async_run_method_batch(
                    method_name="forward",
                    sequences=sub_sequences,
                    action_mask=sub_action_mask,
                    attention_mask=sub_attention_mask,
                )
                if args.colocate_all_models or args.colocate_actor_ref:
                    ray.get(base_action_log_probs_ref_per_agent[aid])  # TODO check
                    ray.get(initial_group.async_run_method(method_name="empty_cache"))
            else:
                base_action_log_probs_ref_per_agent[aid] = ray.put([[None]] * len(indices))

        # ---- critic （shared） 调用（保留原逻辑） ----
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

        # ---- 处理 colocation / sync：等待 per-agent actor/initial refs 完成 ----
        # 同原逻辑，考虑 duplicate_factor。先等待所有 actor/initial refs 完成
        duplicate_factor = args.ring_attn_size * args.ds_tensor_parallel_size

        # 收集所有 actor refs value 到单个列表（按 samples_list 顺序还原）
        # 1) 取得每个 agent 的返回并展开（注意 remote actor 返回的是按其内部顺序的 list）
        # ray.get 会返回 per-agent 的 list-of-lists（可能因为 ring/TP 重复），下面按 duplicate_factor 取样
        # 等待 critic
        ray.get(value_ref)
        value_list = sum(ray.get(value_ref)[::duplicate_factor], [])

        # 取得每个 agent 的 actor & base 返回，按 indices 放回到 action_log_probs_list / base_action_log_probs_list
        # 初始化占位
        action_log_probs_list = [None] * len(samples_list)
        base_action_log_probs_list = [None] * len(samples_list)

        # 遍历每个 agent：取结果并放回原始位置
        for aid, indices in groups.items():
            actor_raw = ray.get(actor_action_log_probs_ref_per_agent[aid])  # TODO check
            actor_expanded = sum(actor_raw[::duplicate_factor], [])  # 扁平化并去重复制
            if len(actor_expanded) != len(indices):
                # 容错：如果长度不匹配，抛出明确错误信息
                raise RuntimeError(
                    f"Actor returned {len(actor_expanded)} outputs for agent {aid}, expected {len(indices)}"
                )

            base_raw = ray.get(base_action_log_probs_ref_per_agent[aid])  # TODO check
            base_expanded = sum(base_raw[::duplicate_factor], [])
            if len(base_expanded) != len(indices):
                # 如果 initial_group 为 None，会通过 ray.put 的 [[None]] 来保证长度一致
                # 但仍做检查
                raise RuntimeError(
                    f"Initial model returned {len(base_expanded)} outputs for agent {aid}, expected {len(indices)}"
                )

            # 把该 agent 的结果按 indices 放回到全局列表（保持 samples_list 顺序）
            for pos_in_group, sample_idx in enumerate(indices):
                action_log_probs_list[sample_idx] = actor_expanded[pos_in_group]
                base_action_log_probs_list[sample_idx] = base_expanded[pos_in_group]

        # 最后检查没有 None（确保每个 sample 都有对应输出）
        assert None not in action_log_probs_list, "Some action logprobs missing after per-agent calls"
        assert None not in base_action_log_probs_list, "Some base action logprobs missing after per-agent calls"
        #logger.warning(f"sample_list_111: {samples_list}")
        # ---- rewards 处理：保持原逻辑 ----
        if samples_list[0].rewards is not None:
            pass
        elif self.remote_rm_url:
            # Get rewards info from remote model （如果你的 reward 也按 agent 分配，这里也需要改）
            rewards_info = ray.get(r_refs)
            update_samples_with_rewards(rewards_info, samples_list)
        else:
            rewards_list = sum(ray.get(r_refs)[::duplicate_factor], [])
            for i, samples in enumerate(samples_list):
                samples.rewards = rewards_list[i]
                samples.info["reward"] = rewards_list[i]

        # ---- 原有一致性断言（保持） ----
        assert (
            len(samples_list) == len(action_log_probs_list) == len(base_action_log_probs_list) == len(value_list)
        ), f"len(samples_list): {len(samples_list)}, len(action_log_probs_list): {len(action_log_probs_list)}, len(base_action_log_probs_list): {len(base_action_log_probs_list)}, len(value_list): {len(value_list)}"

        # ---- 按 sample 更新 experience（与原逻辑一致） ----
        for i, (samples, action_log_probs, base_action_log_probs, value) in enumerate(
            zip(samples_list, action_log_probs_list, base_action_log_probs_list, value_list)
        ):
            if (base_action_log_probs is not None) and (not args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.strategy.args.kl_estimator,
                )
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
        根据 self.groups 信息提取每个 agent 对应的 Experience 列表。

        Args:
            samples_list (List[Experience]): 所有样本组成的列表
        Returns:
            List[List[Experience]]: 每个 agent 对应的样本子列表，顺序与 agent_id 排序一致
        """
        if not hasattr(self, "groups") or self.groups is None:
            raise ValueError("self.groups 未初始化，请在 make_experience 调用后使用该函数。")

        agent_experience_list = []
        # 让 agent_id 按升序排列，以保持顺序稳定（可选）
        for aid in sorted(self.groups.keys()):
            indices = self.groups[aid]
            # 提取对应样本
            agent_samples = [samples_list[i] for i in indices]
            agent_experience_list.append(agent_samples)

        return agent_experience_list