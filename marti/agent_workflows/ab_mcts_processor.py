from typing import List, Dict
from marti.verifiers.auto_reward_alloc import MultiAgentRewardAllocation
import numpy as np
from collections import Counter
from copy import deepcopy


import numpy as np

def tree_reward_shaping_unified(
    trajectories,
    local_rewards,
    gamma=0.3,
    lambda_mix=0.5,
):
    B, T = local_rewards.shape

    for b, traj in enumerate(trajectories):
        nodeid2idx = {
            node["turn_id"]: t
            for t, node in enumerate(traj)
        }

        parent2children = {}
        for t, node in enumerate(traj):
            parent_id = node["metadata"].get("parent_idx", None)
            if parent_id is None:
                continue
            parent2children.setdefault(parent_id, []).append(t)

        for parent_id, child_indices in parent2children.items():
            # ---- get parent reward only if parent exists and is non-root ----
            parent_reward = None
            if parent_id != -1:
                parent_t = nodeid2idx.get(parent_id, None)
                if parent_t is not None:
                    parent_reward = local_rewards[b][parent_t]

            rewards = np.array([local_rewards[b][t] for t in child_indices])

            for t in child_indices:
                r = local_rewards[b][t]

                # sibling mean (leave-one-out)
                sib_mean = None
                if len(child_indices) > 1:
                    sib_mean = (rewards.sum() - r) / (len(child_indices) - 1)

                # ---- build baseline ----
                if parent_reward is not None and sib_mean is not None:
                    baseline = (1 - lambda_mix) * parent_reward + lambda_mix * sib_mean
                elif parent_reward is not None:
                    baseline = parent_reward
                elif sib_mean is not None:
                    baseline = sib_mean
                else:
                    continue  # no valid comparison

                local_rewards[b][t] += gamma * (r - baseline)

    return local_rewards

class MultiAgentMCTSRewardAllocation(MultiAgentRewardAllocation):
    def __init__(self,
                 verify="math",
                 name=None,
                 alpha=0.5, 
                 beta=0.5,
                 *args, **kwargs):
        print("Multi-agent Reward Allocation", locals())
        self.verify = verify
        self.name = name
        self.alpha = alpha
        self.beta = beta
        self.alpha_parent = alpha
        self.beta_sibling = beta
        # if verify == "math":
        #     self.reward_fn = qwen_reward_fn
        # elif verify == "math_format":
        #     self.reward_fn = qwen_reward_fn_format

        self.use_ttrl = kwargs.get("ttrl", False)

    def assign_rewards(self, histories, local_rewards, turn_ids=None):
        local_rewards = np.asarray(local_rewards, dtype=np.float32)

        # compute accuracy of final answer for chain / mixture
        # outcome_rewards = [prob_rewards[-1] for prob_rewards in local_rewards]
        # local_rewards = self._reward_shaping(local_rewards, turn_ids)
        local_rewards = tree_reward_shaping_unified(
            trajectories=histories,
            local_rewards=local_rewards,
            gamma=self.alpha,
            lambda_mix=self.beta,
            # alpha=self.alpha_parent,
            # beta=self.beta_sibling,
        )
        return local_rewards

    def _reward_shaping(self, local_rewards, turn_ids=None):
        if self.name in ["quality", "margin", "conditional_quality", "quality_with_outcome", "margin_with_outcome"]:
            from copy import deepcopy
            raw_rewards = deepcopy(local_rewards)
            M, T = raw_rewards.shape

            for m in range(M):
                # We need to start from turn 1, especially for mixture of agents
                # For chain / debate, turn is equal to turn_id
                start = 1
                if turn_ids is not None:
                    for turn, turn_id in enumerate(turn_ids[m]):
                        if turn_id == 1:
                            start = turn
                            break

                for t in range(start, T):
                    if turn_ids is None:
                        # compute all previous rewards for debate
                        Q = np.mean(raw_rewards[:, :t])
                    else:
                        Q = np.mean(raw_rewards[m][:t])

                    R_final = raw_rewards[m][t]
                    if "quality" in self.name:
                        dynamic_term = Q * R_final - (1-Q) * (1-R_final)
                        # print(f"{m} - {t}", Q, R_final, dynamic_term)
                        shaped_reward = R_final + self.alpha * dynamic_term
                    elif "margin" in self.name:
                        baseline_term = R_final - Q
                        shaped_reward = R_final + self.alpha * baseline_term
                    else:
                        raise NotImplementedError

                    local_rewards[m][t] = shaped_reward

            if "outcome" in self.name:
                for m in range(M):
                    for t in range(T):
                        local_rewards[m][t] += self.beta * raw_rewards[m][-1]

        return local_rewards

    def init_golden_answers_from_majority_voting(self, histories, batch_size):
        n_prompts = len(histories) // batch_size
        all_golden_answers = []
        for prompt_idx in range(n_prompts):
            all_outputs = histories[batch_size * prompt_idx:batch_size*(prompt_idx+1)]
            candidate_answers = []
            # (num_problems, num_agents, num_turns)
            for group_outputs in all_outputs:
                for agent_outputs in group_outputs:
                    if isinstance(agent_outputs, str):
                        candidate_answers.append(extract_answer(agent_outputs, "math"))
                    else:
                        for agent_output in agent_outputs:
                            assert isinstance(agent_output, str)
                            candidate_answers.append(extract_answer(agent_output, "math"))
            
            model_answers = [answer for answer in candidate_answers if answer is not None]

            counter = Counter(model_answers)
            majority_answer, _ = counter.most_common(1)[0]
            all_golden_answers.extend([majority_answer for _ in range(batch_size)])
        assert len(all_golden_answers) == len(histories)
        return all_golden_answers
        
    def run(self, histories, golden_answers, local_rewards=None, turn_ids=None, n_samples_per_prompt=None, answer_key="assistant"):
        """
        mcts
        """
        if self.use_ttrl:
            assert n_samples_per_prompt is not None, "`n_samples_per_prompt` should be provided for test-time RL"
            golden_answers = self.init_golden_answers_from_majority_voting(all_answers, n_samples_per_prompt)

        return self.assign_rewards(histories, local_rewards, turn_ids)


def flatten(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

def processor(
    trajectories: List[Dict],
    num_agents: int,
    global_args
) -> List[Dict[str, List]]:
    """
    Process collected trajectories into per-agent training samples.

    Args:
        trajectories: list of trajectory dicts (each has a "prompt" key plus per-turn items)
        num_agents: total number of agents
        sort_by_prompt: whether to group same-prompt trajectories together via a stable sort

    Returns:
        A list of length num_agents where each element is a dict with keys:
            "prompts", "outputs", "labels"
        Each is a list aligned by order.
    """
    # group same prompts together by sorting
    for traj in trajectories:
        traj["trajectory"] = sorted(traj['trajectory'], key=lambda t: t["turn_id"])

    trajectories = sorted(
        trajectories, key=lambda traj: int(traj.get("prompt_id", 0))
    )

    local_rewards = [[turn["reward"] for turn in traj["trajectory"]] for pid, traj in enumerate(trajectories)]
    turn_id = range(len(trajectories[0]["trajectory"]))
    turn_ids = [turn_id for _ in trajectories]

    reward_alloc = MultiAgentMCTSRewardAllocation(
        verify=global_args.verify_task,
        **global_args.reward_alloc
    )

    shaping_local_rewards = reward_alloc.run(
        [traj["trajectory"] for traj in trajectories],
        [traj["label"] for traj in trajectories],
        local_rewards = local_rewards,
        turn_ids = turn_ids,
        n_samples_per_prompt=global_args.n_samples_per_prompt,
        answer_key="agent_output",
    )
    
    for traj, reward_matrix in zip(trajectories, shaping_local_rewards):
        flattened_agent_data = []
        for turn_id, turn in enumerate(traj["trajectory"]):
            turn["reward"] = reward_matrix[turn_id]
            flattened_agent_data.append(turn)
        traj["trajectory"] = flattened_agent_data

    return trajectories