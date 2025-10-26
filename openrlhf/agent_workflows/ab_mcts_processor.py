from typing import List, Dict
from openrlhf.verifiers.auto_reward_alloc import MultiAgentRewardAllocation


class MultiAgentRewardAllocation_abmcts(MultiAgentRewardAllocation):
    def __init__(self,
                 verify="math",
                 name=None,
                 alpha=0.5, 
                 beta=0.5,
                 *args, **kwargs):
        super().__init__(verify, name, alpha, beta, *args, **kwargs)

    def assign_rewards_mcts(self, all_answers, golden_answers, method_rewards):
        local_rewards = [[] for _ in all_answers]
        outcome_rewards = []
        for pid, prob_answers in enumerate(all_answers):
            for aid, agent_answers in enumerate(prob_answers):
                ## **使用mcts过程中的reward替换agent reward 避免二次计算**
                agent_rewards = method_rewards[pid][aid]
                if not isinstance(agent_rewards, list):
                    agent_rewards = [agent_rewards]
                local_rewards[pid].append(agent_rewards)
        
        return local_rewards, outcome_rewards


    def _assign_rewards(self, all_answers, golden_answers, turn_ids=None, workflow_type="", method_rewards=None):   
        if turn_ids is None and workflow_type == "mcts":
            assert method_rewards is not None, f"mcts must have reward info from rollout"
            return self.assign_rewards_mcts(all_answers, golden_answers, method_rewards)
        else:
            return self.assign_rewards(all_answers, golden_answers, turn_ids)
        
    def run(self, histories, golden_answers, n_samples_per_prompt=None, answer_key="assistant", workflow_type=""):
        all_rewards=None
        # if isinstance(histories[0][0], dict):

        #     all_answers = [
        #         [agent[answer_key] for agent in agent_data]
        #         for agent_data in histories
        #     ]
        #     turn_ids = [
        #         [agent["turn_id"] for agent in agent_data]
        #         for agent_data in histories
        #     ]
        if workflow_type == "mcts":
            try:
                all_answers = [
                    [[node[answer_key] for node in agent] for agent in agent_data] 
                    for agent_data in histories
                ]
                all_rewards =[
                    [[node["agent_inout_score"] for node in agent] for agent in agent_data] 
                    for agent_data in histories
                ]
            except Exception as e:
                raise e
            turn_ids = None
        elif isinstance(histories[0][0], list):
            all_answers = [
                [[turn[answer_key] for turn in agent] for agent in agent_data]
                for agent_data in histories
            ]# num_agents * num_turns
            turn_ids = [
                [[turn["turn_id"] for turn in agent] for agent in agent_data]
                for agent_data in histories
            ]
        else:
            raise ValueError

        if self.use_ttrl:
            assert n_samples_per_prompt is not None, "`n_samples_per_prompt` should be provided for test-time RL"
            golden_answers = self.init_golden_answers_from_majority_voting(all_answers, n_samples_per_prompt)

        return self._assign_rewards(all_answers, golden_answers, turn_ids, workflow_type=workflow_type, method_rewards=all_rewards)

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
        for agent_idx, agent_data in enumerate(traj["trajectory"]):
            traj['trajectory'][agent_idx] = sorted(
                agent_data, key=lambda t: t['metadata'].get("expand_idx", 0)
            )

    trajectories = sorted(
        trajectories, key=lambda traj: int(traj.get("prompt_id", 0))
    )

    reward_alloc = MultiAgentRewardAllocation_abmcts(
        verify=global_args.verify_task,
        **global_args.reward_alloc
    )

    local_rewards, outcome_rewards = reward_alloc.run(
        [traj["trajectory"] for traj in trajectories],
        [traj["label"] for traj in trajectories],
        # rewards=[traj[""] for traj in trajectories],
        n_samples_per_prompt=global_args.n_samples_per_prompt,
        answer_key="agent_output",
        workflow_type="mcts"
    )
    
    # Initialize empty sample buckets for each agent
    samples = [
        {"prompts": [], "outputs": [], "labels": []}
        for _ in range(num_agents)
    ]

    # Single pass: collect inputs, outputs, rewards per agent
    for traj, reward_matrix in zip(trajectories, local_rewards):
        for agent_id, agent_data in enumerate(traj["trajectory"]):
            for turn, reward in zip(agent_data, reward_matrix[agent_id]):
                samples[agent_id]["prompts"].append(turn["agent_input"])
                samples[agent_id]["outputs"].append(turn["agent_output"])
                samples[agent_id]["labels"].append(reward)

    return samples