from typing import List, Dict
from marti.verifiers.auto_reward_alloc import MultiAgentRewardAllocation

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
    trajectories = sorted(trajectories, key=lambda traj: traj.get("prompt", ""))
    
    reward_alloc = MultiAgentRewardAllocation(
        verify=global_args.verify_task,
        **global_args.reward_alloc
    )
    
    local_rewards, outcome_rewards = reward_alloc.run(
        [traj["trajectory"] for traj in trajectories],
        [traj["label"] for traj in trajectories],
        n_samples_per_prompt=global_args.n_samples_per_prompt,
        answer_key="agent_output"
    )
    
    for traj_id, (traj, reward_matrix) in enumerate(zip(trajectories, local_rewards)):
        flattened_agent_data = []
        for agent_id, agent_data in enumerate(traj["trajectory"]):
            assert len(agent_data) == len(reward_matrix[agent_id]), (
                f"Length mismatch: agent {agent_id}, "
                f"{len(agent_data)} turns vs {len(reward_matrix[agent_id])} rewards"
            )
            for turn, reward in zip(agent_data, reward_matrix[agent_id]):
                turn["reward"] = reward
            flattened_agent_data.extend(agent_data)
        traj["trajectory"] = flattened_agent_data

    return trajectories

    # flattened_trajectory = []
    # for agent_idx, agent_turns in enumerate(trajectory):
    #     for turn in agent_turns:
    #         flattened_trajectory.append(turn)

    # # Initialize empty sample buckets for each agent
    # samples = [
    #     {"prompts": [], "outputs": [], "labels": []}
    #     for _ in range(num_agents)
    # ]

    # # Single pass: collect inputs, outputs, rewards per agent
    # for traj, reward_matrix in zip(trajectories, local_rewards):
    #     for agent_id, agent_data in enumerate(traj["trajectory"]):
    #         for turn, reward in zip(agent_data, reward_matrix[agent_id]):
    #             samples[agent_id]["prompts"].append(turn["agent_input"])
    #             samples[agent_id]["outputs"].append(turn["agent_output"])
    #             samples[agent_id]["labels"].append(reward)

    # return samples