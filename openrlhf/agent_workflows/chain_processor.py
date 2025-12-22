from typing import List, Dict
from openrlhf.verifiers.auto_reward_alloc import MultiAgentRewardAllocation

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
        for turn_id, turn in enumerate(traj["trajectory"]):
            turn["reward"] = reward_matrix[turn_id]
            flattened_agent_data.append(turn)
        traj["trajectory"] = flattened_agent_data

    return trajectories