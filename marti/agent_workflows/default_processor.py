from typing import List, Dict

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

    # Initialize empty sample buckets for each agent
    samples = [
        {"prompts": [], "outputs": [], "labels": []}
        for _ in range(num_agents)
    ]

    # Single pass: collect inputs, outputs, rewards per agent
    for traj in trajectories:
        for turn in traj["trajectory"]:
            idx = turn["agent_index"]
            samples[idx]["prompts"].append(turn["agent_input"])
            samples[idx]["outputs"].append(turn["agent_output"])
            samples[idx]["labels"].append(turn["agent_reward"])

    return samples