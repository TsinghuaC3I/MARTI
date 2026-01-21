# Multi-Agent Training

In this section, we present the implementation details of reward allocation and policy training for multi-agent training in MARTI. For reward allocation, we first discuss rule-based reward shaping, followed by AgentPRM for structured credit assignment, and finally generative reward models for open-domain applications. Policy training strategies are then introduced in the last.

## Rule-based Reward Shaping

For mathematical problems with verifiable solutions, we employ rule-based reward models such as DeepSeek-R1. This approach is particularly effective for mixture-of-agents and multi-agent debate scenarios, where each agent's output can be directly evaluated against the ground truth solution, enabling precise reward assignment based on predefined scoring rules.
To improve temporal consistency and leverage historical information in multi-turn interactions, we introduce an inference-aware reward shaping strategy from [MAPoRL](https://arxiv.org/abs/2502.18439). This method integrates past performance estimates with current rewards. 
Specifically, the approach combines an immediate correctness reward from a task verifier with a dynamic adjustment derived from the agent's historical performance. This historical performance is calculated as the average reward across previous interactions.


We implement two variants: 
1. a Quality Mode, which encourages consistency by aligning current performance with historical correctness, and 
2. a Margin Mode, which directly rewards agents for surpassing their historical average performance. Additionally, two historical evaluation scopes are provided: one considers only the most recent interaction, offering immediate but potentially variable feedback, while the other averages across all past interactions for more stable and reliable estimates. These modular and flexible strategies effectively reduce overfitting to single-turn outcomes, enhancing long-term collaboration effectiveness in multi-turn scenarios.

Let $R_t^i \in [0,1]$ denote the immediate correctness reward assigned by a task verifier for agent $i$ at turn $t$, and let $Q_t^i \in [0,1]$ represent the historical performance estimate of the agent, computed over a set of previous interactions:

$$Q_t^i = \frac{1}{|H_t^i|} \sum_{k \in H_t^i} R_k^i$$

where $\mathcal{H}_t^i \subset \{1, \dots, t-1\}$ denotes the historical evaluation scope (e.g., most recent round or all previous rounds). We define the dynamic shaping term $\Delta_t^i$ under two modes:

- Margin Mode: $\Delta_t^i = R_t^i - Q_t^i$
- Quality Mode: $\Delta_t^i = Q_t^i \cdot R_t^i - (1 - Q_t^i)(1 - R_t^i)$

The final shaped reward $R_t^i$ is then given by: 
$$R_t^i = R_t^i + \alpha \cdot \Delta_t^i$$
where $\alpha \in R_{\ge 0}$ is a tunable hyperparameter controlling the influence of historical consistency.

## Tree-based AgentPRM

In scenarios where the final answer does not appear in intermediate agents' outputs and long-term interactions are involved, outcome-level rule-based reward shaping proves ineffective. Recent studies demonstrate that process reward models can provide fine-grained supervision for mathematical and multi-agent problems. Inspired by [SweetRL](https://github.com/facebookresearch/sweet_rl) and [PRIME](https://github.com/PRIME-RL/PRIME), we therefore introduce AgentPRM in MARTI.

Specifically, we adapt the existing token-level [Implicit Process Reward Model (ImplicitPRM)](https://github.com/PRIME-RL/ImplicitPRM) to an agent-level PRM through the following methodology:

- Agent-level Reward Calculation. For each agent action composed of L tokens ( $a_t=(a_t^1,\cdots,a_t^L)$ ), we compute the agent-level reward $A_t$ by averaging the log-likelihood ratios of generated tokens under the current policy $\pi_\theta$ versus a reference policy $\pi_{\text{ref}}$, given the context $c$ and preceding tokens:

$$A_t = \frac{1}{L} \sum_{l=1}^L \left[ \log \frac{\pi_\theta(a_t^l | a_t^{1:l-1}, c)}{\pi_{\text{ref}} (a_t^l | a_t^{1:l-1}, c)} \right]$$

- Cross-Entropy Loss Calculation. We aggregate agent-level rewards ($A_t$) from all $N$ agents to compute the cross-entropy loss $\mathcal{L}_{\text{CE}}$ using the true label $y$ (where $y\in {0,1}$):

$$L_{\text{CE}} = y \cdot \sigma\left( \sum_{t=1}^N \beta \cdot A_t\right) + (1-y) \cdot \left(1 - \sigma\left(\sum_{t=1}^N \beta \cdot A_t\right)\right)$$

Here, $\sigma(\cdot)$ denotes the sigmoid function and $\beta$ serves as a reward weighting factor.

Following SweetRL's approach, we implement tree-based rollouts for multi-turn trajectories, sampling multiple candidate responses per agent at each turn. The AgentPRM is then trained using process-pairs similar to DPO training in SweetRL.


## Generative Reward Models

Recent advances have demonstrated that LLMs can effectively evaluate response quality, enabling their use as generative reward models (GenRMs) to enhance policy model reasoning capabilities. Building on these developments, we implement GenRMs in MARTI for both verifiable and open-domain problems. Our framework supports GenRMs through either local vLLM engines or OpenAI-compatible APIs, with a defined GenRM that assigns scalar rewards to given problem-trajectory pairs.

Furthermore, we investigate specialized GenRMs for multi-agent systems (MAS) that explicitly address common failure modes identified in prior work. These models show particular promise for improving collaborative behaviors in MAS. We continue to optimize this functionality, with further discussion reserved for future work.

## Policy Model Training

Upon obtaining rollout experiences comprising individual trajectories and corresponding rewards for each agent, we initiate distributed training of agent policy models. The training leverages adapted implementations from OpenRLHF, supporting various reinforcement learning algorithms including REINFORCE++, GRPO, and PPO. Notably, all agent policies are trained using identical RL algorithms to maintain consistency.

Furthermore, we augment the training process by incorporating additional imitation learning strategies during on-policy rollouts. These include supervised fine-tuning (SFT) and direct preference optimization (DPO), extending beyond OpenRLHF's native capabilities. This integration enables dynamic selection of training strategies tailored to specific application requirements, such as stable training and faster convergence.
