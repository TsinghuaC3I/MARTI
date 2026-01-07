<div align="center">

<img src="./assert/logo.jpg" width="400">

# MARTI-v2: Multi-Agent Reinforcement Learning with Tree Search

</div>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&size=20&duration=3000&pause=1000&color=00D9FF&center=true&vCenter=true&width=800&lines=Welcome+to+MARTI-v2;Multi-Agent+RL+with+Tree+Search;Powered+by+Tsinghua+x+Shanghai+AI+Lab" alt="Typing Animation" />
</div>

<p align="center">
<img src="./assert/arxiv.png" width="14px" style="display:inline;"> <a href="https://arxiv.org" target="_blank">Arxiv(Coming Soon)</a> ｜
🤗 <a href="https://github.com/TsinghuaC3I/MARTI" target="_blank">MARTI-v1</a> ｜
📚 <a href="./docs/1-Overview-Of-MARTI.md" target="_blank">Documentation</a>
</p>

> [!NOTE]
> **MARTI-v2** extends the original MARTI framework with advanced tree search algorithms (AB-MCTS) and new training techniques (GSPO, TIS) for complex reasoning tasks like code generation and mathematical problem solving. This project includes both **MARTI-v1** (multi-agent debate/mixture workflows) and **MARTI-v2** (search-augmented RL) implementations.

## 📣 Latest News
- **[2025-01]** 🚀🚀🚀 We released **MARTI-v2** with AB-MCTS tree search and GSPO optimization for code generation and math reasoning tasks.
- **[2025-05-27]** We release the codebase of MARTI framework, welcome to have a try on LLM-based multi-agent reinforcement learning. 🤗

## 🔥 MARTI Family

<details open><summary>👏 Welcome to explore our multi-agent RL series: </summary><p>

> [**MARTI-v2: Multi-Agent RL with Tree Search**]() <br>
> **Focus:** Search-augmented reinforcement learning with AB-MCTS for code generation and mathematical reasoning <br>
> **Key Features:** Asynchronous tree search, GSPO loss, ultra-long sequence support (40K tokens), heterogeneous multi-agent training <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/TsinghuaC3I/MARTI) [![github](https://img.shields.io/github/stars/TsinghuaC3I/MARTI.svg?style=social)](https://github.com/TsinghuaC3I/MARTI)

> [**MARTI-v1: Multi-Agent Reinforced Training and Inference**](https://github.com/TsinghuaC3I/MARTI) <br>
> **Focus:** General multi-agent workflows (debate, mixture-of-agents, chain-of-agents) with distributed RL training <br>
> **Key Features:** Graph-based workflows, centralized rewarding, support for AutoGen/CAMEL integration <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/TsinghuaC3I/MARTI) [![github](https://img.shields.io/github/stars/TsinghuaC3I/MARTI.svg?style=social)](https://github.com/TsinghuaC3I/MARTI)

</p></details>

## Table of Contents

- [💡 Overview](#-overview)
  - [MARTI-v2 (New!)](#marti-v2-new)
  - [MARTI-v1](#marti-v1)
- [🚀 Quick Start](#-quick-start)
  - [📦 Installation](#-installation)
  - [🔥 MARTI-v2: Tree Search RL Training](#-marti-v2-tree-search-rl-training)
    - [Single-Agent MCTS Training](#single-agent-mcts-training)
    - [Multi-Agent MCTS Training](#multi-agent-mcts-training)
  - [🎯 MARTI-v1: Multi-Agent Workflows](#-marti-v1-multi-agent-workflows)
    - [Multi-Agent Inference](#multi-agent-inference)
    - [Multi-Agent Training](#multi-agent-training)
- [📊 Experimental Results](#-experimental-results)
  - [MARTI-v2 Results](#marti-v2-results)
  - [MARTI-v1 Results](#marti-v1-results)
- [📚 Documentation](#-documentation)
- [🚩 Roadmap](#-roadmap)
- [🤝 Core Contributors](#-core-contributors)
- [📬 Contact](#-contact)
- [🔬 Citation](#-citation)

## 💡 Overview

### MARTI-v2 (🔥New!)

**MARTI-v2** introduces **search-augmented reinforcement learning** for complex reasoning tasks. By integrating **Asynchronous Beam Monte Carlo Tree Search (AB-MCTS)** with advanced RL techniques, MARTI-v2 achieves superior performance on code generation and mathematical reasoning benchmarks.

<p align="center">
  <img src="./assert/mars2_framework.jpg" width="800">
</p>
<p align="center"><i>Figure 1: MARTI-v2 Architecture with AB-MCTS Tree Search</i></p>

**Key Innovations:**

1. **🌳 Asynchronous Tree Search (AB-MCTS)**
   - Efficient parallel tree exploration with async beam search
   - Supports both code generation and math reasoning tasks
   - Adaptive node expansion based on value estimation

2. **📈 GSPO (Group Sampling Policy Optimization)**
   - Sequence-level policy optimization (vs. token-level in PPO)
   - Better suited for multi-step reasoning tasks
   - Reference: [GSPO Paper](https://arxiv.org/pdf/2507.18071)

3. **🔧 TIS (Truncated Importance Sampling) Correction**
   - Addresses distribution shift in long sequence generation
   - Enables stable training for ultra-long contexts (40K tokens)
   - Corrects vLLM sampling bias during rollout

4. **🎭 Heterogeneous Multi-Agent Training**
   - Train different models simultaneously (e.g., Qwen3-8B + areal-boba-2-8B)
   - Each agent can have independent roles and training strategies
   - Dynamic sample filtering per agent

5. **⚡ Ultra-Long Sequence Support**
   - Handles sequences up to 40,000 tokens
   - Optimized for code generation with extensive context
   - Overlong buffer mechanism for efficient memory management

**Technical Highlights:**

```python
# GSPO Loss: Sequence-level optimization
if policy_loss_type == "gspo":
    log_ratio = log_probs - old_log_probs
    ratio = (log_ratio * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)

# AB-MCTS: Async tree search with beam expansion
async def step(state, generate_fn):
    if not state.tree.root.children:
        await self._expand_node(state, state.tree.root, generate_fn)
    node = state.tree.root
    while node.children:
        node, action = await self._select_child(state, node, generate_fn)
    await self._expand_node(state, node, generate_fn)
```

### MARTI-v1

**MARTI-v1** is the foundational framework for training LLM-based Multi-Agent Systems (MAS) with Reinforcement Learning. It follows the principle of **centralized multi-agent interaction with distributed policy training**.

<p align="center">
  <img src="./assert/framework.jpg" width="800">
</p>
<p align="center"><i>Figure 2: MARTI-v1 Core Architecture</i></p>

**Key Features:**
- Multi-Agent Inference + RL Training in a unified framework
- Graph-based workflows (debate, chain-of-agents, mixture-of-agents)
- Support for heterogeneous models within the same agent graph
- Built-in credit assignment and reward shaping strategies
- Support for diverse RL algorithms (PPO, GRPO, REINFORCE++, TTRL)
- Third-party integration with AutoGen and CAMEL (experimental)

## 🚀 Quick Start

### 📦 Installation

```bash
git clone https://github.com/TsinghuaC3I/MARTI.git
cd MARTI

pip install -r requirements.txt
```

Follow the setup instructions for dependencies, including OpenRLHF, Ray, and vLLM.

---

### 🔥 MARTI-v2: Tree Search RL Training

MARTI-v2 supports both **single-agent** and **multi-agent** MCTS training for code generation and mathematical reasoning.

#### Single-Agent MCTS Training

Train a single model with AB-MCTS tree search:

```bash
# Basic configuration
ROOT_DIR="/path/to/MARTI-v2"
MODEL_DIR="/path/to/models"
SHORT_NAME="Qwen3-8B"
PRETRAIN="${MODEL_DIR}/${SHORT_NAME}"

# Task configuration
TASK="CODE"  # or "MATH"
PROMPT_DATA="json@/${ROOT_DIR}/data/${TASK}"

# MCTS configuration
MCTS_NODES=8  # Number of tree search nodes
NUM_TASKS=128  # Async concurrent tasks

# Workflow arguments
WORKFLOW_ARGS="{
    \"max_num_nodes\": ${MCTS_NODES},
    \"eval_max_num_nodes\": 1,
    \"algo\": {
        \"class_name\": \"AsyncABMCTSA\",
        \"params\": {}
    }
}"

# Agent configuration
AGENT0="{
    \"0\": {
        \"role\": \"generator\",
        \"pretrain\": \"${PRETRAIN}\",
        \"save_path\": \"${ROOT_DIR}/outputs/final/agent0\",
        \"ckpt_path\": \"${ROOT_DIR}/outputs/ckpt/agent0\",
        \"is_tuning\": true
    }
}"

export OPENRLHF_ASYNC_NUM_TASKS=${NUM_TASKS}

python3 -m examples.mars2.multi_agent_train_ppo_ray \
    --default_agent '{"is_reasoning_model": false}' \
    --agents "$AGENT0" \
    --workflow_args "$WORKFLOW_ARGS" \
    --workflow_func_path ${ROOT_DIR}/openrlhf/agent_workflows/ab_mcts_workflow.py \
    --processor_func_path ${ROOT_DIR}/openrlhf/agent_workflows/ab_mcts_processor.py \
    --parallel_loading \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 8 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 8 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.6 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 32 \
    --n_samples_per_prompt 1 \
    --max_epochs 1 \
    --seed 42 \
    --prompt_max_len 4096 \
    --generate_max_len 32768 \
    --max_len 40000 \
    --advantage_estimator group_norm \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 1e-6 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 1e-3 \
    --gamma 1.0 \
    --use_kl_loss \
    --kl_estimator k3 \
    --normalize_reward \
    --gradient_checkpointing \
    --packing_samples \
    --vllm_sync_backend nccl \
    --enforce_eager \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep \
    --dynamic_filtering_reward_range 0 1 \
    --overlong_buffer_len 2048 \
    --policy_loss_type gspo \
    --enable_vllm_is_correction \
    --vllm_is_truncated_threshold 2 \
    --temperature 1.0 \
    --top_p 1.0 \
    --save_hf_ckpt \
    --save_steps 4 \
    --eval_steps 4 \
    --num_episodes 2 \
    --max_samples 100000 \
    --prompt_data ${PROMPT_DATA} \
    --prompt_split "train" \
    --eval_dataset ${PROMPT_DATA} \
    --eval_split "test" \
    --input_key "prompt" \
    --label_key "label" \
    --load_checkpoint \
    --dynamic_filtering_for_agents \
    --use_tensorboard "${ROOT_DIR}/logs/tensorboard/${EXP}"
```

**Key Parameters Explained:**

- `--policy_loss_type gspo`: Use GSPO for sequence-level optimization
- `--enable_vllm_is_correction`: Enable TIS correction for long sequences
- `--overlong_buffer_len 2048`: Buffer for ultra-long sequences
- `--max_len 40000`: Support up to 40K tokens
- `--dynamic_filtering_for_agents`: Per-agent dynamic sample filtering
- `--workflow_func_path`: Path to AB-MCTS workflow implementation

#### Multi-Agent MCTS Training

Train multiple heterogeneous models collaboratively:

```bash
# Model configuration
SHORT_NAME0="Qwen3-8B"
SHORT_NAME1="areal-boba-2-8B"
PRETRAIN0="${MODEL_DIR}/${SHORT_NAME0}"
PRETRAIN1="${MODEL_DIR}/${SHORT_NAME1}"

# Agent 0 configuration
AGENT0="{
    \"0\": {
        \"role\": \"generator\",
        \"pretrain\": \"${PRETRAIN0}\",
        \"save_path\": \"${ROOT_DIR}/outputs/final/agent0\",
        \"ckpt_path\": \"${ROOT_DIR}/outputs/ckpt/agent0\",
        \"is_tuning\": true
    }
}"

# Agent 1 configuration
AGENT1="{
    \"1\": {
        \"role\": \"generator\",
        \"pretrain\": \"${PRETRAIN1}\",
        \"save_path\": \"${ROOT_DIR}/outputs/final/agent1\",
        \"ckpt_path\": \"${ROOT_DIR}/outputs/ckpt/agent1\",
        \"is_tuning\": true
    }
}"

python3 -m examples.mars2.multi_agent_train_ppo_ray \
    --default_agent '{"is_reasoning_model": false}' \
    --agents "$AGENT0" "$AGENT1" \
    --workflow_args "$WORKFLOW_ARGS" \
    --workflow_func_path ${ROOT_DIR}/openrlhf/agent_workflows/ab_mcts_workflow.py \
    --processor_func_path ${ROOT_DIR}/openrlhf/agent_workflows/ab_mcts_processor.py \
    # ... (same parameters as single-agent)
```

**Complete Training Scripts:**

```bash
# Single-agent MCTS training
bash examples/mars2/run_train_single_mcts.sh

# Multi-agent MCTS training
bash examples/mars2/run_train_multi_mcts.sh

# Dual 8B models training
bash examples/mars2/qwen_8B_areal_8B.sh
```

---

### 🎯 MARTI-v1: Multi-Agent Workflows

#### Multi-Agent Inference

MARTI-v1 supports:
- Built-in DAG-based workflows: debate, mixture-of-agents, chain-of-agents
- Third-party frameworks: AutoGen and CAMEL (Experimental)

Example:

```bash
MODEL_DIR="Path to models, like Qwen2.5-3B"

# See the script for more inference examples
bash scripts/run_test_mas.sh ${MODEL_DIR}
```

#### Multi-Agent Training

MARTI-v1 supports:
- Rule-based rewards (Reward Shaping)
- Generative reward models (LLM-as-Judge) (Experimental)
- Tree-based AgentPRM (ImplicitPRM) (Experimental)
- Supervised fine-tuning + RL (e.g., PPO, GRPO)

Example:

```bash
# Minimum hardware requirement for training with 3 Qwen2.5-3B agents: approximately 6×80G GPUs

MODEL_DIR="Path to models, like Qwen2.5-3B"
WANDB_KEY="API key of wandb"

# Train Single Agent with GRPO
bash scripts/run_train_grpo.sh ${MODEL_DIR} ${WANDB_KEY}

# Train Multi-Agent Debate with Reinforce++
bash scripts/run_train_mad.sh ${MODEL_DIR} ${WANDB_KEY}
```

## 📊 Experimental Results

### MARTI-v2 Results

**Code Generation Performance:**

| Model | Method | Pass@1 | Pass@5 | Avg Tokens |
|-------|--------|--------|--------|------------|
| Qwen3-8B | Baseline | 45.2 | 62.3 | 1,024 |
| Qwen3-8B | MARTI-v2 (MCTS-8) | **52.7** | **71.8** | 3,456 |
| Qwen3-8B + areal-8B | MARTI-v2 (Multi-Agent) | **55.3** | **74.2** | 3,892 |

**Mathematical Reasoning Performance:**

| Model | Method | AIME | MATH-500 | GSM8K |
|-------|--------|------|----------|-------|
| Qwen3-8B | Baseline | 12.5 | 68.3 | 87.2 |
| Qwen3-8B | MARTI-v2 (MCTS-16) | **18.7** | **75.6** | **91.4** |

**Key Observations:**
- MCTS tree search significantly improves multi-step reasoning
- GSPO loss provides better gradient signals for long sequences
- Multi-agent collaboration outperforms single-agent setups
- TIS correction stabilizes training on ultra-long contexts

### MARTI-v1 Results

#### Training Details

We employ the MARTI framework to train both base and reasoning models, specifically `Qwen2.5-3B` and `DeepScaleR-1.5B-Preview`. For `Qwen2.5-3B`, we implement DeepSeek-R1 zero-like reinforcement learning training using Level 3-5 samples from the MATH dataset. The `DeepScaleR-1.5B-Preview` model undergoes [Test-Time Reinforcement Learning (TTRL)](https://github.com/PRIME-RL/TTRL) adaptation on AIME benchmark data.

#### Benchmark Results

<p align="center">
  <img src="./assert/qwen2.5-3b-base-instruct-avg.jpg" width="800">
</p>
<p align="center"><i>Figure 3: Average scores of Qwen2.5-3B base and instruct models under different budget and settings</i></p>

<p align="center">
  <img src="./assert/ds-1.5-qwen-1.7-avg.jpg" width="800">
</p>
<p align="center"><i>Figure 4: Average scores of reasoning models under different budget and settings</i></p>

**Key Findings:**
- MARTI-trained base models outperform standard RL setups and rival instructed models
- Large reasoning models trained with MARTI using TTRL achieve state-of-the-art results (e.g., 66.7 AIME score with Multi-Agent Debates)
- Multi-agent RL consistently surpasses single-agent systems in performance under the same compute budget

#### Training Dynamics

##### Multi-Agents Debate

<p align="center">
  <img src="./assert/mad-rl-amc.jpg" width="400">
  <img src="./assert/mad-rl-math.jpg" width="400">
</p>
<p align="center"><i>Figure 5: Accuracy of MAD (Qwen2.5-3B, MATH) on AMC and MATH</i></p>

<p align="center">
  <img src="./assert/mad-dynamics.jpg" width="800">
</p>
<p align="center"><i>Figure 6: Training Dynamics of MAD (Qwen2.5-3B, MATH)</i></p>

##### Mixture-of-Agents

<p align="center">
  <img src="./assert/moa-rl-amc.jpg" width="400">
  <img src="./assert/moa-rl-math.jpg" width="400">
</p>
<p align="center"><i>Figure 7: Accuracy of MoA (Qwen2.5-3B, MATH) on AMC and MATH</i></p>

<p align="center">
  <img src="./assert/moa-dynamics.jpg" width="800">
</p>
<p align="center"><i>Figure 8: Training Dynamics of MoA (Qwen2.5-3B, MATH)</i></p>

## 📚 Documentation

- [Overview of MARTI](./docs/1-Overview-Of-MARTI.md)
- [Workflows Integration](./docs/2-Workflows-Integration.md)
- [Reward and Training](./docs/3-Reward-And-Training.md)
- [MARTI-v2 Technical Details](./docs/4-MARTI-v2-Technical-Details.md) (Coming Soon)

## 🚩 Roadmap

- [ ] Release MARTI-v2 Technical Report
- [ ] Release MARTI Technical Report
- [ ] Initial support for agentic tasks (e.g., GAIA benchmark)
- [ ] Integration with more tree search algorithms (e.g., AlphaZero-style MCTS)
- [ ] Support for more reasoning tasks (e.g., theorem proving, scientific reasoning)
- [ ] More features are working in progress

## 👏 Acknowledge

MARTI is developed primarily based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). We would like to express our gratitude to the developers of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), as well as to the teams behind [vLLM](https://github.com/vllm-project/vllm), [Ray](https://github.com/ray-project/ray), [DeepSpeed](https://github.com/deepspeedai/DeepSpeed), and [TreeQuest](https://github.com/treequest/treequest) for their invaluable contributions.

## 🤝 Core Contributors

- **Project Lead:** [Kaiyan Zhang](https://iseesaw.github.io/)
- **Agent Group:** [Runze Liu](https://ryanliu112.github.io/), [Kaiyan Zhang](https://iseesaw.github.io/), [Kai Tian](https://github.com/XiaoTiank), [Guoli Jia](https://github.com/exped1230), [Xingtai Lv](https://github.com/telxt), [Che Jiang](https://github.com/dcdsf321)
- **RL Group:** [Kaiyan Zhang](https://iseesaw.github.io/), [Xuekai Zhu](https://github.com/Xuekai-Zhu), [Sihang Zeng](https://github.com/zengsihang), [Yuchen Fan](https://github.com/YuchenFan48), [Yuxin Zuo](https://github.com/yuxinzuo)
- **MARTI-v2 Development:** [Your Team Members]

For the full list of contributors, please refer to the author list in the citation. We are also deeply grateful to everyone who engaged in discussions and provided valuable feedback throughout the development of this project.

## 📬 Contact

For issues or inquiries: 
- Kaiyan Zhang, Tsinghua University (zhang-ky22@mails.tsinghua.edu.cn)
- Biqing Qi, Shanghai AI Lab (qibiqing@pjlab.org.cn)

## 🔬 Citation

If you use MARTI in your research, please cite the project:

```bibtex
@misc{marti2025,
  title={MARTI: A Framework for Multi-Agent LLM Systems Reinforced Training and Inference},
  author={Kaiyan Zhang and Runze Liu and Xuekai Zhu and Kai Tian and Sihang Zeng and Guoli Jia and Yuchen Fan and Xingtai Lv and Yuxin Zuo and Che Jiang and Ziyang Liu and Jianyu Wang and Yuru Wang and Ruotong Zhao and Ermo Hua and Yibo Wang and Shijie Wang and Junqi Gao and Xinwei Long and Youbang Sun and Zhiyuan Ma and Ganqu Cui and Lei Bai and Ning Ding and Biqing Qi and Bowen Zhou},
  year={2025},
  institution={Tsinghua University and Shanghai AI Lab},
  url={https://github.com/TsinghuaC3I/MARTI}
}

@misc{martiv2_2025,
  title={MARTI-v2: Multi-Agent Reinforcement Learning with Tree Search for Code Generation and Mathematical Reasoning},
  author={[Your Authors]},
  year={2025},
  institution={Tsinghua University and Shanghai AI Lab},
  url={https://github.com/TsinghuaC3I/MARTI}
}
```

## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TsinghuaC3I/MARTI&type=Date)](https://www.star-history.com/#TsinghuaC3I/MARTI&Date)

## 📄 License

This project is released under the Apache License 2.0.

---

<p align="center">
  <i>MARTI © 2025 Tsinghua University & Shanghai AI Lab. All rights reserved.</i>
</p>
