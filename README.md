<div align="center">

<img src="./assert/logo.jpg" width="400">

# MARTI: A Framework for LLM-based Multi-Agent Reinforced Training and Inference

</div>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&size=20&duration=3000&pause=1000&color=00D9FF&center=true&vCenter=true&width=800&lines=Welcome+to+MARTI;Multi-Agent+RL+Framework;Now+with+Tree+Search+Support;Powered+by+Tsinghua+x+Shanghai+AI+Lab" alt="Typing Animation" />
</div>

<p align="center">
<img src="./assert/arxiv.png" width="14px" style="display:inline;"> <a href="https://arxiv.org" target="_blank">Arxiv(Coming Soon)</a> ｜
📚 <a href="./docs/1-Overview-Of-MARTI.md" target="_blank">Documentation</a> ｜
🤗 <a href="https://github.com/TsinghuaC3I/MARTI" target="_blank">GitHub</a>
</p>

> **MARTI** is an open-source framework for training LLM-based Multi-Agent Systems (MAS) with Reinforcement Learning (RL). It enables powerful, scalable, and adaptive workflows by combining centralized multi-agent interactions with distributed policy training. MARTI supports both built-in graph-based workflows and popular third-party multi-agent frameworks.

> **MARTI-v2** extends the framework with **tree search-augmented RL** for complex reasoning tasks like code generation. By integrating multi-agent tree search, MARTI-v2 enables efficient multi-step exploration with adaptive node expansion and refinement, allowing agents to systematically explore solution spaces and discover high-quality reasoning trajectories. The framework also incorporates advanced RL training techniques (GSPO loss for sequence-level optimization, TIS correction for vLLM sampling mismatch, dynamic data filtering, overlong buffer for token penalty) to support ultra-long sequences up to 32K tokens and heterogeneous multi-agent training.

>  We hope that MARTI not only advances reasoning capabilities beyond those of individual large language models or reasoning models, but also fosters collective intelligence as a step toward general artificial intelligence.

## 📣 Latest News
- **[2026-01-10]** 🚀🚀🚀 We release **MARTI-v2** with scaling multi-agent tree search via reinforcement learning for code generation.
- **[2025-05-27]** We release the codebase of MARTI framework, welcome to have a try on LLM-based multi-agent reinforcement learning. 🤗

## Table of Contents

- [💡 Overview](#-overview)
  - [MARTI-v2: Tree Search-Augmented Multi-Agent RL (New!)](#marti-v2-tree-search-augmented-multi-agent-rl-new)
  - [MARTI](#marti)
- [🚀 Quick Start](#-quick-start)
  - [📦 Installation](#-installation)
  - [🌳 Tree Search RL Training (New!)](#-tree-search-rl-training-new)
    - [Single-Agent MCTS Training](#single-agent-mcts-training)
    - [Multi-Agent MCTS Training](#multi-agent-mcts-training)
  - [🎭 MARTI](#-marti)
    - [Multi-Agent Inference](#multi-agent-inference)
    - [Multi-Agent Training](#multi-agent-training)
- [📊 Experimental Results](#-experimental-results)
  - [MARTI-v2 (New!)](#marti-v2-new)
    - [Training Details](#training-details)
    - [Benchmark Results](#benchmark-results)
  - [MARTI](#marti-1)
    - [Training Details](#training-details-1)
    - [Benchmark Results](#benchmark-results-1)
    - [Training Dynamics](#training-dynamics)
- [📚 Documentation](#-documentation)
- [🚩 Roadmap](#-roadmap)
- [👏 Acknowledge](#-acknowledge)
- [🤝 Core Contributors](#-core-contributors)
- [📬 Contact](#-contact)
- [🔬 Citation](#-citation)
- [⭐️ Star History](#️-star-history)

## 💡 Overview

### MARTI-v2: Tree Search-Augmented Multi-Agent RL (🔥New!)

MARTI-v2 extends the framework with **tree search-augmented reinforcement learning** for complex reasoning tasks like code generation. By integrating multi-agent tree search with advanced RL techniques, MARTI-v2 enables efficient multi-step exploration with adaptive node expansion and refinement, allowing agents to systematically explore solution spaces and discover high-quality reasoning trajectories.

The framework has been adapted to the latest [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) infrastructure, incorporating state-of-the-art RL training techniques for heterogeneous multi-agent training.

<p align="center">
  <img src="./assert/mars2_framework.png" width="800">
</p>
<p align="center"><i>Figure 1: Overview of Core Components of MARTI-v2</i></p>

**Key Features:**

- **Multi-Agent Tree Search**: Efficient tree exploration with asynchronous multi-agent tree search, supporting code generation tasks with adaptive node expansion and refinement
- **GSPO Loss**: Sequence-level policy optimization (vs. token-level in PPO) better suited for complex reasoning tasks
- **TIS Correction**: Truncated Importance Sampling addresses distribution shift in long sequence generation, enabling stable training for ultra-long contexts and correcting vLLM sampling bias during rollout
- **Heterogeneous Multi-Agent Training**: Train different models simultaneously (e.g., Qwen3-8B + AreaL-boba-2-8B) with independent roles, training strategies, and dynamic sample filtering per agent

### MARTI

We designed the MARTI framework following the principle of centralized multi-agent interaction with distributed policy training, where all agent interactions and reward allocation occur centrally while policy training is distributed across individual agents. As illustrated in Figure 1, MARTI comprises three core modules: Multi-Agent World, Centralized Rewarding, and Single Agent Trainer.

<p align="center">
  <img src="./assert/framework.jpg" width="800">
</p>
<p align="center"><i>Figure 2: Overview of Core Components of MARTI</i></p>

**Key Features:**
- Multi-Agent Inference + RL Training in a unified framework
- Graph-based workflows (debate, chain-of-agents, mixture-of-agents)
- Support for heterogeneous models within the same agent graph
- Built-in credit assignment and reward shaping strategies
- Support for diverse RL algorithms ([PPO](https://arxiv.org/abs/1707.06347), [GRPO](https://arxiv.org/abs/2402.03300), [REINFORCE++](https://arxiv.org/abs/2501.03262v3), [TTRL](https://arxiv.org/abs/2504.16084))
- Third-party integration with AutoGen and CAMEL (experimental)
- Advanced performance on reasoning benchmarks (e.g., AIME)

Additionally, building on single-agent RL frameworks like [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) and [verl](https://github.com/volcengine/verl), MARTI supports the vLLM v1 Engine and a Hybrid Engine to enable fast and efficient training.

## 🚀 Quick Start

### 📦 Installation

```bash
git clone https://github.com/TsinghuaC3I/MARTI.git
cd MARTI

pip install -r requirements.txt
```

Follow the setup instructions for dependencies, including OpenRLHF, Ray, and vLLM.

---

### 🌳 Tree Search RL Training (New!)

**MARTI-v2 supports:**
- **Single-agent** and **multi-agent** MCTS training for code generation.
- GSPO Loss (Sequence-level policy optimization)
- TIS Correction (Mitigates vLLM sampling distribution mismatch during rollout)
- Dynamic Filtering (Per-agent sample filtering for heterogeneous training)
- Overlong Buffer (Applies penalty to excessively long token sequences)

#### Single-Agent MCTS Training

```bash
# Minimum hardware requirement: approximately 8×80G GPUs

ROOT_DIR="/path/to/MARTI-v2"
MODEL_DIR="/path/to/models"

# Single-agent MCTS training
# See the script for more training examples
bash examples/mars2/run_train_single_mcts.sh ${ROOT_DIR} ${MODEL_DIR}

```

#### Multi-Agent MCTS Training

```bash
# Minimum hardware requirement: approximately 8×80G GPUs per agent

ROOT_DIR="/path/to/MARTI-v2"
MODEL_DIR="/path/to/models"

# Multi-agent MCTS training
# See the script for more training examples
bash examples/mars2/run_train_multi_mcts.sh ${ROOT_DIR} ${MODEL_DIR}

```

---

### 🎭 MARTI

#### Multi-Agent Inference

MARTI supports:
- Built-in DAG-based workflows: debate, mixture-of-agents, chain-of-agents
- Third-party frameworks: AutoGen and CAMEL (Experimental)

Example:

```bash
MODEL_DIR="Path to models, like Qwen2.5-3B"

# See the script for more inference examples
bash scripts/run_test_mas.sh ${MODEL_DIR}
```

#### Multi-Agent Training

MARTI supports:
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

### MARTI-v2 (New!)

#### Training Details

We employ the MARTI-v2 framework to train reasoning models, specifically `Qwen3-8B`, `Qwen3-14B`, `AreaL-boba-2-8B`, `AreaL-boba-2-14B`, and `DeepCoder-14B`. For multi-agent reinforcement learning, we employ a cluster configuration consisting of 3 nodes, each equipped with 8 H200 GPUs, allocating one full node per agent.

#### Benchmark Results

We evaluate MARTI-v2 on the LCB code generation benchmark under both single-agent and multi-agent settings compared to baseline methods. As shown in Figure 3 and Figure 4, our experiments demonstrate that:

- **Single-agent MCTS achieves faster convergence**: The single-agent setting outperforms Vanilla GRPO baseline across all base models, with Pass@1 improvements up to 4.6% and Pass@1(MCTS) improvements up to 5.1%, exhibiting faster early-stage convergence and stronger deep optimization capabilities.
- **Multi-agent MCTS breaks performance bottlenecks**: The multi-agent setting maintains policy diversity and effectively addresses the performance saturation issue in later training stages. For Qwen3-8B, multi-agent training achieves 8.0% improvement over the base model, 4.4% over Vanilla GRPO, and 2.9% over single-agent peak performance.
- **Enhanced system-level collaboration**: With 14B-scale heterogeneous agent teams, multi-agent training achieves 71.2% Pass@1(MCTS), with consistent improvements in Pass@N metrics, validating comprehensive enhancements in collaborative problem-solving capabilities.

<p align="center">
  <img src="./assert/homo_results.png" width="800">
</p>
<p align="center"><i>Figure 3: Experimental results of single-agent MCTS and baseline methods on LCB benchmarks</i></p>

<p align="center">
  <img src="./assert/heter_pass1.png" width="800">
</p>
<p align="center"><i>Figure 4: Pass@1 results of multi-agent MCTS and baseline methods on LCB benchmarks</i></p>




### MARTI

#### Training Details

We employ the MARTI framework to train both base and reasoning models, specifically `Qwen2.5-3B` and `DeepScaleR-1.5B-Preview`. For `Qwen2.5-3B`, we implement DeepSeek-R1 zero-like reinforcement learning training using Level 3-5 samples from the MATH dataset. The `DeepScaleR-1.5B-Preview` model, which exhibits strong inherent reasoning capabilities but presents training challenges, undergoes [Test-Time Reinforcement Learning (TTRL)](https://github.com/PRIME-RL/TTRL) adaptation on AIME benchmark data. For multi-agent reinforcement learning, we employ a cluster configuration consisting of 3 nodes, each equipped with 8 A800 80GB GPUs, allocating one full node per agent.

#### Benchmark Results
We compare non-reasoning and reasoning models under various configurations and show that majority voting consistently outperforms multi-agent workflows when trained conventionally. This reflects known limitations of current LLM-based agent systems, such as poor role adherence and ineffective inter-agent communication.

To address this, MARTI enhances model reasoning through structured agent interactions. As shown in Figure 3 and Figure 4, our experiments show that:

- MARTI-trained base models outperform standard RL setups and rival instructed models.
- Large reasoning models trained with MARTI using TTRL achieve state-of-the-art results on challenging tasks (e.g., 66.7 AIME score with Multi-Agent Debates).
- Multi-agent RL consistently surpasses single-agent systems in performance under the same compute budget.

<p align="center">
  <img src="./assert/qwen2.5-3b-base-instruct-avg.jpg" width="800">
</p>
<p align="center"><i>Figure 3: Average scores of Qwen2.5-3B base and instruct models under different budget and settings</i></p>


<p align="center">
  <img src="./assert/ds-1.5-qwen-1.7-avg.jpg" width="800">
</p>
<p align="center"><i>Figure 4: Average scores of reasoning models under different budget and settings</i></p>


#### Training Dynamics

##### Multi-Agents Debate
We conduct multi-agent debate training with `Qwen2.5-3B` The `Qwen2.5-3B` model is trained using REINFORCE++ on Level 3 to 5 samples from the MATH-500 dataset.

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
We evaluate a mixture-of-agents approach using the `Qwen2.5-3B` model, trained on Levels 3 through 5 of the MATH-500 training dataset.

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
- [MARTI-v2 Technical Details](./docs/4-Tree-Search-Technical-Details.md) (Coming Soon)

## 🚩 Roadmap

- [ ] Release MARTI Technical Report
- [ ] Initial support for agentic tasks (e.g., GAIA benchmark)
- [ ] Integration with more tree search algorithms (e.g., AlphaZero-style MCTS)
- [ ] Support for more reasoning tasks (e.g., theorem proving, scientific reasoning)
- [ ] More features are working in progress

## 👏 Acknowledge

MARTI is developed primarily based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). We would like to express our gratitude to the developers of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), as well as to the teams behind [vLLM](https://github.com/vllm-project/vllm), [Ray](https://github.com/ray-project/ray), [DeepSpeed](https://github.com/deepspeedai/DeepSpeed), and [TreeQuest](https://github.com/treequest/treequest) for their invaluable contributions.

# TODO
## 🤝 Core Contributors   

- **Project Lead:** [Kaiyan Zhang](https://iseesaw.github.io/)
- **Agent Group:** [Runze Liu](https://ryanliu112.github.io/), [Kaiyan Zhang](https://iseesaw.github.io/), [Kai Tian](https://github.com/XiaoTiank), [Guoli Jia](https://github.com/exped1230), [Xingtai Lv](https://github.com/telxt), [Che Jiang](https://github.com/dcdsf321)
- **RL Group:** [Kaiyan Zhang](https://iseesaw.github.io/), [Xuekai Zhu](https://github.com/Xuekai-Zhu), [Sihang Zeng](https://github.com/zengsihang), [Yuchen Fan](https://github.com/YuchenFan48), [Yuxin Zuo](https://github.com/yuxinzuo)

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

```

## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TsinghuaC3I/MARTI&type=Date)](https://www.star-history.com/#TsinghuaC3I/MARTI&Date)


---

<p align="center">
  <i>MARTI © 2025 Tsinghua University & Shanghai AI Lab. All rights reserved.</i>
</p>
