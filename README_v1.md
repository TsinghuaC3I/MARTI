<div align="center">

<img src="./assert/logo.jpg" width="400">

# A Framework for LLM-based Multi-Agent Reinforced Training and Inference

</div>

MARTI is an open-source framework for training LLM-based Multi-Agent Systems (MAS) with Reinforcement Learning (RL). It enables powerful, scalable, and adaptive workflows by combining centralized multi-agent interactions with distributed policy training. MARTI supports both built-in graph-based workflows and popular third-party multi-agent frameworks. We hope that MARTI not only advances reasoning capabilities beyond those of individual large language models or reasoning models, but also fosters collective intelligence as a step toward general artificial intelligence. 

**MARTI is still in a very early experimental stage, we are actively developing more powerful LLM-based multi-agent RL approaches and warmly welcome collaborations in this direction.**

## 🔥 News
- [2025-05-27] We release the codebase of MARTI framework, welcome to have a try on LLM-based multi-agent reinforcement learning. 🤗

## 🔖 Content
- [A Framework for LLM-based Multi-Agent Reinforced Training and Inference](#a-framework-for-llm-based-multi-agent-reinforced-training-and-inference)
  - [🔥 News](#-news)
  - [🔖 Content](#-content)
  - [🔍 Overview](#-overview)
  - [📦 Installation](#-installation)
  - [⚙️ Usage](#️-usage)
    - [🔁 Multi-Agent Inference](#-multi-agent-inference)
    - [🏋️ Multi-Agent Training](#️-multi-agent-training)
    - [📊 Preliminary Experiments](#-preliminary-experiments)
      - [Training Details](#training-details)
      - [Benchmark Results](#benchmark-results)
      - [Training Dynamics](#training-dynamics)
        - [Multi-Agents Debate](#multi-agents-debate)
        - [Mixture-of-Agents](#mixture-of-agents)
  - [📚 Documentation](#-documentation)
  - [🚩 Roadmap](#-roadmap)
  - [👏 Acknowledge](#-acknowledge)
  - [🤝 Core Contributors](#-core-contributors)
  - [📬 Contact](#-contact)
  - [🔬 Citation](#-citation)


## 🔍 Overview

Modern LLM applications often rely on complex task-solving strategies. While single-agent LLMs struggle with scalability and context tracking, multi-agent systems (MAS) offer a promising alternative. MARTI fills the gap between inference and training in MAS by introducing reinforcement learning for structured, collaborative agent behavior.

We designed the **MARTI** framework following the principle of centralized multi-agent interaction with distributed policy training, where all agent interactions and reward allocation occur centrally while policy training is distributed across individual agents. As illustrated in Figure 1, MARTI comprises three core modules: Multi-Agent World, Centralized Rewarding, and Single Agent Trainer.

<p align="center">
  <img src="./assert/framework.jpg" width="800">
</p>
<p align="center"><i>Figure 1: Overview of Core Components of MARTI</i></p>

Key Features:
- Multi-Agent Inference + RL Training in a unified framework
- Graph-based workflows (debate, chain-of-agents, mixture-of-agents)
- Support for heterogeneous models within the same agent graph
- Built-in credit assignment and reward shaping strategies
- Support for diverse RL algorithms ([PPO](https://arxiv.org/abs/1707.06347), [GRPO](https://arxiv.org/abs/2402.03300), [REINFORCE++](https://arxiv.org/abs/2501.03262v3), [TTRL](https://arxiv.org/abs/2504.16084))
- Third-party integration with AutoGen and CAMEL (experimental)
- Advanced performance on reasoning benchmarks (e.g., AIME)

Additionally, building on single-agent RL frameworks like [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) and [verl](https://github.com/volcengine/verl), MARTI supports the vLLM v1 Engine and a Hybrid Engine to enable fast and efficient training.

## 📦 Installation

```bash
git clone https://github.com/TsinghuaC3I/MARTI.git
cd MARTI

pip install -r requirements.txt
```

Follow the setup instructions for dependencies, including OpenRLHF, Ray, and vLLM.

## ⚙️ Usage

### 🔁 Multi-Agent Inference

MARTI supports:
- Built-in DAG-based workflows: debate, mixture-of-agents, chain-of-agents
- Third-party frameworks: AutoGen and CAMEL (Experimental)

Example:

```bash
MODLE_DIR="Path to models, like Qwen2.5-3B"

# See the script for more inference examples
bash scripts/run_test_mas.sh ${MODEL_DIR}
```

### 🏋️ Multi-Agent Training

MARTI supports:
- Rule-based rewards (Reward Shaping)
- Generative reward models (LLM-as-Judge) (Experimental)
- Tree-based AgentPRM (ImplicitPRM) (Experimental)
- Supervised fine-tuning + RL (e.g., PPO, GRPO)

Example:

```bash
# Minimum hardware requirement for training with 3 Qwen2.5-3B agents: approximately 6×80G GPUs

MODLE_DIR="Path to models, like Qwen2.5-3B"
WANDB_KEY="API key of wandb"

# Train Single Agent with GRPO
bash scripts/run_train_grpo.sh ${MODEL_DIR} ${WANDB_KEY}

# Train Multi-Agent Debate with Reinforce++
bash scripts/run_train_mad.sh ${MODEL_DIR} ${WANDB_KEY}
```

### 📊 Preliminary Experiments

#### Training Details

We employ the MARTI framework to train both base and reasoning models, specifically `Qwen2.5-3B` and `DeepScaleR-1.5B-Preview`. For `Qwen2.5-3B`, we implement DeepSeek-R1 zero-like reinforcement learning training using Level 3-5 samples from the MATH dataset. The `DeepScaleR-1.5B-Preview` model, which exhibits strong inherent reasoning capabilities but presents training challenges, undergoes [Test-Time Reinforcement Learning (TTRL)](https://github.com/PRIME-RL/TTRL) adaptation on AIME benchmark data. For multi-agent reinforcement learning, we employ a cluster configuration consisting of 3 nodes, each equipped with 8 A800 80GB GPUs, allocating one full node per agent.

#### Benchmark Results
We compare non-reasoning and reasoning models under various configurations and show that majority voting consistently outperforms multi-agent workflows when trained conventionally. This reflects known limitations of current LLM-based agent systems, such as poor role adherence and ineffective inter-agent communication.

To address this, MARTI enhances model reasoning through structured agent interactions. As shown in Figure 2 and Figure 3, our experiments show that:

- MARTI-trained base models outperform standard RL setups and rival instructed models.
- Large reasoning models trained with MARTI using TTRL achieve state-of-the-art results on challenging tasks (e.g., 66.7 AIME score with Multi-Agent Debates).
- Multi-agent RL consistently surpasses single-agent systems in performance under the same compute budget.

<p align="center">
  <img src="./assert/qwen2.5-3b-base-instruct-avg.jpg" width="800">
</p>
<p align="center"><i>Figure 2: Average scores of Qwen2.5-3B base and instruct models under different budget and settings</i></p>


<p align="center">
  <img src="./assert/ds-1.5-qwen-1.7-avg.jpg" width="800">
</p>
<p align="center"><i>Figure 3: Average scores of reasoning models under different budget and settings</i></p>


#### Training Dynamics

##### Multi-Agents Debate
We conduct multi-agent debate training with `Qwen2.5-3B` The `Qwen2.5-3B` model is trained using REINFORCE++ on Level 3 to 5 samples from the MATH-500 dataset.

<p align="center">
  <img src="./assert/mad-rl-amc.jpg" width="400">
  <img src="./assert/mad-rl-math.jpg" width="400">
</p>
<p align="center"><i>Figure 4: Accuracy of MAD (Qwen2.5-3B, MATH) on AMC and MATH</i></p>


<p align="center">
  <img src="./assert/mad-dynamics.jpg" width="800">
</p>
<p align="center"><i>Figure 5: Training Dynamics of MAD (Qwen2.5-3B, MATH)</i></p>


##### Mixture-of-Agents
We evaluate a mixture-of-agents approach using the `Qwen2.5-3B` model, trained on Levels 3 through 5 of the MATH-500 training dataset.

<p align="center">
  <img src="./assert/moa-rl-amc.jpg" width="400">
  <img src="./assert/moa-rl-math.jpg" width="400">
</p>
<p align="center"><i>Figure 6: Accuracy of MoA (Qwen2.5-3B, MATH) on AMC and MATH</i></p>


<p align="center">
  <img src="./assert/moa-dynamics.jpg" width="800">
</p>
<p align="center"><i>Figure 7: Training Dynamics of MoA (Qwen2.5-3B, MATH)</i></p>


## 📚 Documentation
- [Overview of MARTI](./docs/1-Overview-Of-MARTI.md)
- [Workflows Integration](./docs/2-Workflows-Integration.md)
- [Reward and Training](./docs/3-Reward-And-Training.md)

## 🚩 Roadmap

- [ ] Release MARTI Technical Report
- [ ] Initial support for agentic tasks (e.g., GAIA benchmark)
- [ ] More features are working in progress

## 👏 Acknowledge

MARTI is developed primarily based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). We would like to express our gratitude to the developers of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), as well as to the teams behind [vLLM](https://github.com/vllm-project/vllm), [Ray](https://github.com/ray-project/ray) and [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) for their invaluable contributions.

## 🤝 Core Contributors
- Project Lead: [Kaiyan Zhang](https://iseesaw.github.io/)
- Agent Group: [Runze Liu](https://ryanliu112.github.io/), [Kaiyan Zhang](https://iseesaw.github.io/), [Kai Tian](https://github.com/XiaoTiank), [Guoli Jia](https://github.com/exped1230), [Xingtai Lv](https://github.com/telxt), [Che Jiang](https://github.com/dcdsf321)
- RL Group: [Kaiyan Zhang](https://iseesaw.github.io/), [Xuekai Zhu](https://github.com/Xuekai-Zhu), [Sihang Zeng](https://github.com/zengsihang), [Yuchen Fan](https://github.com/YuchenFan48), [Yuxin Zuo](https://github.com/yuxinzuo)

For the full list of contributors, please refer to the author list in the citation. We are also deeply grateful to everyone who engaged in discussions and provided valuable feedback throughout the development of this project.

## 📬 Contact

For issues or inquiries: 
- Kaiyan Zhang, Tsinghua University (zhang-ky22@mails.tsinghua.edu.cn)
- Biqing Qi, Shanghai AI Lab (qibiqing@pjlab.org.cn)

## 🔬 Citation

If you use MARTI in your research, please cite the project:

```
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
