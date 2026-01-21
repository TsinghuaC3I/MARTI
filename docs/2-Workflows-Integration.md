# Multi-Agent Workflows

## Built-in Workflows

- Multi-Agent LLM Workflow
  - The multi-agent LLM workflow is designed to enhance the reasoning capabilities of LLMs through structured multi-agent interactions. By leveraging a graph-based architecture and multi-turn dynamics, the framework orchestrates collaborative workflows among agents to tackle complex tasks. The core implementation revolves around a modular graph structure, spatial and temporal interaction mechanisms, and configurable workflows, including chain-of-agents, multi-agent debate, and mixture-of-agents. These components collectively enable scalable and adaptable agent coordination, as detailed below.

- Multi-Agent Graph
  - The multi-agent workflow leverages directed acyclic graphs (DAG) to model agent interactions, such as [GPTSwarm](https://github.com/metauto-ai/GPTSwarm), and [AgentPrune](https://github.com/yanweiyue/AgentPrune). Each node, an instance of `MathSolver`, is defined by attributes including a unique identifier, role (e.g., generator, verifier), and LLM configuration (e.g., `pretrain`). Edges are categorized as spatial (intra-round communication) or temporal (cross-round dependencies), governed by adjacency matrices. A spatial mask $\text{SM}_t[i,i] = 1$ permits the agent $A_i$ to participate in the workflow in round $t$ and $\text{SM}_t[i,j] = 1$ permits information flow from agent $A_i$ to $A_j$ in round $t$, while a temporal mask $\text{TM}_t[i,j] = 1$ allows $A_j$ to access $A_i$'s output from its latest interaction round $t' < t$. This structure ensures flexibility in defining agent topologies, supporting diverse workflows through adjacency configurations. In each round, the framework constructs a directed acyclic graph (DAG) using spatial masks and temporal masks and the workflow is executed by traversing the entire graph, as illustrated in Algorithm 1.


<p align="center">
  <img src="../assert/marti-multi-turn-agent.jpg" width="800">
</p>
<p align="center"><i>Algorithm 1: Multi-Turn Agent Execution</i></p>



- Workflow Configurations
  - The framework adopts a configuration-driven approach using Hydra. The number of rounds is defined in `workflow_args` and default agent parameters are defined in `default_agent`, including LLMs and inference settings (e.g., `temperature`, `top-p`). Agent parameters defined in the `agents` configuration can override the default configurations, including unique ids, roles, prompt templates and LLMs.

<p align="center">
  <img src="../assert/marti-workflow-config.jpg" width="800">
</p>
<p align="center"><i>Figure 1: Workflows Configures</i></p>


The framework implements three multi-agent workflows as shown in Figure 1: chain-of-agents, multi-agents debate, and mixture-of-agents, each tailored to specific collaborative paradigms.


## Third-party Frameworks

To support diverse multi-agent applications, we integrate third-party frameworks that enable different agent interaction patterns, including AutoGen and CAMEL. Our framework uses Ray to support parallel and scalable multi-agent RL and its RLlib provides native support for multi-agent reinforcement learning.

- [AutoGen](https://github.com/microsoft/autogen) provides an efficient and flexible framework that facilitates collaborative interactions and task assignment among various agents. We integrate AutoGen with MARTI by implementing several key adaptations. First, we re-encapsulate a Ray-based infra interface that matches OpenAI's API format. To achieve concurrent API calls during training, we design a port management system featuring automatic port conflict detection and retry logic during service initialization. The system includes interface availability monitoring through periodic HTTP checks and exception handling to maintain service availability. We refactor AutoGen's logging system by extracting structured LLM response data and develop a DialogueProcessor module. This component uses regex patterns to identify agent roles and manages conversation history through a message queue management strategy. These modifications enable precise tracking of agent interactions while maintaining compatibility with our training framework's requirements.

- [CAMEL](https://github.com/camel-ai/camel) is a specialized multi-agent framework designed for complex tasks involving tool usage. CAMEL facilitates step-by-step task decomposition and efficient response through coordinated interactions between user and assistant agents. We integrate it with MARTI by implementing the same port conflict mechanism as implemented in AutoGen, along with a service availability check to ensure stable runtime behavior. Moreover, we encapsulate the output format of the Ray framework, allowing direct compatibility with the CAMEL via OpenAI's ChatCompletionClient class. For logging and multi-agent context management, we rewrite the execution of CAMEL's society module, capturing the responses from user and system agents. Multi-turn dialogues are organized into parallel message queues indexed by task IDs, enabling concurrent multi-agent communication and consistent state tracking.
