import ray
import torch
import numpy as np
from ray.util.placement_group import placement_group

from marti.models.vllm.engine import create_vllm_engines
from marti.models.ray_launcher import PPORayActorGroup, ReferenceModelRayActor, RewardModelRayActor
from marti.trainers.ppo.actor import ActorModelRayActor
from marti.trainers.ppo.critic import CriticModelRayActor
from marti.trainers.ppo.prime import PrimeModelRayActor
from marti.trainers.ppo.saliency import SaliencyModelRayActor
from marti.helpers.common import get_tokenizer
from marti.controllers.base_controller import Agent, generate_samples_remote, BaseController
from marti.worlds.multi_agent_world import MultiAgentWorld, Samples

class MultiAgentController(BaseController):
    def __init__(self, strategy):
        super().__init__(strategy=strategy)

        # Record global seed and set different seed for each agent
        self.global_seed = self.args.seed

    def get_seed(self, num_workers=4):
        agent_seed = self.global_seed
        self.global_seed += num_workers * 10
        return agent_seed

    def build(self):
        self.load_dataset(tokenizer=None)
        # Prepare config for each agent
        self.agent_configs = []
        for agent_dict in self.args.agents:
            for agent_id, agent_config in agent_dict.items():
                self.agent_configs.append([agent_id, agent_config])

        if self.args.agent_workflow in ["multi-agents-debate", "mixture-of-agents"]:
            self.args.credit_model = None
            print("Credit model is not supported for Multi-Agents-Debate and Mixture-of-Agents. Set Credit model to None.")

        if self.args.credit_model == "prime":
            self.credit_model = self._init_prime()
        elif self.args.credit_model == "saliency":
            self.credit_model = self._init_saliency()
        else:
            self.credit_model = None

        if self.args.credit_pretrain is not None:
            self.shared_tokenizer = get_tokenizer(self.args.credit_pretrain, None, "left", self.strategy, use_fast=not self.args.disable_fast_tokenizer)
        else:
            self.shared_tokenizer = None

        self.agent_list = self._init_agents_parallel() if self.args.parallel_loading else self._init_agents_sequential()

        self.num_agents = len(self.agent_list)
        
        # create samples maker
        self.world = MultiAgentWorld(strategy=self.strategy, agents=[agent.get_metadata() for agent in self.agent_list], shared_tokenizer=self.shared_tokenizer)

    def _init_agents_sequential(self):
        agents = []
        for i, (agent_id, agent_config) in enumerate(self.agent_configs):
            # We copy the first agent to others under shared_agents setttings
            if self.args.shared_agents and i > 0:
                agent = agents[0]
            else:
                agent = self._init_agent(agent_id=agent_id,
                                    agent_config=agent_config,
                                    global_config=self.args)
            agents.append(agent)
        return agents

    def _init_agents_parallel(self):
        """Parallelly initialize multiple agents"""
        agent_refs = []
        
        for i, (agent_id, agent_config) in enumerate(self.agent_configs):
            # If shared_agents is enabled and the agent is not the first one, skip initialization.
            if self.args.shared_agents and i > 0:
                continue

            # Create an asynchronous initialization task
            agent_ref = self._init_agent_async.remote(
                self, agent_id, agent_config, self.args
            )
            agent_refs.append(agent_ref)

        # Wait for all agents to complete initialization.
        print(f"Initializing {len(agent_refs)} agents in parallel...")
        agents = ray.get(agent_refs)
        
        # If "shared_agents" is enabled, duplicate the first agent to other locations.
        if self.args.shared_agents:
            first_agent = agents[0]
            agents = [first_agent] * len(self.agent_configs)
        
        print(f"Successfully initialized {len(agents)} agents")
        return agents

    @ray.remote
    def _init_agent_async(self, agent_id, agent_config, global_config):
        """Asynchronous initialization of a single agent's Ray remote function"""
        return self._init_agent(agent_id, agent_config, global_config)

    def _init_prime(self):
        args = self.args
        strategy = self.strategy
        # if colocated, create placement group for actor and ref model explicitly.
        pg = None
        if args.colocate_credit_ref:
            assert (
                args.credit_num_nodes == args.credit_ref_num_nodes and args.credit_num_gpus_per_node == args.credit_ref_num_gpus_per_node
            ), "num_nodes and num_gpus_per_node must be the same when colocate credit and ref model."

            bundles = [
                {
                    "GPU": args.credit_num_gpus_per_node,
                    "CPU": args.credit_num_gpus_per_node
                }
                for _ in range(args.credit_num_nodes)
            ]
            pg = placement_group(bundles, strategy=self.args.pg_strategy)
            ray.get(pg.ready())
        credit_model = PPORayActorGroup(
            args.credit_num_nodes,
            args.credit_num_gpus_per_node,
            PrimeModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.75 if pg else 1,
        )

        credit_ref_model = PPORayActorGroup(
            args.credit_ref_num_nodes,
            args.credit_ref_num_gpus_per_node,
            ReferenceModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.25 if pg else 1,
        )

        refs = credit_ref_model.async_init_model_from_pretrained(
            strategy, args.credit_pretrain)
        ray.get(refs)
        
        refs = credit_model.async_init_model_from_pretrained(
            strategy, args.credit_pretrain, self._max_steps)
        ray.get(refs)

        # init actor and critic mdoel
        refs = credit_model.async_init_credit_trainer(
            credit_ref_model
        )
        ray.get(refs)

        return credit_model

    def _init_saliency(self):
        args = self.args
        strategy = self.strategy
        # if colocated, create placement group for actor and ref model explicitly.
        credit_model = PPORayActorGroup(
            args.credit_num_nodes,
            args.credit_num_gpus_per_node,
            SaliencyModelRayActor,
            num_gpus_per_actor=1,
        )

        refs = credit_model.async_init_model_from_pretrained(
            strategy, args.credit_pretrain, self._max_steps)
        ray.get(refs)

        return credit_model

    def _init_agent(self, agent_id, agent_config, global_config):
        strategy = self.strategy
        print("Create agent for", agent_id, agent_config)

        # Set default parameters
        for key, value in global_config.default_agent.items():
            if key not in agent_config:
                agent_config[key] = value

        print(agent_id, agent_config)
        # Create tokenizer
        tokenizer = get_tokenizer(
            agent_config.pretrain, None, "left", strategy, use_fast=not agent_config.disable_fast_tokenizer)

        generate_kwargs = {
            "do_sample": True,
            "max_new_tokens": agent_config.generate_max_len,
            "max_length": agent_config.max_len,
            "temperature": agent_config.temperature,
            "top_p": agent_config.top_p,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if agent_config.is_tuning:
            # if colocated, create placement group for actor and ref model explicitly.
            pg = None
            if agent_config.colocate_actor_ref or agent_config.colocate_all_models:
                assert (
                    agent_config.actor_num_nodes == agent_config.ref_num_nodes and agent_config.actor_num_gpus_per_node == agent_config.ref_num_gpus_per_node
                ), "num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

                bundles = [{"GPU": 1, "CPU": 1} for _ in range(agent_config.actor_num_nodes * agent_config.actor_num_gpus_per_node)]
                pg = placement_group(bundles, strategy="PACK")

                ray.get(pg.ready())

            # init vLLM engine for text generation
            vllm_engines = None
            if agent_config.vllm_num_engines is not None and agent_config.vllm_num_engines > 0:
                max_len = agent_config.max_len if agent_config.max_len else agent_config.prompt_max_len + agent_config.generate_max_len
                
                if agent_config.colocate_all_models:
                    assert (
                        agent_config.actor_num_nodes * agent_config.actor_num_gpus_per_node
                        == agent_config.vllm_num_engines * agent_config.vllm_tensor_parallel_size
                    ), (
                        f"actor_num_nodes * actor_num_gpus_per_node must be equal to "
                        f"vllm_num_engines * vllm_tensor_parallel_size, got {agent_config.actor_num_nodes * agent_config.actor_num_gpus_per_node} "
                        f"and {agent_config.vllm_num_engines * agent_config.vllm_tensor_parallel_size}"
                    )

                vllm_engines = create_vllm_engines(
                    agent_config.vllm_num_engines,
                    agent_config.vllm_tensor_parallel_size,
                    agent_config.pretrain,
                    self.get_seed(agent_config.vllm_num_engines),
                    agent_config.enable_prefix_caching,
                    agent_config.enforce_eager,
                    max_len,
                    pg if agent_config.colocate_all_models else None,
                    agent_config.vllm_gpu_memory_utilization,
                    agent_config.vllm_enable_sleep
                )

            actor_model = PPORayActorGroup(
                agent_config.actor_num_nodes,
                agent_config.actor_num_gpus_per_node,
                ActorModelRayActor,
                pg=pg,
                num_gpus_per_actor=0.2 if pg else 1,
            )

            ref_model = PPORayActorGroup(
                agent_config.ref_num_nodes,
                agent_config.ref_num_gpus_per_node,
                ReferenceModelRayActor,
                pg=pg,
                num_gpus_per_actor=0.2 if pg else 1,
            )

            if not agent_config.colocate_all_models:
                pg = None

            if agent_config.critic_pretrain and agent_config.colocate_critic_reward:
                assert (
                    agent_config.critic_num_nodes == agent_config.reward_num_nodes
                    and agent_config.critic_num_gpus_per_node == agent_config.reward_num_gpus_per_node
                ), "num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."


                bundles = [{"GPU": 1, "CPU": 1} for _ in range(agent_config.critic_num_nodes * agent_config.critic_num_gpus_per_node)]
                pg = placement_group(bundles, strategy="PACK")
                ray.get(pg.ready())

            if agent_config.critic_pretrain:
                critic_model = PPORayActorGroup(
                    agent_config.critic_num_nodes,
                    agent_config.critic_num_gpus_per_node,
                    CriticModelRayActor,
                    pg=pg,
                    num_gpus_per_actor=0.2 if pg else 1,
                )
            else:
                critic_model = None

            # multiple reward models
            if agent_config.reward_pretrain is not None:
                reward_pretrains = agent_config.reward_pretrain.split(",")
                reward_models = []
                for _ in reward_pretrains:
                    reward_models.append(
                        PPORayActorGroup(
                            agent_config.reward_num_nodes,
                            agent_config.reward_num_gpus_per_node,
                            RewardModelRayActor,
                            pg=pg,
                            num_gpus_per_actor=0.2 if pg else 1,
                        )
                    )
            else:
                reward_models = None

            if ref_model is not None:
                # init reference/reward/actor model
                refs = ref_model.async_init_model_from_pretrained(
                    strategy, agent_config.pretrain)
                ray.get(refs)

            eos_token_id = agent_config.eos_token_id if agent_config.eos_token_id is not None else tokenizer.eos_token_id
            refs = actor_model.async_init_model_from_pretrained(
                strategy, agent_config.pretrain, self._max_steps, rolename=f"actor_{agent_id}", eos_token_id=eos_token_id)
            ray.get(refs)

            if agent_config.reward_pretrain is not None:
                for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
                    refs = reward_model.async_init_model_from_pretrained(
                        strategy, reward_pretrain)
                    ray.get(refs)

        if agent_config.is_tuning:
            if agent_config.critic_pretrain:
                # critic scheduler initialization depends on max_step, so we have to init critic after actor
                refs.extend(critic_model.async_init_model_from_pretrained(
                    strategy, agent_config.critic_pretrain, self._max_steps, rolename=f"critic_{agent_id}"))
                ray.get(refs)

            # init actor and critic mdoel
            refs = actor_model.async_init_actor_trainer(
                critic_model, ref_model, reward_models, agent_config.remote_rm_url, vllm_engines=vllm_engines
            )
            ray.get(refs)
        else:
            actor_model = None
            critic_model = None

        agent = Agent(
            agent_id=agent_id,
            agent_config=agent_config,
            vllm_engines=vllm_engines,
            actor_model_group=actor_model,
            critic_model_group=critic_model,
            tokenizer=tokenizer,
            generate_kwargs=generate_kwargs,
            is_reasoning_model=agent_config.is_reasoning_model
        )

        return agent

    def run(self):
        # update steps if load checkpoints
        num_update_steps_per_episodes, consumed_samples = self.load_checkpoint_steps()

        # start fitting
        self.fit(
            consumed_samples=consumed_samples,
            num_update_steps_per_episodes=num_update_steps_per_episodes
        )

        # save actor and critic workers in agent
        for agent in self.agent_list:
            agent.save_actor_and_critic_model()
            if self.args.shared_agents:
                break

        if self.credit_model is not None:
            self.credit_model.async_save_model()


    def generate_shared_samples(self, steps, rand_prompts):
        # TODO: world_size of differnt agents should be same!!!
        world_size = None
        for agent in self.agent_list:
            if agent.actor_model_group is not None:
                world_size = agent.actor_model_group.world_size
                break

        any_key = next(iter(rand_prompts.keys()))
        length = len(rand_prompts[any_key])
        chunk_size = (length + world_size - 1) // world_size
        chunked = [dict() for _ in range(world_size)]
        for key, data_list in rand_prompts.items():
            for i in range(world_size):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, length)
                sub_slice = data_list[start_idx:end_idx]
                chunked[i][key] = sub_slice

        all_refs = []
        for rank in range(world_size):
            samples_ref = generate_samples_remote.remote(
                self.world, chunked[rank], rank, world_size)
            all_refs.append(samples_ref)

        # TODO: build chain-level samples, along with agent_id for each samples.sequences
        # training prime and compute implicit rewards
        # 1) firstly, compute log_probs_ref and save in samples
        # 2) then, training prime model with samples
        # 3) later,compute implicit rewards and save in samples
        # 4) finally, pass samples to actor models
        if self.args.credit_model is None or steps <= self.args.warmup_steps_for_credit:
            all_results = [r["sample"] for r in ray.get(all_refs)]
            credit_status = None
        else:
            all_results = sum([r["sample"] for r in ray.get(all_refs)], [])
            all_results_with_rewards_refs = self.credit_model.async_fit_and_reward_credit_model(
                    steps, all_results)

            all_results = ray.get(all_results_with_rewards_refs)
            all_results_with_rewards = [a[0] for a in all_results]
            # get credit status on rank 0
            credit_status = [a[1] for a in all_results if a[1]["is_rank_0"]][0]
            del credit_status["is_rank_0"]
            credit_status["credit/token_score"] = self.compute_average_rewards(
                all_results_with_rewards)

        sharded_data_refs = [[None for _ in range(world_size)] for _ in range(self.num_agents)]

        # Create samples for each agent from samples.info
        for rank in range(world_size):
            # TODO: build agent-level samples
            # process_rewards
            if self.credit_model is None or steps <= self.args.warmup_steps_for_credit:
                shared_data = all_results[rank]
            else:
                shared_data = all_results_with_rewards[rank]
                self.apply_agent_level_reward_shaping(shared_data)

            if self.args.shared_agents:
                rank_samples = []
                if isinstance(shared_data[0], Samples):
                    for agent_id in range(self.num_agents):
                        rank_samples.extend([samples.info[agent_id] for samples in shared_data])
                elif isinstance(shared_data[0], list):
                    for agent_id in range(self.num_agents):
                        rank_samples.extend(shared_data[agent_id])

                sharded_data_refs[0][rank] = ray.put(rank_samples)
            else:
                # get agent data from samples and put into sharded_data_refs for each agent
                for agent_id in range(self.num_agents):
                    if isinstance(shared_data[0], Samples) and self.args.agent_workflow in ["chain-of-agents", "mixture-of-agents"]:
                        rank_data = [samples.info[agent_id] for samples in shared_data]
                    elif isinstance(shared_data[0], list) and self.args.agent_workflow == "multi-agents-debate":
                        rank_data = shared_data[agent_id]

                    sharded_data_refs[agent_id][rank] = ray.put(rank_data)

        return sharded_data_refs, None, credit_status

    def apply_agent_level_reward_shaping(self, shared_data):
        for samples in shared_data:
            # Ensure each sample has exactly num_agents
            assert len(samples.info) == self.num_agents, \
                f"Found {len(samples.info)} info entries, expected {self.num_agents}"

            packing_agent_rewards = [[] for _ in range(self.num_agents)]
            for agent_level_scores in samples.agent_level_scores:
                for agent_id, agent_score in enumerate(agent_level_scores):
                    packing_agent_rewards[agent_id].append(agent_score)

            # Now put agent-level rewards into samples.info
            for i, agent_rewards in enumerate(packing_agent_rewards):
                agent_rewards = torch.tensor(agent_rewards, device="cpu", dtype=torch.float)
                verifier_rewards = samples.info[i].labels
                assert len(agent_rewards) == len(verifier_rewards), f"{len(agent_rewards)} vs {len(verifier_rewards)}"
                credit_coef = getattr(self.args, "credit_score_coef", 1.0)
                verifier_coef = getattr(self.args, "verifier_score_coef", 1.0)
                samples.info[i].labels = agent_rewards * credit_coef + verifier_rewards * verifier_coef

    def set_agent_vllm_engine(self, command):
        if self.args.vllm_enable_sleep:
            from marti.models.vllm.engine import batch_vllm_engine_call
            for agent in self.agent_list:
                batch_vllm_engine_call(agent.vllm_engines, command)

    def compute_average_rewards(self, all_samples):
        all_rewards = []
        for samples in sum(all_samples, []):
            all_rewards.extend([rewards.sum().item()
                               for rewards in samples.agent_level_scores])
        return np.mean(all_rewards)
    
    def step(self, rand_prompts, episode, steps, pbar):
        self.set_agent_vllm_engine("wake_up")
        # make shared samples refs
        shared_data_refs, sft_dataset, credit_status = self.generate_shared_samples(steps, rand_prompts=rand_prompts)
        self.set_agent_vllm_engine("sleep")

        # if steps <= self.args.warmup_steps_for_credit:
        #     self.save_logs(self.args, steps, credit_status, None)
        #     steps += 1
        #     pbar.set_postfix(credit_status)
        #     pbar.update()
        #     return None

        # start training each agent
        refs = []
        
        agent_set = range(self.num_agents)

        # if self.args.use_iterative_agent_training:
        #     agent_set = [episode % self.num_agents]

        # We only train the first agent
        if self.args.shared_agents:
            agent_set = [0]

        for idx in agent_set:
            if self.agent_list[idx].agent_config.is_tuning:
                refs.extend(self.agent_list[idx].actor_model_group.async_fit_actor_model(
                    steps, shared_data_refs[idx], sft_dataset))

        all_results = ray.get(refs)
        all_results = [
            result for result in all_results if result["is_rank_0"]]
        
        assert len(all_results) > 0, "No results from actor model."

        logs_dict, perf_stats = {}, {}
        for result in all_results:
            cur_logs_dict, cur_perf_stats, agent_id = result["status"], result["perf_stats"], result["agent"]
            logs_dict.update(
                {f"{agent_id}/{k}": v for k, v in cur_logs_dict.items()})
            if cur_perf_stats is not None:
                perf_stats.update(
                    {f"{agent_id}/{k}": v for k, v in cur_perf_stats.items()})

        if credit_status is not None:
            logs_dict.update(credit_status)

        return [logs_dict, perf_stats]
