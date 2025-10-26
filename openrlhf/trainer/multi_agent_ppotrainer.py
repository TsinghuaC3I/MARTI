import os
import time
from abc import ABC
from datetime import timedelta

import ray
import torch
from tqdm import tqdm

from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.multi_agent_experience_maker import MultiAgent_RemoteExperienceMaker as RemoteExperienceMaker
from openrlhf.trainer.ppo_utils.multi_agent_samples_generator import MultiAgentSamplesGenerator as SamplesGenerator
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import get_tokenizer
from openrlhf.trainer.ppo_trainer import BasePPOTrainer

logger = init_logger(__name__)

@ray.remote
class MultiAgent_PPOTrainer(BasePPOTrainer):
    """
    Trainer for Proximal Policy Optimization (PPO) / REINFORCE++ / GRPO / RLOO and their variants.
    Single Controller with Multiple ActorGroups
    """
    def __init__(
        self,
        pretrain: str,     # shijie TODO 修改为多个pretrain path list？  List[str]
        strategy: DeepspeedStrategy,
        # actor_model_group: RayActorGroup,
        # critic_model_group: RayActorGroup,
        # reward_model_group: RayActorGroup,
        # reference_model_group: RayActorGroup,
        # vllm_engines=None,
        # 修改为多个actor_model_group, critic_model_group, reward_model_group, reference_model_group, vllm_engines；参考marti multiagent-controller的实现 agent_list
        agent_list,  # : List[Agent]  # TODO 需要check shijie根据marti的build初始化建立的agent_list是否正确
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        super().__init__(
            pretrain,  # TODO pretrain的作用为在BasePPOTrainer init时, 初始化了self.tokenizer
            strategy,
            None,
            None,
            None,
            None,
            None,
            prompt_max_len,
            dataloader_pin_memory,
            prompt_split,
            eval_split,
            **generate_kwargs,
        )

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(self.init_kl_coef, self.kl_target, self.kl_horizon)
        else:
            self.kl_ctl = FixedKLController(self.init_kl_coef)

        if self.args.remote_rm_url and not self.args.remote_rm_url[0] == "agent":
            from openrlhf.utils.remote_rm_utils import RemoteRewardModel

            self.remote_reward_model = RemoteRewardModel.remote(self.args, self.remote_rm_url)

        self.agent_list = agent_list
        self.num_agents = len(self.agent_list)

        # 初始化tokenizer list
        self.tokenizer_list = []
        for agent_idx in range(self.num_agents):
            # TODO add agents_pretrains的路径list，指定多智能体的pretrain path list
            pretrain = self.args.agents_pretrain[agent_idx]
            self.tokenizer_list.append(get_tokenizer(pretrain, None, "left", strategy, use_fast=not self.args.disable_fast_tokenizer))
        #fyk:这里不需要tokenizer list了，因为每个agent都有自己的tokenizer
        #fyk:marti里不是直接传入agent_list，而是传入agent_list.metadata方法 需要check下agents和agent_list的差异
        # TODO generator class目前是写死的，后期可能需要适配特定的generator class
        # TODO lpf：fix--get metadata 没有返回generate kwargs
        self.samples_generator = SamplesGenerator(  # TODO 根据yikun修改的SamplesGeneratorAsync类, 传入对应参数进行初始化
            [agent.get_metadata(
            ) for agent in self.agent_list],
            self.strategy,
            self.prompt_max_len,
            **generate_kwargs,
        )

        self.experience_maker = RemoteExperienceMaker(
            self.agent_list,
            self.kl_ctl,
            self.strategy,
            self.tokenizer_list,
            remote_reward_model=self.remote_reward_model,
        )

        # TODO check是否需要传入apply chat template
        assert self.args.apply_chat_template == False, "apply_chat_template为True时, 会使用self.tokenizer(第一个agent)对数据进行apply_chat_template"
        self.prepare_datasets()
        self._init_wandb()

        # get eval and save steps
        if self.args.eval_steps == -1:
            self.args.eval_steps = float("inf")  # do not evaluate
        if self.args.save_steps == -1:
            self.args.save_steps = float("inf")  # do not save ckpt

    def fit(
        self,
    ) -> None:
        args = self.args

        # broadcast init checkpoint to vllm
        for agent_idx, agent in enumerate(self.agent_list):
            ckpt_path = os.path.join(agent.agent_config["ckpt_path"], "_actor")
            if args.load_checkpoint and os.path.exists(ckpt_path):
                # checkpoint_states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[
                checkpoint_states = ray.get(agent.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[
                    0
                ]
                logger.info(f"Agent {agent_idx} checkpoint_states: {checkpoint_states}")
                self._broadcast_to_vllm(agent)
            else:
                checkpoint_states = {"global_step": 0, "episode": 0, "data_loader_state_dict": {}}

        # Restore step and start_epoch
        steps = checkpoint_states["global_step"] + 1
        episode = checkpoint_states["episode"]
        data_loader_state_dict = checkpoint_states["data_loader_state_dict"]
        if data_loader_state_dict:
            self.prompts_dataloader.load_state_dict(data_loader_state_dict)

        generate_kwargs = self.generate_kwargs
        generate_kwargs["n_samples_per_prompt"] = self.args.n_samples_per_prompt

        for episode in range(episode, args.num_episodes):
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=False,
                initial=steps,
            )

            filtered_samples = []
            number_of_samples = 0
            for _, rand_prompts, labels, metadata in self.prompts_dataloader:
                remote_reward_model = self.remote_reward_model if self.args.dynamic_filtering else None
                # 得到 list of experience
                #这里出来是GPU个数的列表，其中每个元素是对应GPU上的多智能体的sample list，sample list中每个元素是每个agent的sample
                # update-lpf：这里得到的rollout_samples应该是List(Experience)
                rollout_samples = self.samples_generator.generate_samples(
                    rand_prompts, labels, metadata, remote_reward_model=remote_reward_model, **self.generate_kwargs  # Check lfy 多余的remote_reward_model=remote_reward_model,--lpf，保留用于远程reward 部署的接口remote_reward_model，默认设置为None
                )
                pbar.update()

                # TODO 完善多智能体dynamic filter 逻辑
                # dynamic filtering
                pass_rate = None
                if self.args.dynamic_filtering:
                    number_of_samples += len(rollout_samples)
                    # Group individual samples into batches of n_samples size
                    for i in range(0, len(rollout_samples), self.args.n_samples_per_prompt):
                        batch_samples = rollout_samples[i : i + self.args.n_samples_per_prompt]
                        if len(batch_samples) < self.args.n_samples_per_prompt:
                            continue

                        # Calculate average reward for this batch of samples
                        avg_reward = sum(sample.scores[0].item() for sample in batch_samples) / len(batch_samples)

                        # Check if average reward is within the specified range
                        min_reward, max_reward = self.args.dynamic_filtering_reward_range
                        if min_reward + 1e-6 < avg_reward < max_reward - 1e-6:
                            filtered_samples.extend(batch_samples)

                    # Continue sampling if filtered samples are insufficient
                    if len(filtered_samples) / self.args.n_samples_per_prompt < self.args.rollout_batch_size:
                        logger.info(
                            f"filtered_samples {len(filtered_samples) / self.args.n_samples_per_prompt} < rollout_batch_size {self.args.rollout_batch_size}, continue sampling"
                        )
                        continue

                    pass_rate = len(filtered_samples) / number_of_samples * 100
                    logger.info(
                        f"Dynamic filtering pass rate: {pass_rate:.2f}% ({len(filtered_samples)}/{number_of_samples})"
                    )
                    rollout_samples = filtered_samples[: self.args.rollout_batch_size * self.args.n_samples_per_prompt]
                    filtered_samples = []
                    number_of_samples = 0
                # TODO lpf：增加多智能体下对micro rollout batch size的判断，必须为1
                print("lpf 收集rollout 完毕 开始make experience")
                experiences = self.experience_maker.make_experience_batch(rollout_samples)# 只支持micro batch size =1；方便处理由不同agent构成的组进行

                # TODO lpf：check多智能体如何支持dynamic batch，不好适配的话先搁置
                # balance experiences across dp
                if args.use_dynamic_batch:
                    experiences = balance_experiences(experiences, args)

                # 将experiences处理成list of agent experiences，分发数据到不同agent
                # lpf: get List(List(Experience) of agent)，将不同agent的List(Experience) 通过append 分发到对应的缓存
                experiences = self.experience_maker.get_agent_experiences(experiences)
                refs = []
                for idx in range(self.num_agents):
                    refs.extend(
                        self.agent_list[idx].actor_model_group.async_run_method_batch(method_name="append", experience=experiences[idx])
                    )
                    if self.agent_list[idx].critic_model_group is not None:
                        refs.extend(
                            self.agent_list[idx].critic_model_group.async_run_method_batch(method_name="append", experience=experiences[idx])
                        )
                ray.get(refs)


                # TODO: 使用agent_list并行进行训练
                agents_status = self.ppo_train(steps)  # 返回的status格式为: {"agent_0": status, ...}

                for agent_key, status in agents_status.items():
                    agent_idx = int(agent_key.split("_")[1])  # 从 "agent_0" → 0
                    if "kl" in status:
                        self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)

                    # Add generated samples to status dictionary
                    if self.args.dynamic_filtering:
                        status["dynamic_filtering_pass_rate"] = pass_rate
                    logger.info(f"✨ Global step {steps}: {status}")
                    sample0 = self.tokenizer_list[agent_idx].batch_decode(
                        experiences[agent_idx][0].sequences[0].unsqueeze(0), skip_special_tokens=True
                    )
                    # status["generated_samples"] = [sample0[0], experiences[agent_idx][0].info["reward"][0]]
                    status["generated_samples"] = [sample0[0], experiences[agent_idx][0].info["reward"]]

                # logs/checkpoints
                client_states = {
                    "global_step": steps,
                    "episode": episode,
                    "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                }
                self.save_logs_and_checkpoints(args, steps, pbar, agents_status, client_states)

                steps = steps + 1

        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()

    def ppo_train(self, global_steps):
        all_status = {}

        # triger remote critic model training
        reload_critic_refs, critic_refs, critic_agents, offload_critic_refs = [], [], [], []
        num_critic_refs = 0
        for agent_idx, agent in enumerate(self.agent_list):
            if agent.critic_model_group is not None:
                if self.strategy.args.deepspeed_enable_sleep:
                    reload_critic_refs.extend(agent.critic_model_group.async_run_method(method_name="reload_states"))
                critic_refs.extend(
                    agent.critic_model_group.async_run_method(method_name="fit")
                )
                critic_agents.append((agent_idx, num_critic_refs))
                num_critic_refs = len(critic_refs)
                if self.strategy.args.deepspeed_enable_sleep:
                    offload_critic_refs.extend(agent.critic_model_group.async_run_method(method_name="offload_states"))

        if reload_critic_refs and self.strategy.args.deepspeed_enable_sleep:
            ray.get(reload_critic_refs)
        if critic_refs and (self.strategy.args.colocate_all_models or self.strategy.args.deepspeed_enable_sleep):
            critic_results = ray.get(critic_refs)
            for agent_idx, start_idx in critic_agents:
                all_status.setdefault(f"agent_{agent_idx}", {}).update(critic_results[start_idx])
        if offload_critic_refs and self.strategy.args.deepspeed_enable_sleep:
            ray.get(offload_critic_refs)

        # actor model training
        reload_actor_refs, actor_refs, actor_agents, offload_actor_refs = [], [], [], []
        num_actor_refs = 0
        for agent_idx, agent in enumerate(self.agent_list):
            if global_steps > self.freezing_actor_steps:
                if self.strategy.args.deepspeed_enable_sleep:
                    reload_actor_refs.extend(agent.actor_model_group.async_run_method(method_name="reload_states"))
                actor_refs.extend(
                    agent.actor_model_group.async_run_method(method_name="fit", kl_ctl=self.kl_ctl.value)
                )
                actor_agents.append((agent_idx, num_actor_refs))
                num_actor_refs = len(actor_refs)
                if self.strategy.args.deepspeed_enable_sleep:
                    offload_actor_refs.extend(agent.actor_model_group.async_run_method(method_name="offload_states"))

        if reload_actor_refs and self.strategy.args.deepspeed_enable_sleep:
            ray.get(reload_actor_refs)
        if actor_refs:
            actor_results = ray.get(actor_refs)
            offset = 0
            for agent_idx, start_idx in actor_agents:
                all_status.setdefault(f"agent_{agent_idx}", {}).update(actor_results[start_idx])
        if offload_actor_refs and self.strategy.args.deepspeed_enable_sleep:
            ray.get(offload_actor_refs)

        # 4. broadcast weights to vllm engines
        if global_steps > self.freezing_actor_steps:
            for agent in self.agent_list:
                if getattr(agent, "vllm_engines", None):
                    self._broadcast_to_vllm(agent)

        # 5. wait remote critic model training done
        if critic_refs and not self.strategy.args.colocate_all_models:
            critic_results = ray.get(critic_refs)
            for agent_idx, result in zip(critic_agents, critic_results):
                all_status.setdefault(f"agent_{agent_idx}", {}).update(result[0])

        return all_status

    def _broadcast_to_vllm(self, agent):
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(agent.vllm_engines, "wake_up")

        ray.get(agent.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(agent.vllm_engines, "sleep")


    def _init_wandb(self):
        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        # 原来是单表，这里改成每个 agent 一张表
        self.generated_samples_table_map = {}

        if self.strategy.args.use_wandb:
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=self.strategy.args.use_wandb)

            wandb.init(
                entity=self.strategy.args.wandb_org,
                project=self.strategy.args.wandb_project,
                group=self.strategy.args.wandb_group,
                name=self.strategy.args.wandb_run_name,
                config=self.strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)
            # 不再创建单一表；各 agent 的表在第一次写入时再初始化
            # self.generated_samples_table = wandb.Table(columns=["global_step", "text", "reward"])

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, self.strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def evaluate(self, eval_dataloader, global_step, temperature=0.6, n_samples_per_prompt=1):
        """Evaluate model performance on eval dataset.

        Args:
            eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
            global_step: Current training step for logging
            n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
        """
        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # vLLM wakeup when vllm_enable_sleep
        # if self.strategy.args.vllm_enable_sleep:
        #     from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
        #     for agent in self.agent_list:
        #         batch_vllm_engine_call(agent.vllm_engines, "wake_up")

        with torch.no_grad():
            # First collect all prompts and labels
            all_prompts = []
            all_labels = []
            all_metadatas = []
            prompt_to_datasource = {}  # Dictionary to store mapping between prompts and their data sources

            for datasources, prompts, labels, metadatas in eval_dataloader:
                all_prompts.extend(prompts)
                all_labels.extend(labels)
                all_metadatas.extend(metadatas)
                # Create mapping for each prompt to its corresponding data source
                for prompt, datasource in zip(prompts, datasources):
                    prompt_to_datasource[prompt] = datasource

            # Generate samples and calculate rewards
            generate_kwargs = self.generate_kwargs.copy()
            generate_kwargs["temperature"] = temperature
            generate_kwargs["n_samples_per_prompt"] = n_samples_per_prompt
            samples_list = self.samples_generator.generate_samples(
                all_prompts, all_labels, all_metadatas, remote_reward_model=self.remote_reward_model, is_eval=True, **generate_kwargs
            )

            # Only for mcts system evaluate
            samples_list = samples_list[::8]

            # duplicate prompts and labels for each sample
            all_prompts = sum([s.prompts for s in samples_list], [])
            all_labels = sum([s.labels for s in samples_list], [])

            # Get rewards from samples, such as agent rewards or remote reward models
            rewards_list = []
            for samples in samples_list:
                rewards_list.append(samples.rewards)# 接受mcts系统的pass@ 1和pass@ k
            # Reshape rewards to (num_prompts, n_samples_per_prompt)
            # TODO 添加mncts workflow 每个prompt的节点数量
            # Only for mcts system evaluate
            rewards = torch.tensor(rewards_list).reshape(-1, n_samples_per_prompt, 2)
            # rewards = torch.tensor(rewards_list).reshape(-1, n_samples_per_prompt)

            # Collect local statistics for each data source
            global_metrics = {}  # {datasource: {"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}}

            # Process rewards in chunks of n_samples_per_prompt
            num_prompts = len(all_prompts) // n_samples_per_prompt
            for i in range(num_prompts):
                # Get the original prompt (first one in the chunk)
                original_prompt = all_prompts[i * n_samples_per_prompt]
                datasource = prompt_to_datasource[original_prompt]  # Get corresponding data source using the mapping

                if datasource not in global_metrics:
                    global_metrics[datasource] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}

                # Get rewards for this chunk
                chunk_rewards = rewards[i]

                # Calculate pass@k and pass@1
                if n_samples_per_prompt > 1:
                    global_metrics[datasource][f"pass{n_samples_per_prompt}"] += chunk_rewards[:, 1].max().float().item()
                global_metrics[datasource]["pass1"] += chunk_rewards[:, 0].mean().float().item()
                global_metrics[datasource]["count"] += 1

            # Calculate global averages
            logs = {}
            for datasource, metrics in global_metrics.items():
                logs[f"eval_{datasource}_pass{n_samples_per_prompt}"] = (
                    metrics[f"pass{n_samples_per_prompt}"] / metrics["count"]
                )
                logs[f"eval_{datasource}_pass1"] = metrics["pass1"] / metrics["count"]

            # Log to wandb/tensorboard
            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        # if self.strategy.args.vllm_enable_sleep:
        #     for agent in self.agent_list:
        #         batch_vllm_engine_call(agent.vllm_engines, "sleep")

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

    # TODO 需要修改agent的ckpt和logs的保存策略
    def save_logs_and_checkpoints(self, args, global_step, step_bar, agents_status={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None:
                # Add generated samples to wandb using Table
                # 逐 agent 记录
                for agent_key, logs_dict in agents_status.items():
                    if "generated_samples" in logs_dict:
                        table = self.generated_samples_table_map.get(agent_key, None)
                        if table is None:
                            table = self._wandb.Table(columns=["global_step", "text", "reward"])
                        
                        new_table = self._wandb.Table(columns=table.columns, data=table.data)
                        new_table.add_data(global_step, *logs_dict.pop("generated_samples"))
                        self.generated_samples_table_map[agent_key] = new_table

                        self._wandb.log({f"train/{agent_key}/generated_samples": new_table})

                    logs = {
                        f"train/{agent_key}/{k}": v
                        for k, v in {
                            **logs_dict,
                            "global_step": global_step,
                        }.items()
                    }
                    self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None:
                for agent_key, logs_dict in agents_status.items():
                    for k, v in logs_dict.items():
                        if k == "generated_samples":
                            # Record generated samples in TensorBoard using simple text format
                            text, reward = v
                            formatted_text = f"Sample:\n{text}\n\nReward: {reward:.4f}"
                            self._tensorboard.add_text(f"train/{agent_key}/generated_samples", formatted_text, global_step)
                        else:
                            self._tensorboard.add_scalar(f"train/{agent_key}/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0 and self.eval_dataloader and len(self.eval_dataloader) > 0:
            self.evaluate(self.eval_dataloader, global_step, args.eval_temperature, args.eval_n_samples_per_prompt)
        # 保存所有agent的ckpt
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            # 逐 agent 启动保存（先发起，后统一 ray.get）
            refs = []
            for agent_idx, agent in enumerate(getattr(self, "agent_list", [])):
                refs.extend(agent.actor_model_group.async_run_method(
                    method_name="save_checkpoint", tag=tag, client_states=client_states
                ))
                if agent.critic_model_group is not None:
                    refs.extend(agent.critic_model_group.async_run_method(method_name="save_checkpoint", tag=tag))
            ray.get(refs)