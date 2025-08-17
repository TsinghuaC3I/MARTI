import os
import ray
import hydra
import srsly
from omegaconf import DictConfig, OmegaConf
from vllm import SamplingParams
from marti.models.openai import OpenAIModel, FakeTokenizer
from marti.worlds.third_party.chain import MultiAgentChain
from marti.worlds.third_party.debate import MultiAgentDebate

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@hydra.main(config_path="../configs", config_name="default.yaml", version_base=None)
def train(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    for key, value in cfg.default_agent.items():
        cfg[key] = value

    assert os.path.exists(cfg.prompt_data), f"Prompt data file {cfg.prompt_data} does not exist."
    data = srsly.read_json(cfg.prompt_data)

    agent = {
        "llms": [OpenAIModel.remote(api_key=cfg.api_key, base_url=cfg.api_base_url, config={"model_name": cfg.api_model_name})],
        "tokenizer": FakeTokenizer(),
        "sampling_params": SamplingParams(temperature=cfg.temperature, max_tokens=cfg.generate_max_len),
        "is_reasoning_model": False
    }
    world = MultiAgentDebate(
        agent_list=[agent, agent, agent],
        template_id=cfg['workflow_args']['template_id'],
        sampling_params=SamplingParams(temperature=cfg.temperature, max_tokens=cfg.generate_max_len)
    )
    world.run([d[cfg.input_key] for d in data])

    for sample, output in zip(data, world.get_history()):
        sample[cfg.output_key] = output

    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)

    srsly.write_json(os.path.join(cfg.save_path, cfg.save_file), data)

if __name__ == "__main__":
    train()
