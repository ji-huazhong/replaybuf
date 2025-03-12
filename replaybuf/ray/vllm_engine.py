import ray
from vllm import LLM
from transformers import AutoTokenizer

from replaybuf.utils.deepspeed import DeepSpeedStrategy


@ray.remote
class VLLMRayActor:
    def __init__(
        self,
        actor_model_name_or_path: str,
        worker_cls: str,
        seed: int,
        max_model_len: int,
        dytpe: str = "bfloat16",
        trust_remote_code: bool = True,
        enforce_eager: bool = False,
        gpu_memory_utilization: float = 0.9,
        strategy: DeepSpeedStrategy = None,
        id: int = 0,
    ):
        strategy.set_seed(seed)
        self.strategy = strategy
        self.id = id
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            actor_model_name_or_path, trust_remote_code=trust_remote_code, use_fast=False,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.llm = LLM(
            model=actor_model_name_or_path,
            enforce_eager=enforce_eager,
            worker_cls=worker_cls,
            seed=seed + id,
            max_model_len=max_model_len,
            dtype=dytpe,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def generate_sequences(self, *, sampling_params, prompt_token_ids):
        return self.llm.generate_sequences(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
