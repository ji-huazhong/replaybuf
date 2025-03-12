import argparse
import random

import torch
import ray
from torch.utils.data import BatchSampler, Dataset, DataLoader, RandomSampler
from transformers import AutoTokenizer

from replaybuf import ACCELERATOR_TYPE
from replaybuf.utils import get_strategy, blending_datasets
from replaybuf.ray import VLLMRayActor
from replaybuf.datasets import PromptDataset


def _validate_args(args):
    assert args.vllm_num_engines > 0, "vLLM is disabled, set --vllm_num_engines > 0"


def prepare_datasets(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.actor_model_name_or_path, trust_remote_code=True, use_fast=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # prepare datasets
    prompts_data = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        args.seed,
        max_count=args.max_samples,
        return_eval=False,
        train_split=args.prompt_split,
    )
    prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    prompts_dataset = PromptDataset(
        prompts_data, tokenizer, args, input_template=args.input_template
    )
    return prompts_dataset


def main(args):
    _validate_args(args)

    # configure strategy
    strategy = get_strategy(args)

    # init vLLM engine for text generation
    vllm_engines = []
    # TODO: support tp
    for i in range(args.vllm_num_engines):
        vllm_engines.append(
            VLLMRayActor.options(
                num_cpus=0,
                resources={ACCELERATOR_TYPE: 1},
            ).remote(
                model=args.actor_model_name_or_path,
                worker_cls="replaybuf.ray.vllm_worker_wrap.WorkerWrap",
                seed=args.seed,
                max_model_len=args.prompt_max_len + args.generate_max_len,
                dtype="bfloat16",
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                strategy=strategy,
                engine_id=i,
            )
        )

    # prepare prompt datasets
    prompts_dataset = prepare_datasets(args)
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        seed = epoch
        random.seed(seed)
        torch.manual_seed(seed)

        # 创建随机采样点
        sampler = RandomSampler(prompts_dataset)
        batch_sampler = BatchSampler(sampler, batch_size=args.vllm_num_engines * , drop_last=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Ray and vLLM
    parser.add_argument("--actor_num_nodes", type=int, default=1, help="number of nodes for actor")
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=8, help="number of gpus per node for actor")
    parser.add_argument("--ref_num_nodes", type=int, default=1, help="number of nodes for reference")
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reference")
    parser.add_argument("--reward_num_nodes", type=int, default=1, help="number of nodes for reward model")
    parser.add_argument(
        "--reward_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reward model"
    )
    # optional vLLM for text generation
    parser.add_argument(
        "--vllm_num_engines", type=int, default=1, help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument("--enforce_eager", action="store_true", default=False, help="Disable CUDA graph in vLLM")
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization",
    )

    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")

    # GRPO
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per dp")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Reinforce
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo", "reinforce_baseline", "group_norm"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo, reinforce_baseline, group_norm",
    )
    parser.add_argument("--use_kl_loss", action="store_true", default=False, help="whether to use KL loss from GRPO")

    # Models
    parser.add_argument("--actor_model_name_or_path", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--label_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    args = parser.parse_args()

    main(args)
