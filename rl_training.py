import os, json
import argparse
from typing import Optional, List

import torch
import torch.nn as nn

from accelerate import Accelerator

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    pipeline,
    set_seed,
)
import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig

from utils import Prompter
prompter = Prompter()

def get_model_trl_config(args):
    print("Setup config...")
    return TRLConfig(
        train=TrainConfig(
            seq_length=args.max_seq_length,
            epochs=args.num_epochs,
            total_steps=10000,
            batch_size=args.ppo_batch_size,
            eval_interval=args.eval_freq,
            minibatch_size=1,
            checkpoint_interval=10000,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            save_best=False,
            checkpoint_dir=args.output_dir,
        ),
        model=ModelConfig(
            model_path=args.sft_model_path,
            num_layers_unfrozen=args.num_layers_unfrozen,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_path=args.tokenizer_path,
            truncation_side='left',
            padding_side='left'
        ),
        optimizer=OptimizerConfig(
            name="adamw",
            kwargs=dict(lr=args.learning_rate, betas=[0.9, 0.95], eps=1.0e-8, weight_decay=args.weight_decay)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1.0e-6)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=args.num_rollouts,
            chunk_size=args.chunk_size,
            ppo_epochs=args.ppo_epochs,
            init_kl_coef=args.kl_coef, # should choose either 0.1 or 0.05
            target=6,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1.0,
            scale_reward=None,
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs={
                "max_new_tokens": args.max_new_tokens,
                "do_sample": True,
                "top_k": 0,
                "top_p": 1,
                "temperature": 1,
            })
    )

def create_reward_fn(args):
    if accelerator.is_main_process:
        print("Setup reward model")
        # if os.environ.get("RANK", "0") == "0":
        reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_tokenizer_path)
        config = AutoConfig.from_pretrained(args.reward_model_path)
        
        if "Llama" in config.architectures[0]:
            print("Setting EOS, BOS, UNK, and PAD tokens for LLama tokenizer")
            reward_tokenizer.add_special_tokens(
                {
                    "eos_token": "</s>",
                    "bos_token": "<s>",
                    "unk_token": "<unk>",
                }
            )
            reward_tokenizer.pad_token_id = (
                0
            )
            
        reward_tokenizer.truncation_side = "left"
        reward_tokenizer.padding_side = "left"

        reward_model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model_path,
            use_auth_token="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",
            torch_dtype=torch.bfloat16,
            # token="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"
        )
        reward_model.requires_grad_(False)
        reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
        reward_model.config.use_cache = True

        rm_device = torch.cuda.device_count() - 1
        print("Rewad device:", rm_device)
        reward_model.eval().to(rm_device)

        sigmoid_fn = nn.Sigmoid()
        def get_reward(samples: List[str]):
            all_scores = []
            for i in range(0, len(samples), args.rw_batch_size):
                batch = reward_tokenizer(
                    samples[i : i + args.rw_batch_size],
                    padding=True,
                    truncation=True,
                    max_length=768,
                    return_tensors="pt").to(rm_device)
                with torch.no_grad():
                    scores = reward_model(
                        batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )[0].squeeze(-1).cpu()
                all_scores.append(scores)
            scores = torch.hstack(all_scores)
            # scale reward scores
            # scores = 2*sigmoid_fn(scores)
            return scores

        def reward_fn(samples: List[str], original_output: List[str], **kwargs) -> torch.Tensor:
            rewards = get_reward(samples)
            return rewards

        return reward_fn
    else:
        return True

def create_datasets(args, save_test_set=False):
    print("Start create_datasets")
    def create_prompt(data_point):
        prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            # data_point["output"],
        )
        
        original_output = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        return {'prompt': prompt, 'original_output': original_output}

    prompter = Prompter()
    try:
        dataset = load_dataset('json', split=args.split, data_files=args.data_path)
    except:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            dataset = json.loads(f.read())
        for entry in dataset:
            for k, v in entry.items():
                if not isinstance(v, str):
                    entry[k] = str(v)
        dataset = Dataset.from_list(dataset)
    dataset = dataset.train_test_split(test_size=args.size_valid_set, seed=args.seed)

    train_prompts = dataset["train"].shuffle().map(create_prompt)
    valid_prompts = dataset["test"].map(create_prompt)

    train_prompts = [{'prompt': instance['prompt'], 'original_output': instance['original_output']} for instance in train_prompts]
    valid_prompts = [{'prompt': instance['prompt'], 'original_output': instance['original_output']} for instance in valid_prompts]

    print(f"Size of the train set: {len(train_prompts)}. Size of the validation set: {len(valid_prompts)}")

    if save_test_set:
        dataset["test"].to_json('dataset/val_data_hlhf.json', force_ascii=False)

    return train_prompts, valid_prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_model_path", type=str, default="bloom-560m",
                        help="LLaMa/Bloom weight that has trained by supervised finetuning")
    parser.add_argument("--tokenizer_path", type=str, default="bloom-560m",
                        help="LLaMa/Bloom tokenizer path")
    parser.add_argument("--data_path", type=str, default="dataset/data_rlhf.json")
    parser.add_argument("--reward_model_path", type=str, default="checkpoints/reward_model/checkpoint-7000")
    parser.add_argument("--reward_tokenizer_path", type=str, default="bloom-560m")
    parser.add_argument("--rw_batch_size", type=int, default=16)

    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=512)

    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--num_layers_unfrozen", type=int, default=5)
    parser.add_argument("--ppo_batch_size", type=int, default=2)
    parser.add_argument("--num_rollouts", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--kl_coef", type=int, default=0.1)

    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--save_freq", default=2000, type=int)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/rl_output_dir")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--hub_model_id", default='rlhf.debug', type=str)

    args = parser.parse_args()
    set_seed(args.seed)
    
    accelerator = Accelerator()
    
    reward_fn = create_reward_fn(args)

    config = get_model_trl_config(args)
    train_prompts, val_prompts = create_datasets(args, save_test_set=False)
    
    accelerator.wait_for_everyone()
    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts[:300],
        config=config,
    )
    # push to huggingface
    model = trainer.accelerator.unwrap_model(trainer.model)
    if accelerator.is_main_process:
        model.save_pretrained(args.output_dir)