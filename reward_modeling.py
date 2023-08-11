import os
os.environ["WANDB_PROJECT"] = "reward"

import argparse
from typing import Optional
from dataclasses import dataclass, field
from multiprocessing import cpu_count

import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch import inf
import datasets
from accelerate import Accelerator
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    set_seed,
    logging,
)

from utils import Prompter

def load_model(args):
    print('Loading model...')
    config = AutoConfig.from_pretrained(args.model_path)
    architecture = config.architectures[0]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    if "Llama" in architecture:
        print("Setting EOS, BOS, UNK, and PAD tokens for LLama tokenizer")
        tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>",
            }
        )
        tokenizer.pad_token_id = (
            0
        )
        args.fsdp_transformer_layer_cls_to_wrap = "LlamaDecoderLayer"
    elif 'Bloom' in architecture:
        args.fsdp_transformer_layer_cls_to_wrap = "BloomBlock"
    else:
        raise ValueError("We only support Llama and Bloom models")
        

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().process_index},
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True

    def freeze_layers(num_unfrozen: int):
        try:
            layers = model.transformer.h
        except:
            layers = model.model.layers
        for layer in layers[:-num_unfrozen]:
            layer.requires_grad_(False)

    freeze_layers(num_unfrozen=args.num_unfrozen_layers)

    return model, tokenizer

def load_dataset(args, tokenizer):
    def preprocess_function(examples):
        new_examples = {
            "input_ids_j": [],
            "attention_mask_j": [],
            "input_ids_k": [],
            "attention_mask_k": [],
        }
        for instruction, input, prefered_output, rejected_output in zip(examples['instruction'], examples['input'], examples['prefered_output'], examples['rejected_output']):
            tokenized_j = tokenizer(prompter.generate_prompt(
                    instruction,
                    input,
                    prefered_output,
                ), truncation=True, max_length=768)

            tokenized_k = tokenizer(prompter.generate_prompt(
                    instruction,
                    input,
                    rejected_output,
                ), truncation=True, max_length=768)

            new_examples["input_ids_j"].append(tokenized_j["input_ids"])
            new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
            new_examples["input_ids_k"].append(tokenized_k["input_ids"])
            new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

        return new_examples

    prompter = Prompter()
    dataset = datasets.load_dataset('json', split='train', data_files=args.data_path)
    original_columns = dataset.column_names

    dataset = dataset.train_test_split(test_size=args.size_valid_set, seed=args.seed)

    train_data = dataset["train"].shuffle().map(
        preprocess_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=original_columns)

    valid_data = dataset["test"].map(
        preprocess_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=original_columns)

    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    return train_data, valid_data

def run_training(args):
    model, tokenizer = load_model(args)
    train_data, val_data = load_dataset(args, tokenizer)

    @dataclass
    class DataCollatorReward:
        tokenizer: PreTrainedTokenizerBase
        pad_to_multiple_of: Optional[int] = None

        def __call__(self, data):
            batch = {}
            features_j, features_k = [], []
            for feature in data:
                features_j.append({
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                })

                features_k.append({
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                })

            batch_j = self.tokenizer.pad(
                features_j,
                padding=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            batch_k = self.tokenizer.pad(
                features_k,
                padding=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            batch = {
                "input_ids_j": batch_j["input_ids"],
                "attention_mask_j": batch_j["attention_mask"],
                "input_ids_k": batch_k["input_ids"],
                "attention_mask_k": batch_k["attention_mask"],
                "return_loss": True,
            }
            return batch

    class RewardTrainer(Trainer):
        # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
        def compute_loss(self, model, inputs, return_outputs=False):
            rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])['logits'].squeeze()
            rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])['logits'].squeeze()
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
            if return_outputs:
                return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
            return loss

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_freq,
        save_strategy="no",
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=not args.bf16,
        bf16=args.bf16,
        local_rank=args.local_rank,
        label_names=[],
        remove_unused_columns=False,
        weight_decay=args.weight_decay,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=args.log_freq,
        report_to='wandb',
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap=args.fsdp_transformer_layer_cls_to_wrap,
    )

    def compute_metrics(eval_preds):
        chosen_end_scores = eval_preds.predictions[0]  # chosen scores
        rejected_end_scores = eval_preds.predictions[1]  # rejected scores

        result = {}
        acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
        result["accuracy"] = acc.item()

        return result

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorReward(tokenizer, pad_to_multiple_of=8))

    trainer.train()
    print("Saving last checkpoint of the model")

    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(args.output_dir, state_dict=cpu_state_dict)  # noqa

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, default="bloom-7b1")
    parser.add_argument("--model_path", type=str, default="bloom-7b1",
                        help="LLaMa/Bloom weights that is converted to huggingface format!")
    parser.add_argument("--data_path", type=str, default="dataset/moose-ranking/vi.json")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=768)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_unfrozen_layers", type=int, default=12, help="The number of trainable layers.")

    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/reward_model")
    parser.add_argument("--log_freq", default=50, type=int)
    parser.add_argument("--eval_freq", default=500, type=int)
    # parser.add_argument("--save_freq", default=1000, type=int)
    parser.add_argument("--hub_model_id", default='reward.debug', type=str)
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    args.gradient_accumulation_steps = args.batch_size // (args.micro_batch_size*n_gpus)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    run_training(args)
