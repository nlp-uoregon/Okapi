import argparse
import os

import torch
from accelerate import Accelerator
from datasets import load_dataset

from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    logging,
    set_seed
)

from utils import Prompter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="bloom-7b1", help="LLaMa weights that is converted to huggingface format!")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=float, default=0.01)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", type=str, default='BloomBlock')

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=300)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--hub_model_id", type=str)
    parser.add_argument("--log_freq", default=10, type=int)
    parser.add_argument("--eval_freq", default=200, type=int)
    parser.add_argument("--save_freq", default=20000, type=int)

    return parser.parse_args()

def create_datasets(tokenizer, args):
    """Create the datasets for training and validation."""
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.seq_length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.seq_length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    prompter = Prompter()
    dataset = load_dataset('json', split=args.split, data_files=args.data_path)
    original_columns = dataset.column_names

    dataset = dataset.train_test_split(test_size=args.size_valid_set, seed=args.seed)

    train_data = dataset["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=128, remove_columns=original_columns)
    valid_data = dataset["test"].map(generate_and_tokenize_prompt, remove_columns=original_columns)
    # print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    return train_data, valid_data

def run_training(args, train_data, val_data, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().process_index},
        use_cache=False,
    )

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="steps",
        save_strategy="no",
        eval_steps=args.eval_freq,
        # save_steps=args.save_freq,
        save_total_limit=2,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        optim="adamw_torch",
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.95,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=True,
        weight_decay=args.weight_decay,
        ddp_find_unused_parameters=False,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=args.log_freq,
        report_to='wandb',
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap=args.fsdp_transformer_layer_cls_to_wrap,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True))

    # model.config.use_cache = False

    print("Training...")
    trainer.train()
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(args.output_dir, state_dict=cpu_state_dict)  # noqa
        tokenizer.save_pretrained(args.output_dir)

def main(args):
    print('Start config')
    config = AutoConfig.from_pretrained(args.model_path)
    architecture = config.architectures[0]
    print('Start tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print('End tokenizer')

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
        raise ValueError("We only support Llama and Bloom models"")

    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset, tokenizer)

if __name__ == "__main__":
    args = get_args()
    assert args.model_path != "", "Please provide the model path"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)