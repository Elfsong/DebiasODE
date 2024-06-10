# coding: utf-8

"""
Author: Du Mingzhe (mingzhe@nus.edu.sg)
Date: 09/06/2024
Description: Bias Trainer
"""

# Fine-Tune Llama2-7b on SE paired dataset

# 0. imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import torch
import datasets
from tqdm import tqdm
from typing import Optional
from accelerate import Accelerator
from accelerate import PartialState
from dataclasses import dataclass, field
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B", metadata={"help": "the model name"})
    max_seq_length: Optional[int] = field(default=4096, metadata={"help": "the maximum sequence length"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "Model 4 bit quant"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "Model 8 bit quant"})
    
    dataset_name: Optional[str] = field(default="Elfsong/BBQ_DPO", metadata={"help": "the dataset name"})
    split: Optional[str] = field(default="religion", metadata={"help": "the split to use"})
    
    max_steps: Optional[int] = field(default=100, metadata={"help": "the maximum number of training steps"})
    logging_steps: Optional[int] = field(default=4, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=200, metadata={"help": "the saving frequency"})

    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})

    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "whether to use gradient checkpointing"})

    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group by length"})
    packing: Optional[bool] = field(default=False, metadata={"help": "whether to use packing for SFTTrainer"})

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    output_dir: Optional[str] = field(default="data/checkpoints", metadata={"help": "the output directory"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")

print("Model Loading...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=script_args.load_in_4bit,
    load_in_8bit=script_args.load_in_8bit, 
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    token=True,
)
base_model.config.use_cache = False,
print("Model Loaded.")

peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_steps=script_args.num_warmup_steps,
    optim=script_args.optimizer_type,
    bf16=True,
    remove_unused_columns=True,
    run_name=f"bbq_sft_train_{script_args.model_name}",
    gradient_checkpointing=script_args.gradient_checkpointing,
)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['context'])):
        text = f"{example['context'][i]} {example['win'][i]}"
        output_texts.append(text)
    return output_texts

print("Downloading dataset...")
dataset = datasets.load_dataset(script_args.dataset_name, split=script_args.split).train_test_split(test_size=0.1)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    peft_config=peft_config,
    max_seq_length=script_args.max_seq_length,
    formatting_func=formatting_prompts_func,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()

output_dir = os.path.join(script_args.output_dir, f"{script_args.model_name}-sft-final_checkpoint")
trainer.save_model(output_dir)

model = trainer.model.merge_and_unload()
model.save_pretrained(output_dir)