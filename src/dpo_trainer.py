# coding: utf-8

"""
Author: Du Mingzhe (mingzhe@nus.edu.sg)
Date: 10/06/2024
Description: DPO Trainer
"""

# 0. Library Import
import os
import torch
from peft import LoraConfig
from typing import Dict, Optional
from accelerate import Accelerator
from trl import DPOConfig, DPOTrainer
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, BitsAndBytesConfig, set_seed

# 1. Define and Parse Arguments.
@dataclass
class ScriptArguments:
    # DPO Hyperparameter
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # Model Parameters
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B", metadata={"help": "the location of the SFT model name or path"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "whether to load the model in 8bit"})
    model_dtype: Optional[str] = field(default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."})

    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "the number of gradient accumulation steps"})

    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "whether to use gradient checkpointing"})
    gradient_checkpointing_use_reentrant: Optional[bool] = field(default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"})

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=600, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=200, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=200, metadata={"help": "the evaluation frequency"})

    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(default="wandb", metadata={"help": 'The list of integrations to report the results and logs to.'})
    output_dir: Optional[str] = field(default="./data/dpo", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=5, metadata={"help": "the logging frequency"})
    ignore_bias_buffers: Optional[bool] = field(default=False, metadata={"help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation."})
    seed: Optional[int] = field(default=0, metadata={"help": "Random seed that will be set at the beginning of training."})

# 2. Data Format
def get_paired_data(sanity_check: bool = False, num_proc=24) -> Dataset:
    dataset = load_dataset("Elfsong/BBQ_DPO", split="religion")
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": samples["context"],
            "chosen": samples["win"],
            "rejected": samples["lose"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    set_seed(script_args.seed)

    # Model Precision
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=script_args.load_in_4bit,
        load_in_8bit=script_args.load_in_8bit, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Data Fsilter
    train_dataset = get_paired_data(sanity_check=script_args.sanity_check)
    train_dataset = train_dataset.filter(lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length)
    eval_dataset = get_paired_data(sanity_check=True)
    eval_dataset = eval_dataset.filter(lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length)

    # Initialize Training Arguments:
    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        eval_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="BBQ_DPO",
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        seed=script_args.seed,
    )

    # LoRA Configuration
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Initialize the DPO Trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 6. Train the Model
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. Save the Model
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)