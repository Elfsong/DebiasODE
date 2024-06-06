# 🎲 DebiasODE

## Environment Setup
```shell
# Conda Workspace
conda create -n dice python=3.10

# Libraries
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

# HuggingFace Login
huggingface-cli login

# OpenAI API Key (optional)
export OPENAI_API_KEY=[YOUR_KEY]
```

# Model Downloading
```shell
# A better way to download HuggingFace models
export HF_HUB_ENABLE_HF_TRANSFER=1
model_name="meta-llama/Meta-Llama-3-70B"
huggingface-cli download $model_name --cache-dir /raid/hpc/mingzhe/transformers_cache/
```

## Evaluation
- A Hand-Built Bias Benchmark for Question Answering (BBQ)

## Methodology
- LoRA Fine-tune for each attribute.
- LoRA for all attributes.

## Log
- 2024.05.27: Setup the evaluation framework for HuggingFace models.
