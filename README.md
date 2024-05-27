# DebiasODE

## Environment
```shell
# Conda Workspace
conda create -n dice python=3.10

# Libraries
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Evaluation Metrics
- A Hand-Built Bias Benchmark for Question Answering (BBQ)

## Methodology
- LoRA Fine-tune for each attribute.
- LoRA for all attributes.

## Log
- 2024.05.27: Setup the evaluation framework for HuggingFace models.
