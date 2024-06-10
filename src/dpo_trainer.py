# coding: utf-8

"""
Author: Du Mingzhe (mingzhe@nus.edu.sg)
Date: 10/06/2024
Description: DPO Trainer
"""


# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed

from trl import DPOConfig, DPOTrainer
