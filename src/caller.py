# coding: utf-8

"""
Author: Du Mingzhe (mingzhe@nus.edu.sg)
Date: 27/05/2024
Description: A collection of caller for Huggingface / OpenAI
"""

from abc import ABC, abstractmethod
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

class Caller(ABC):
    @abstractmethod
    def generate(self, model_inputs: List[str]): pass 

class HF_Caller(Caller):
    def __init__(self, model_path: str, device_map: str) -> None:
        super().__init__()
        self.model_path = model_path
        self.device_map = device_map
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map=self.device_map)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    
    def generate(self, inputs: List[str]):
        model_inputs = self.tokenizer(inputs, return_tensors="pt").to(self.device_map)
        generated_ids = self.model.generate(**model_inputs)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    

# Unit Test
if __name__ == "__main__":
    hf_caller = HF_Caller(model_path="gpt2", device_map="cuda:3")
    outputs = hf_caller.generate(["def hello_world():"])
    print(outputs)