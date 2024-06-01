# coding: utf-8

"""
Author: Du Mingzhe (mingzhe@nus.edu.sg)
Date: 27/05/2024
Description: A collection of caller for Huggingface / OpenAI
"""

import torch
from typing import List
from openai import OpenAI
from abc import ABC, abstractmethod
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


class Caller(ABC):
    @abstractmethod
    def generate(self, model_inputs: List[str]) -> List[str]: pass 

    @abstractmethod
    def prompt_generate(self, model_inputs: List[object]) -> List[str]: pass 

class HF_Caller(Caller):
    def __init__(self, model_path: str, device_map: str, max_new_token: int) -> None:
        super().__init__()
        self.model_path = model_path
        self.device_map = device_map
        self.max_new_token = max_new_token
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map=self.device_map, quantization_config=nf4_config, cache_dir="/raid/hpc/mingzhe/transformers_cache/")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, cache_dir="/raid/hpc/mingzhe/transformers_cache/")
    
    def stop_at_stop_token(self, stop_words, decoded_string: str) -> str:
        min_stop_index = len(decoded_string)
        for stop_token in stop_words:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]
    
    def generate(self, inputs: List[str]) -> List[str]:
        model_inputs = self.tokenizer(inputs, return_tensors="pt").to(self.device_map)
        generated_ids = self.model.generate(
            **model_inputs, 
            pad_token_id=self.tokenizer.eos_token_id, 
            max_new_tokens=self.max_new_token
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    def prompt_generate(self, model_inputs: List[object]) -> List[str]:
        model_prompts = list()
        for instance in model_inputs:
            model_prompts += [f"Provide your answer to the following question strictly using the JSON format {{'answer': '<upper_case_answer_id>'}}:\nContext: {instance['context']}\nQuestion: {instance['question']}\nA) {instance['ans0']}\nB) {instance['ans1']}\nC) {instance['ans2']}\n\nAnswer JSON:"]
        return model_prompts
    
class OpenAI_Caller(Caller):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.client = OpenAI()
        
    def generate(self, model_inputs: List[str]) -> List[str]:
        model_inputs = self.prompt_generate(model_inputs)
        model_outputs = list()
        for model_input in model_inputs:
            response = self.client.chat.completions.create(
                model = self.model_name,
                response_format = { "type": "json_object" },
                messages = model_input
            )
        model_outputs += [response.choices[0].message.content]
        return model_outputs
    
    def prompt_generate(self, model_inputs: List[object]) -> List[str]:
        model_inputs_with_prompt = list()
        for model_input in model_inputs:
            model_inputs_with_prompt += [[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON only."},
                {"role": "user", "content": f"{model_input}"}
            ]]
        return model_inputs_with_prompt

# Unit Test
if __name__ == "__main__":
    hf_caller = HF_Caller(model_path="gpt2", device_map="cuda:3", max_new_token=20)
    outputs = hf_caller.generate(["Hello,"])
    print(outputs)

    hf_caller = HF_Caller(model_path="meta-llama/Meta-Llama-3-8B", device_map="cuda:3", max_new_token=20)
    outputs = hf_caller.generate(["Hello,"])
    print(outputs)

    openai_caller = OpenAI_Caller(model_name="gpt-3.5-turbo-0125")
    outputs = openai_caller.generate(["Hello,"])
    print(outputs)