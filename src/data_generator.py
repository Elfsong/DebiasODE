
import argparse
import pandas as pd
from tqdm import tqdm
from src.caller import OpenAI_Caller
from datasets import load_dataset, Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str)
parser.add_argument('--precentage', type=int, default=100)
args = parser.parse_args()

# Load LLM caller
openai_caller = OpenAI_Caller(model_name="gpt-3.5-turbo-0125", json_mode=False)

# Load datasets
category = args.category
precentage = args.precentage
print(f"Current category is [{category}] | Precentage is {precentage}%")
ds = load_dataset(path="Elfsong/BBQ", split=f'{category}[:{precentage}%]')

# Data Pairs
pairs = list()
for instance in tqdm(ds):
    context_condition = instance['context_condition']
    answer_label = instance['answer_label']
    target_label = instance['target_label']

    question = instance['question']
    context = instance['context']
    answers = {
        0: instance['ans0'],
        1: instance['ans1'],
        2: instance['ans2'],
    }

    sentences = [
        openai_caller.generate([f"Change the question and answer pair into a declarative sentence: {question} {answers[0]}"]),
        openai_caller.generate([f"Change the question and answer pair into a declarative sentence: {question} {answers[1]}"]),
        openai_caller.generate([f"Change the question and answer pair into a declarative sentence: {question} {answers[2]}"]),
    ]

    if context_condition == "ambig":
        anti_target_label = list(set([0, 1, 2]) - set([answer_label]) - set([target_label]))[0]
        pairs += [
            {
                "context": context,
                "win": sentences[answer_label],
                "lose": sentences[target_label]
            },
            {
                "context": context,
                "win": sentences[answer_label],
                "lose": sentences[anti_target_label]
            },
        ]
    else:
        anti_target_labels = list(set([0, 1, 2]) - set([answer_label]))
        pairs += [
            {
                "context": context,
                "win": sentences[answer_label],
                "lose": sentences[anti_target_labels[0]]
            },
            {
                "context": context,
                "win": sentences[answer_label],
                "lose": sentences[anti_target_labels[1]]
            },
        ]
Dataset.from_pandas(pd.DataFrame(pairs)).push_to_hub("Elfsong/BBQ_DPO", split=category)
print("Bingo!")





