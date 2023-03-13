import numpy as np

from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from typing import Dict

from .tasks import Task


def tokenize_examples_for_enc_dec_model(examples, tokenizer, max_seq_length: int,
                                        padding_strategy: PaddingStrategy,
                                        truncation_strategy: TruncationStrategy):
    option_keys = sorted([
        key for key in examples
        if key.startswith("q_op")
    ])
    input_strs = []
    target_strs = []

    include_expl = True
    ans_type = 'taskA_pos' # or 'taskA_pos', 'taskA_neg', 'taskB'

    for i in range(len(examples[option_keys[0]])):
        all_options = " ".join([f"- {j+1}. {examples[option_key][i]}\n" for j, option_key in enumerate(option_keys)])
        input_str = f"Question: {examples['query'][i]}\n{all_options}</s>"

        if include_expl:
            input_str = f"Answer to the following question by selecting the most appropriate option and generating a logical explanation.\nQuestion: {examples['query'][i]}\n{all_options}</s>"

        target_str = f"{examples['q_ans'][i]}. {examples[ans_type][i]}"
        input_strs.append(input_str)
        target_strs.append(target_str)
        
    tokenized_inputs = tokenizer(
        input_strs,
        max_length=max_seq_length,
        padding=padding_strategy,
        truncation=truncation_strategy,
        return_tensors="pt",
    )
    tokenized_targets = tokenizer(
        target_strs,
        max_length=max_seq_length,
        padding=padding_strategy,
        truncation=truncation_strategy,
        return_tensors="pt",
    )
    target_ids = tokenized_targets["input_ids"]
    target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100

    return {
        "input_ids": tokenized_inputs["input_ids"].numpy(),
        "attention_mask": tokenized_inputs["attention_mask"].numpy(),
        "labels": target_ids.numpy(),
    }


def tokenize_examples_for_mc_lm_model(examples, tokenizer, max_seq_length: int,
                                      padding_strategy: PaddingStrategy,
                                      truncation_strategy: TruncationStrategy):
    """
    Takes a dictionary of examples, with keys:
        context: str (before [SEP])
        query: str (after [SEP], can be empty)
        option_0: str
        option_1: str
        ...
        label: int
    """

    # This assumes option_keys sorted order corresponds labels order
    # which is fine for num_labels < 10
    
    # ASSUMING GPT2.

    option_keys = sorted([
        key for key in examples
        if key.startswith("q_op")
    ])

    results = []
    include_expl = True
    ans_type = 'taskA_pos' # or 'taskA_pos', 'taskA_neg', 'taskB'

    for i in range(len(examples[option_keys[0]])):
        all_options = "".join([f"- {j+1}. {examples[option_key][i]}\n" for j, option_key in enumerate(option_keys)])
        input_str = f"Question: {examples['query'][i]}\n{all_options}"

        if include_expl:
            input_str = f"Question: {examples['query'][i]}\n{all_options}\nAnswer:"
        
        input_len = len(input_str)
        input_str = input_str + f"\n{examples['q_ans'][i]}. {examples[ans_type][i]} <|endoftext|>"
        results.append(input_str)

    final_inputs = tokenizer(
        results,
        max_length=max_seq_length,
        padding=padding_strategy,
        truncation=truncation_strategy,
        return_tensors="pt",
    )

    return {
        "input_ids": final_inputs['input_ids'].numpy(),
        "attention_mask": final_inputs['attention_mask'].numpy()
    }


def get_tokenized_dataset(task: Task, dataset_dict,
                          tokenizer,
                          max_seq_length: int,
                          padding_strategy: PaddingStrategy,
                          truncation_strategy: TruncationStrategy,
                          model_mode: str,
                          ) -> Dict:

    tokenized_dataset = {}
    for phase in ["train", "validation", "test"]:
        if phase not in dataset_dict:
            continue

        # Make it into [num_choices, N]
        standard_examples = dataset_dict[phase].map(
            task.standardize_examples,
            batched=True,
            remove_columns=task.drop_columns,
        )

        if model_mode in ["mc", "generation"]:
            tokenize_examples = lambda examples: tokenize_examples_for_mc_lm_model(examples, tokenizer, max_seq_length,
                                             padding_strategy,
                                                                                   truncation_strategy)
        else:
            tokenize_examples = lambda examples: tokenize_examples_for_enc_dec_model(examples, tokenizer,
                                                                                     max_seq_length,
                                                                                     padding_strategy,
                                                                                     truncation_strategy)

        tokenized_examples = standard_examples.map(tokenize_examples, batched=True)
        tokenized_dataset[phase] = tokenized_examples

    return tokenized_dataset

## MANUALLY ADDED JANUARY 2ND
def get_custom_tokenized_dataset(task: Task, dataset_dict,
                          tokenizer,
                          max_seq_length: int,
                          padding_strategy: PaddingStrategy,
                          truncation_strategy: TruncationStrategy,
                          model_mode: str,c
                          ) -> Dict:
    
    tokenized_dataset = {}

    print("padding strategy", padding_strategy)
    print("truncation_stratgegy", truncation_strategy)

    for phase in ["train", "validation", "test"]:
        if phase not in dataset_dict:
            continue    

        standard_examples = dataset_dict[phase].map(
            task.standardize_examples,
            batched=True,
            remove_columns=task.drop_columns,
        )

        if model_mode in ["mc", "generation"]:
            # Only Difference is here!
            tokenize_examples = lambda examples: custom_tokenize_examples_for_mc_lm_model(examples, tokenizer, max_seq_length,
                                                                                   padding_strategy,
                                                                                   truncation_strategy)
        else:
            tokenize_examples = lambda examples: tokenize_examples_for_enc_dec_model(examples, tokenizer,
                                                                                     max_seq_length,
                                                                                     padding_strategy,
                                                                                     truncation_strategy)
        tokenized_examples = standard_examples.map(tokenize_examples, batched=True)
        tokenized_dataset[phase] = tokenized_examples
    return tokenized_dataset