import os
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np
import datasets
import transformers
from datasets import Dataset
import pandas as pd
import itertools
import random

from .io_utils import read_json, read_jsonl

class Task:

    @property
    @abstractmethod
    def num_choices(self) -> int:
        raise NotImplementedError()

    @property
    def drop_columns(self) -> list:
        """Returns list of columns to drop when tokenizing
        (Not really necessary, just reduces clutter in the batch objects)

        Don't include any of:
            label
            context
            query
            option_*

        :return: list columns to drop
        """
        return []

    @abstractmethod
    def standardize_examples(self, examples) -> dict:
        """Called by (batched) dataset method to convert data to standard format
        Output is a dict of lists, with the following types
            - context: str
            - query: str
            - label: int
            - option_[0..NUM_CHOICES]: str

        Ultimately, examples will be formatted as:
            context + query + option
        or
            context + [sep] + query + option

        with NO SPACES, so adjust accordingly (e.g. prepending space to query/options)

        :return: dict of lists
        """
        raise NotImplementedError()

    @abstractmethod
    def get_datasets(self) -> dict:
        """Returns dict (or dict-like) of datasets, with keys:
            train
            validation
            test

        :return: dict[str, Dataset]
        """
        raise NotImplementedError()

    # noinspection PyMethodMayBeStatic
    def compute_metrics(self, p: transformers.EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=-1)
        
        if preds.ndim < 3:
            return {"accuracy": (preds[:, 0] == p.label_ids[:, 0]).astype(np.float32).mean().item()}
        else:
            label_ids = p.label_ids
            total = 0
            num_correct = 0
            for idx, ex_labels in enumerate(label_ids):
                ex_labels[ex_labels == -100] = 1
                total += 1
                if (ex_labels == preds[idx]).all():
                    num_correct += 1
            return {'accuracy': num_correct / total}

# By Hyeonbin Hwang, @01-31-22
class ECQATask(Task):
    def get_datasets(self) -> list:
        return datasets.load_dataset('csv', data_files={
            'train': "/workspace/expl/ECQA-Dataset/cqa_data_train.csv", 
            "validation":  "/workspace/expl/ECQA-Dataset/cqa_data_val.csv",
            'test': "/workspace/expl/ECQA-Dataset/cqa_data_test.csv"})
    
    @classmethod
    def standardize_examples(cls, examples):

        result = {
            "query": examples["q_text"],
            "label": []
        }

        for idx, ans in enumerate(examples['q_ans']):
            try:
                label_mappings = {examples[f'q_op{i+1}'][idx]: i + 1 for i in range(5)} 
                result["label"].append(label_mappings[ans])

            except:
                import pdb
                pdb.set_trace()
            
        return result

    @property
    def drop_columns(self) -> list:
        return []
    
    @property
    def num_choices(self) -> int:
        return 5

    def compute_metrics(self, p: transformers.EvalPrediction):

        # Manually change this for now.
        tokenizer = transformers.AutoTokenizer.from_pretrained("google/flan-t5-large")

        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=-1)
        
        TGT_ID = tokenizer.get_vocab()["."]

        if preds.ndim < 3:
            # return {"accuracy": (preds[:, 0] == p.label_ids[:, 0]).astype(np.float32).mean().item()}
            acc = 0
            for idx, x in enumerate(preds):
                # Answer and Expl is separated by semicolon.
                # Acc only ruled by Answer choice.

                try:
                    target_idx = np.where(p.label_ids[idx] == TGT_ID)[0][0]
                    pred_idx = np.where(preds[idx] == TGT_ID)[0][0]

                    if (preds[idx][:pred_idx] == p.label_ids[idx][:target_idx]).all():
                        acc += 1

                except Exception as e:
                    print("error occurred:", e)
                    continue
            
            n  = random.randint(0, len(preds) - 1)
            neg_idx = np.where(p.label_ids[n] == -100)[0][0]
            
            print("[Predicted]\n", tokenizer.decode(preds[n]))
            print("\n[ANSWER]\n", tokenizer.decode(p.label_ids[n][:neg_idx]))
            
            return {"accuracy": acc / len(preds)}
        else:
            label_ids = p.label_ids
            total = 0
            num_correct = 0
            for idx, ex_labels in enumerate(label_ids):
                ex_labels[ex_labels == -100] = 1
                total += 1
                if (ex_labels == preds[idx]).all():
                    num_correct += 1
            return {'accuracy': num_correct / total}

class CustomJSONLTask(Task):
    def __init__(self, base_path, num_choices, drop_columns=None):
        self.base_path = base_path
        self._drop_columns = drop_columns if drop_columns else []
        self._num_choices = num_choices

    @property
    def drop_columns(self) -> list:
        return self._drop_columns

    @property
    def num_choices(self) -> int:
        return self._num_choices

    @classmethod
    def standardize_examples(cls, examples):
        # jsonl data should already be preformatted to have keys
        #    context
        #    query
        #    label
        #    option_*
        return examples

    def get_datasets(self) -> dict:
        phases = ["train", "validation", "test"]
        dataset_dict = {}
        for phase in phases:
            phase_path = os.path.join(self.base_path, f"{phase}.jsonl")
            if not os.path.exists(phase_path):
                continue
            dataset_dict[phase] = datasets.load_dataset(
                "json",
                data_files=phase_path,
            )["train"]  # <- yes this is weird
 
        return dataset_dict

    

    # ADDED ON JAUNUARY 2nd.
    def get_custom_datasets(self) -> dict:

        # Assume self.base_path is:
        # /data/processed:

        # and the directory structure:
        #   - option-0_based
        #   - option-1_based
        #   - option-2_based
        #   - option-3_based

        def merge_samples(samples, dataset, idx):

            if len(dataset) == 0:
                return samples
            else:
                assert samples['query'] == dataset[idx]['query'], "[ERROR] Dataset and Sample don't match."
                return {**dataset[idx], **samples}


        # Create a list of all the paths to the different options
        paths = [os.path.join(self.base_path, f"option-{i}_based") for i in range(4)]
        
        # Initialize an empty dictionary to store the datasets
        dataset_dict = {}
        
        # Iterate over the paths
        for path, option_index in zip(paths, range(4)):
            # Check if the path exists

            if not os.path.exists(path):
                continue
            
            # Iterate over the phases
            for phase in ["train", "validation", "test"]:
                # Construct the full path to the phase file
                phase_path = os.path.join(path, f"{phase}.jsonl")
                
                # Check if the phase file exists
                if not os.path.exists(phase_path):
                    continue
                
                # Load the dataset
                dataset = datasets.load_dataset("json", data_files=phase_path)["train"]
                
                # Modify the context key and add the option field
                dataset = dataset.map(lambda x: {
                    "query": x["query"],
                    "label": x["label"],
                    f"context_{option_index}": x["context"],
                })
                
                # Convert the dataset to a list of dictionaries
                dataset_list = [x for x in dataset]
                
                # Group the samples by query and merge them
                if phase in dataset_dict:
                    dataset_dict[phase] = [merge_samples(group, dataset_dict[phase], idx) for idx, group in enumerate(dataset_list)]
                else:
                   dataset_dict[phase] = [merge_samples(group, [], idx) for idx, group in enumerate(dataset_list)]

        # Change list of jsons to dataset.
        for key in dataset_dict:
            dataset_dict[key] = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset_dict[key]))
        return dataset_dict
   

    @classmethod
    def create_from_path(cls, base_path):
        config = read_json(os.path.join(base_path, "config.json"))
        return cls(
            base_path=base_path,
            num_choices=config["num_choices"],
            drop_columns=config.get("drop_columns", []),
        )

def prepend_space(list_of_strings: list) -> list:
    return [" " + x for x in list_of_strings]


@dataclass
class TaskArguments:
    task_name: str = field(
        metadata={"help": "Task name (e.g. CosmosQA, CustomJSONLTask)"}
    )
    task_base_path: str = field(
        metadata={"help": "Path to data from CustomJSONLTask"},
        default=None,
    )


def get_task(task_args: TaskArguments):
    if task_args.task_name == "custom":
        return CustomJSONLTask.create_from_path(base_path=task_args.task_base_path)
    task_dict = {
        "ecqa": ECQATask,
    }
    return task_dict[task_args.task_name]()
