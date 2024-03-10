import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import Dataset

import pyarrow.parquet as pq
from datasets import load_from_disk

class SummarizationDataset(Dataset):

    def __init__(self, file_path, tokenizer, max_source_length, max_target_length):

        self.dataset = load_from_disk(file_path)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        source_text = item["content"]
        target_text = item["summary"]

        # Tokenize and encode the source and target texts
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding=True, #"max_length", # replace with padding=True
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding=True, #"max_length", # replace with padding=True
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": source_encoding.input_ids.squeeze(0),
            "attention_mask": source_encoding.attention_mask.squeeze(0),
            "labels": target_encoding.input_ids.squeeze(0)
        }
        