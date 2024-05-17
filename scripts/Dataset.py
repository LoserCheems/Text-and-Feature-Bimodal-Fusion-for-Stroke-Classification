from datasets import load_from_disk
import torch
from torch.utils.data import Dataset
import numpy as np
import math
import logging

logging.basicConfig(level=logging.INFO)

class HealthCare_Dataset(Dataset):
    def __init__(self, dataset_path, split, tokenizer, max_length=256, padding="max_length", truncation=True):
        self.dataset = load_from_disk(dataset_path)[split]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 编码
        text = self.tokenizer(self.dataset[idx]["text"], max_length=self.max_length, padding=self.padding, truncation=self.truncation)

        # 输入
        text_ids = torch.tensor(text["input_ids"], dtype=torch.long)
        text_attention_mask = torch.tensor(text["attention_mask"], dtype=torch.long)


        feature_ids = torch.tensor(self.dataset[idx]["feature"], dtype=torch.long)
        feature_attention_mask = torch.tensor([1]*len(feature_ids), dtype=torch.long)
        # 向左填充
        feature_ids = torch.nn.functional.pad(feature_ids, (0, self.max_length-feature_ids.size(0)), "constant", 0)
        feature_attention_mask = torch.nn.functional.pad(feature_attention_mask, (0, self.max_length-feature_attention_mask.size(0)), "constant", 0)


        # 类别标签
        label = torch.tensor(self.dataset[idx]["label"], dtype=torch.long)

        return {
            "text_ids": text_ids,
            "text_attention_mask": text_attention_mask,
            "feature_ids": feature_ids,
            "feature_attention_mask": feature_attention_mask,
            "label": label
        }