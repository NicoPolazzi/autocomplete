import pickle
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset, random_split, DataLoader, IterableDataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from src.preprocess import get_embeddings_and_next_token_pairs
from src.logger import get_logger

logger = get_logger(__name__)


class CodeDataset(Dataset):
    def __init__(self, max_length, max_samples: int = 10000):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.max_length = max_length
        self.max_samples = max_samples

        self.input_sequences = []
        self.target_sequences = []
        self.raw_dataset = load_dataset("flytech/python-codes-25k", split=f"train[:{self.max_samples}]")["output"]

        for item in self.raw_dataset:
            code = self._clean(item)
            tokens = self.tokenizer.tokenize(code)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            for i in range(1, len(token_ids)):
                input_seq = token_ids[:i]
                target_seq = token_ids[i]

                if len(input_seq) > self.max_length:
                    input_seq = input_seq[-self.max_length :]

                self.input_sequences.append(input_seq)
                self.target_sequences.append(target_seq)

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        input_seq = self.input_sequences[idx]
        target_seq = self.target_sequences[idx]

        if len(input_seq) < self.max_length:
            input_seq = [self.tokenizer.pad_token_id] * (self.max_length - len(input_seq)) + input_seq

        return torch.tensor(input_seq), torch.tensor(target_seq)

    def _clean(self, code: str) -> str:
        code = code.replace("```python\n", "")
        return "\n".join(
            line
            for line in code.splitlines()
            if not line.strip().startswith("#") and not line.strip().startswith("```")
        )
