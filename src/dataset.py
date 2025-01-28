import pickle
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset, random_split, DataLoader, IterableDataset
from transformers import RobertaTokenizer, RobertaModel
from datasets import load_dataset
from src.preprocess import get_embeddings_and_next_token_pairs
from src.logger import get_logger

logger = get_logger(__name__)


class CodeSnippetIterableDataset(IterableDataset):
    def __init__(self, model_name: str = "microsoft/codebert-base", max_samples: int = 10000, context_length: int = 50):
        super().__init__()
        self.model_name = model_name
        self.max_samples = max_samples
        self.context_length = context_length

        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaModel.from_pretrained(self.model_name).eval()

        self.raw_dataset = load_dataset("flytech/python-codes-25k", split=f"train[:{self.max_samples}]")

    def _clean(self, code: str) -> str:
        code = code.replace("```python\n", "")
        return "\n".join(
            line
            for line in code.splitlines()
            if not line.strip().startswith("#") and not line.strip().startswith("```")
        )

    def _create_pairs(self, tokens: list[int]) -> list[tuple[list[int], int]]:
        pairs = []

        for i in range(len(tokens) - self.context_length):
            context = tokens[i : i + self.context_length]
            next_token = tokens[i + self.context_length]
            pairs.append((context, next_token))
        return pairs

    def __iter__(self):
        with torch.no_grad():

            for item in self.raw_dataset:
                code = self._clean(item["output"])
                token_ids = self.tokenizer.encode(code, add_special_tokens=False)

                for context_ids, label_id in self._create_pairs(token_ids):
                    inputs = torch.tensor(context_ids)[None, :]
                    outputs = self.model(inputs).last_hidden_state.squeeze(0)
                    yield outputs, torch.tensor(label_id)


def new_data_loader(dataset: Dataset, batch_size: int = 4) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_fn, num_workers=4)


def _collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings, labels = zip(*batch)
    inputs = torch.stack(embeddings)
    targets = torch.stack(labels)
    return inputs, targets
