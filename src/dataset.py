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


class CodeDataset(Dataset):
    def __init__(self) -> None:
        self.context_embeddings, self.next_tokens_ids = get_embeddings_and_next_token_pairs()
        logger.info(f"dataset created with n = {len(self)}")

    def __len__(self) -> int:
        return len(self.context_embeddings)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        input_embedding = self.context_embeddings[idx]
        next_token_ids = torch.tensor(self.next_tokens_ids[idx], dtype=torch.long)
        return input_embedding, next_token_ids

    @staticmethod
    def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        # Stack inputs and target to [batch_size, sequence_length, hidden_size] and return them

        inputs, targets = zip(*batch)
        input_tensors = torch.stack(inputs, dim=0)
        target_tensors = torch.stack(targets, dim=0)
        return input_tensors, target_tensors

    def write_to_file(self, file_path: str) -> None:
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"dataset written to {file_path}")

    def create_train_evaluation_split(self, lengths: Sequence[float] = [0.90, 0.10]) -> tuple[Subset, Subset]:
        train_dataset, evaluation_dataset = random_split(self, lengths, torch.Generator().manual_seed(42))
        logger.info(
            f"train dataset created with n = {len(train_dataset)} and evaluation dataset created with n = {len(evaluation_dataset)}"
        )
        return train_dataset, evaluation_dataset


def get_dataloader(dataset: Dataset, batch_size: int = 32) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=CodeDataset.collate_fn)


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


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings, labels = zip(*batch)
    inputs = torch.stack(embeddings)
    targets = torch.stack(labels)
    return inputs, targets
