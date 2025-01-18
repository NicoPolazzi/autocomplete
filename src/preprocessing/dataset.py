import pickle

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, Subset, random_split
from datasets import load_dataset  # type: ignore
from transformers import AutoTokenizer, AutoModel  # type: ignore

from src.preprocessing.tokenizer import (
    post_tokenization,
    remove_triple_backticks_and_comments,
)
from src.logger import get_logger

logger = get_logger(__name__)

PREPROCESSED_DATA_PATH = "data/preprocessed/"
TRAIN_DATASET_FILENAME = "train_tokenized_dataset.pkl"
TEST_DATASET_FILENAME = "test_tokenized_dataset.pkl"


class CodeDataset(Dataset):
    def __init__(self, dataset_path: str = "flytech/python-codes-25k") -> None:
        self.dataset_path = dataset_path
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")

        self.context_embeddings: list[Tensor] = []
        self.labels: list[int] = []

        for idx, code_snippet in enumerate(load_raw_dataset(self.dataset_path)[:100]):
            code_tokens = self.tokenizer.tokenize(
                remove_triple_backticks_and_comments(code_snippet)
            )
            tokens_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)

            with torch.no_grad():
                token_embeddings = self.model(torch.tensor(tokens_ids)[None, :])

            self.context_embeddings.append(token_embeddings[0])
            self.labels.append(idx % 10)  # Example: Assigning class labels cyclically

    def __len__(self) -> int:
        return len(self.context_embeddings)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        # The context_embeddings list contains tensors of shape (1, num_tokens, 768)

        input_sequence = self.context_embeddings[idx]
        label = self.labels[idx]
        return input_sequence, label

    def write_to_file(self, file_path: str) -> None:
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Dataset written to file {file_path} successfully")


def load_raw_dataset(dataset_path: str) -> list[str]:
    dataset = load_dataset(dataset_path, split="train")
    return dataset["output"]


def create_train_test_split(dataset: CodeDataset) -> tuple[Subset, Subset]:
    train_dataset, test_dataset = random_split(
        dataset, [0.90, 0.10], torch.Generator().manual_seed(42)
    )

    return train_dataset, test_dataset
