import re

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import random_split, DataLoader

from src.logger import new_logger

logger = new_logger(__name__)

MIN_SEQUENCE_LENGTH = 50
CODEBERT_MODEL = "microsoft/codebert-base"

Sequence = list[int]
AttentionMask = list[int]


class CodeDataset(Dataset):
    def __init__(self, max_length: int, max_samples: int) -> None:
        self.input_sequences: list[Sequence] = []
        self.target_sequences: list[int] = []
        self.attention_masks: list[AttentionMask] = []
        self.tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
        self.max_length = min(max_length, self.tokenizer.model_max_length - 1)
        self.raw_dataset = load_dataset(
            "code_search_net", "python", split=f"train[:{max_samples}]", trust_remote_code=True
        )["func_code_string"]

        self._build_sequences()
        logger.info(f"Dataset created with {len(self)} samples")

    def __len__(self) -> int:
        return len(self.input_sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_sequences[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "target_ids": torch.tensor(self.target_sequences[idx], dtype=torch.long),
        }

    def _build_sequences(self) -> None:
        for func_code in self.raw_dataset:
            func_code = self._remove_comments(func_code)

            if self._short_sequence(func_code):
                continue

            input_ids = self.tokenizer.encode(
                func_code,
                max_length=self.max_length + 1,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
            )
            target_ids = input_ids[-1]
            input_ids = input_ids[:-1]
            attention_mask = [
                1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in input_ids
            ]

            self.input_sequences.append(input_ids)
            self.target_sequences.append(target_ids)
            self.attention_masks.append(attention_mask)

    def _remove_comments(self, code: str) -> str:
        code = re.sub(r"'''[\s\S]*?'''", "", code)
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        return re.sub(r"#.*", "", code)

    def _short_sequence(self, code: str) -> bool:
        return len(code.strip()) < MIN_SEQUENCE_LENGTH


def load_train_and_validation(
    dataset: CodeDataset, batch_size: int, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator(device).manual_seed(42)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    logger.info(
        f"Created train and validation datasets with sizes: {len(train_dataset)} and {len(val_dataset)}"
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, generator=generator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, generator=generator
    )

    return train_loader, val_loader
