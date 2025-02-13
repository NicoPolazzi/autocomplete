import re

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from src.utils import new_logger

logger = new_logger(__name__)


class CodeDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, context_length: int, max_snippets: int, train: bool
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.train = train
        dataset = load_dataset("code_search_net", "python")

        # I assume always using the first 80% of the dataset for training and the last 20% for validation
        # I know that it could be better using a more sophisticated split for different dataset sizes
        if self.train:
            data = (
                dataset["train"]
                .select(range(int(max_snippets * 0.8)))
                .map(self._tokenize_function, batched=True)
            )
        else:
            data = (
                dataset["validation"]
                .select(range(int(max_snippets * 0.2)))
                .map(self._tokenize_function, batched=True)
            )

        self.input_ids = data["input_ids"]
        logger.info(f"Dataset created with {len(self)} samples")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.input_ids[idx]

        return (
            torch.tensor(sequence[:-1]),
            torch.tensor(sequence[1:]),
        )

    def get_loader(self, batch_size: int) -> DataLoader:
        generator = torch.Generator(device=torch.get_default_device()).manual_seed(42)

        if self.train:
            loader = DataLoader(
                self, batch_size=batch_size, shuffle=True, num_workers=8, generator=generator
            )
        else:
            loader = DataLoader(self, batch_size=batch_size, num_workers=8, generator=generator)

        logger.info(f"Loader created with {len(loader)} batches")
        return loader

    def _tokenize_function(self, examples: dict[str, list[str]]) -> dict[str, list[int]]:
        cleaned_code = [
            _clean_code(code)
            for code in examples["func_code_string"]
            if not self._short_sequence(code)
        ]

        return self.tokenizer(
            cleaned_code,
            max_length=self.context_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=False,
        )

    def _short_sequence(self, code: str) -> bool:
        return len(code.strip()) < self.context_length


def _clean_code(code: str) -> str:
    import_from_pattern = re.compile(r"^\s*(import|from)\s+.*$", re.MULTILINE)
    single_quote_doc = re.compile(r"'''[\s\S]*?'''")
    double_quote_doc = re.compile(r'"""[\s\S]*?"""')
    inline_comment = re.compile(r"#.*")
    code = import_from_pattern.sub("", code)
    code = single_quote_doc.sub("", code)
    code = double_quote_doc.sub("", code)
    code = inline_comment.sub("", code)
    code = "\n".join(line for line in code.splitlines() if not line.strip().startswith("#"))
    return code
