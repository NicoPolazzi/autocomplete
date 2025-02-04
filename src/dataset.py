import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from collections import Counter
from src.logger import get_logger

logger = get_logger(__name__)

MIN_SEQUENCE_LENGTH = 10
CODEBERT_PATH = "microsoft/codebert-base"


class CodeDataset(Dataset):
    def __init__(self, max_length: int, max_samples: int, top_n_vocab: int = 10000):
        self.input_sequenceuences = []
        self.target_sequenceuences = []
        self.attention_masks = []

        self.tokenizer = AutoTokenizer.from_pretrained(CODEBERT_PATH)
        self.max_length = min(max_length, 512)
        self.max_samples = max_samples
        self.raw_dataset = load_dataset(
            "code_search_net", "python", split=f"train[:{self.max_samples}]", trust_remote_code=True
        )["func_code_tokens"]

        logger.info(f"Original CodeBERT vocabulary size: {self.tokenizer.vocab_size}")

        token_counter: Counter[str] = Counter()
        for tokens in self.raw_dataset:
            token_counter.update(tokens)

        top_tokens = set([token for token, _ in token_counter.most_common(top_n_vocab)])
        special_tokens = {
            self.tokenizer.pad_token,
            self.tokenizer.unk_token,
            self.tokenizer.bos_token,
            self.tokenizer.eos_token,
        }

        self.allowed_tokens = top_tokens.union(special_tokens)
        logger.info(f"Reduced vocabulary size (allowed tokens): {len(self.allowed_tokens)}")

        for tokens in self.raw_dataset:
            if len(tokens) < MIN_SEQUENCE_LENGTH:
                continue

            sequence = [self.tokenizer.bos_token] + tokens + [self.tokenizer.eos_token]
            sequence = sequence[: self.max_length]
            padding_length = self.max_length - len(sequence)
            if padding_length > 0:
                sequence = sequence + [self.tokenizer.pad_token] * padding_length

            sequence = [
                token if token in self.allowed_tokens else self.tokenizer.unk_token
                for token in sequence
            ]

            input_ids = self.tokenizer.convert_tokens_to_ids(sequence)
            # pad token at the end is needed to match the target sequence length
            target_ids = input_ids[1:] + [self.tokenizer.pad_token_id]
            attention_mask = [1] * (self.max_length - padding_length) + [0] * padding_length

            assert (
                len(input_ids) == self.max_length
            ), f"Input sequenceuence length mismatch: {len(input_ids)} vs {self.max_length}"
            assert (
                len(target_ids) == self.max_length
            ), f"Target sequenceuence length mismatch: {len(target_ids)} vs {self.max_length}"

            self.input_sequenceuences.append(input_ids)
            self.target_sequenceuences.append(target_ids)
            self.attention_masks.append(attention_mask)

        logger.info(f"Total samples: {self.__len__()}")

    def __len__(self):
        return len(self.input_sequenceuences)

    def __getitem__(self, idx: int):
        return {
            "input_ids": torch.tensor(self.input_sequenceuences[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "target_ids": torch.tensor(self.target_sequenceuences[idx], dtype=torch.long),
        }

    def _clean(self, code: str) -> str:
        code = code.replace("```python\n", "")
        return "\n".join(
            line
            for line in code.splitlines()
            if not line.strip().startswith("#") and not line.strip().startswith("```")
        )
