import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast, PreTrainedTokenizerFast
from datasets import load_dataset
from collections import Counter
from src.logger import get_logger

logger = get_logger(__name__)

MIN_SEQUENCE_LENGTH = 10
FREQUENCY_THRESHOLD = 5


class CodeDataset(Dataset):
    def __init__(self, max_length: int, max_samples: int):
        self.input_sequences = []
        self.target_sequences = []
        self.attention_masks = []

        self.raw_dataset = load_dataset(
            "code_search_net", "python", split=f"train[:{max_samples}]", trust_remote_code=True
        )["func_code_tokens"]

        self.tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")

        token_counter = Counter(token for tokens in self.raw_dataset for token in tokens)
        self.common_tokens = {
            token for token, count in token_counter.items() if count >= FREQUENCY_THRESHOLD
        }

        self.max_length = min(max_length, 512)
        for tokens in self.raw_dataset:
            if len(tokens) < MIN_SEQUENCE_LENGTH:
                continue

            tokens = [
                token if token in self.common_tokens else self.tokenizer.unk_token
                for token in tokens
            ]
            sequence = [self.tokenizer.bos_token] + tokens + [self.tokenizer.eos_token]
            sequence = sequence[: self.max_length]
            padding_length = self.max_length - len(sequence)
            if padding_length > 0:
                sequence = sequence + [self.tokenizer.pad_token] * padding_length

            input_ids = self.tokenizer.convert_tokens_to_ids(sequence)
            if self.tokenizer.unk_token_id in input_ids:
                continue
            target_ids = input_ids[-1]
            input_ids = input_ids[:-1]
            attention_mask = [1] * (self.max_length - padding_length - 1) + [0] * padding_length

            self.input_sequences.append(input_ids)
            self.target_sequences.append(target_ids)
            self.attention_masks.append(attention_mask)

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx: int):
        return {
            "input_ids": torch.tensor(self.input_sequences[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "target_ids": torch.tensor(self.target_sequences[idx], dtype=torch.long),
        }

    def decode_prediction(self, predicted_ids):
        return self.tokenizer.convert_ids_to_tokens(predicted_ids)
