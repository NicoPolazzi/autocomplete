import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from collections import Counter
from src.logger import get_logger

logger = get_logger(__name__)


class CodeDataset(Dataset):
    def __init__(self, max_length, max_samples):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.max_length = max_length
        self.max_samples = max_samples

        self.raw_dataset = load_dataset(
            "code_search_net", "python", split=f"train[:{self.max_samples}]", trust_remote_code=True
        )["whole_func_string"]

        logger.info(f"Using CodeBERT vocabulary size: {self.tokenizer.vocab_size}")

        self.input_sequences = []
        self.target_sequences = []
        self.attention_masks = []
        skipped = 0

        for item in self.raw_dataset:
            code = self._clean(item)
            tokens = self.tokenizer.tokenize(code)

            # Meaybe I can remove this check
            if len(tokens) > 512:
                skipped += 1
                continue

            for i in range(1, len(tokens)):
                if i > self.max_length:
                    break

                input_tokens = [self.tokenizer.bos_token] + tokens[:i]
                input_tokens = input_tokens[: self.max_length]
                padding_length = self.max_length - len(input_tokens)
                if padding_length > 0:
                    input_tokens = input_tokens + [self.tokenizer.pad_token] * padding_length

                input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
                target_id = self.tokenizer.convert_tokens_to_ids([tokens[i]])[0]
                attention_mask = [1] * (len(input_tokens) - padding_length) + [0] * padding_length

                assert (
                    len(input_ids) == self.max_length
                ), f"Input sequence length mismatch: {len(input_ids)} vs {self.max_length}"

                self.input_sequences.append(input_ids)
                self.target_sequences.append(target_id)
                self.attention_masks.append(attention_mask)
        logger.info(f"Skipped {skipped} sequences that exceeded maximum length")

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_sequences[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "target_ids": torch.tensor(self.target_sequences[idx], dtype=torch.long),
        }

    def _clean(self, code: str) -> str:
        code = code.replace("```python\n", "")
        return "\n".join(
            line
            for line in code.splitlines()
            if not line.strip().startswith("#") and not line.strip().startswith("```")
        )
