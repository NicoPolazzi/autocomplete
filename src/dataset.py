import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder  # type: ignore


def create_dataloader(tokens: list[str], vocab: dict, batch_size: int) -> DataLoader:
    dataset = _CodeDataset(tokens, vocab)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class _CodeDataset(Dataset):
    def __init__(self, tokens: list[str], vocab: dict):
        self.tokens = tokens
        self.vocab = vocab
        self.one_hot_encoded = _one_hot_encode(tokens, vocab)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tensor:
        return self.one_hot_encoded[idx]


def _one_hot_encode(tokens: list[str], vocab: dict) -> Tensor:
    encoder = OneHotEncoder(categories=[list(range(len(vocab)))], sparse_output=False)

    token_indices = [[vocab[token]] for token in tokens]
    one_hot_encoded = encoder.fit_transform(token_indices)

    return torch.tensor(one_hot_encoded, dtype=torch.float32)
