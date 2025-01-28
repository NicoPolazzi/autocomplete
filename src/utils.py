import pickle

import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


# Maybe not usefull
def set_device() -> None:
    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")

    torch.set_default_device(device)


def load_processed_dataset(dataset_path: str) -> Dataset:
    with open(dataset_path, "rb") as f:
        preprocessed_dataset = pickle.load(f)

    return preprocessed_dataset


def get_code_from_tokens(token_ids: list[int], tokenizer: RobertaTokenizer) -> str:
    decoded_string = tokenizer.decode(token_ids)
    special_tokens = set(tokenizer.special_tokens_map.values())
    cleaned_string = " ".join(word for word in decoded_string.split() if word not in special_tokens)
    cleaned_string = cleaned_string.replace("Ġ", " ")
    cleaned_string = cleaned_string.strip()
    return cleaned_string
