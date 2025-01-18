import pickle


import torch
from torch.utils.data import Dataset


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
