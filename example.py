import torch
from torch.utils.data import DataLoader
from src.dataset import CodeSnippetIterableDataset, collate_fn


def train_autocomplete_model():
    dataset = CodeSnippetIterableDataset(model_name="microsoft/codebert-base", max_samples=2000, context_length=50)
    data_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)  # small batch

    for batch_idx, (embs, labels) in enumerate(data_loader):
        print("Shape of embeddings:", embs.shape)
        print("Labels:", labels)

        # Here you can feed these embeddings into your next-layer model or decode them, etc.
        if batch_idx > 2:
            break


if __name__ == "__main__":
    train_autocomplete_model()
