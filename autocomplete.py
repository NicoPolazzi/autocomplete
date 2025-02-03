import argparse

import torch

from src.logger import get_logger
from src.dataset import CodeDataset
from src.model import CodeAutocompleteModel
from src.optimization import train_and_evaluate

from torch.utils.data import random_split, DataLoader


logger = get_logger(__name__)


batch_size = 32
epochs = 10
lr = 1e-5

hidden_dimension = 256
num_layers = 2


def main():
    # TODO: implement all command line choices
    parser = argparse.ArgumentParser(description="Python autocompletion")
    parser.add_argument("command", choices=["train", "evaluate", "inference"])
    args = parser.parse_args()

    logger.info("Welcome to the python autocomplete tool!")

    dataset = CodeDataset(max_length=8, max_samples=1000)

    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info("Data loaded successfully!")

    logger.info(f"Starting {args.command} procedure...")
    if args.command == "train":
        model = CodeAutocompleteModel(
            vocab_size=dataset.tokenizer.vocab_size,
            hidden_dim=hidden_dimension,
            num_layers=num_layers,
        )
        logger.info("Model created successfully!")
        train_and_evaluate(model, train_loader, val_loader, epochs, lr)

    logger.info("End of the python autocomplete tool!")


if __name__ == "__main__":
    main()
