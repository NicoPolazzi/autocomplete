import argparse

from src.logger import get_logger
from src.dataset import CodeDataset
from src.model import CodeAutocompleteModel
from src.optimization import train_and_evaluate

from torch.utils.data import DataLoader


logger = get_logger(__name__)


def main():
    # TODO: implement all command line choices
    parser = argparse.ArgumentParser(description="Python autocompletion")
    parser.add_argument("command", choices=["train", "evaluate", "inference"])
    args = parser.parse_args()

    logger.info("Welcome to the python autocomplete tool!")

    dataset = CodeDataset(max_length=128, max_samples=1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    logger.info("Data loaded successfully!")

    logger.info(f"Starting {args.command} procedure...")
    if args.command == "train":
        model = CodeAutocompleteModel()
        logger.info("Model created successfully!")
        train_and_evaluate(model, dataloader, epochs=10, lr=1e-3)

    logger.info("End of the python autocomplete tool!")


if __name__ == "__main__":
    main()
