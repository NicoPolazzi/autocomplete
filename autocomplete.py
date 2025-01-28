import argparse

from src.logger import get_logger
from src.dataset import CodeSnippetIterableDataset, new_data_loader

logger = get_logger(__name__)


def main():
    # TODO: implement all command line choices
    parser = argparse.ArgumentParser(description="Python autocompletion")
    parser.add_argument("command", choices=["train", "evaluate", "inference"])
    args = parser.parse_args()

    logger.info("Welcome to the python autocomplete tool!")
    dataset = CodeSnippetIterableDataset(model_name="microsoft/codebert-base", max_samples=2000, context_length=50)
    data_loader = new_data_loader(dataset, batch_size=4)
    logger.info("Data loaded successfully!")

    logger.info(f"Starting {args.command} procedure...")

    logger.info("End of the python autocomplete tool!")


if __name__ == "__main__":
    main()
