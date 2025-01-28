import argparse

from src.logger import get_logger
from src.dataset import CodeSnippetIterableDataset, new_data_loader
from src.model import AutocompleteModel
from src.optimization import train_and_evaluate


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
    if args.command == "train":
        model = AutocompleteModel(output_size=dataset.vocab_size)
        logger.info("Model created successfully!")
        train_and_evaluate(data_loader, model)

    logger.info("End of the python autocomplete tool!")


if __name__ == "__main__":
    main()
