import argparse
import os

from src.logger import new_logger
from src.dataset import CodeDataset, load_train_and_validation
from src.model import CodeAutocompleteRNN
from src.train import train_and_evaluate
import src.utils as utils


os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # disable parallelism early, needed to prevent deadlock warning (huggingface)
)


def main() -> None:
    logger = new_logger(__name__)
    logger.info("Welcome to the python autocomplete tool!")
    parser = argparse.ArgumentParser(description="Python autocompletion")
    parser.add_argument("command", choices=["train", "evaluate", "inference"])
    args = parser.parse_args()
    config = utils.load_config("config.yaml")
    device = utils.get_device()
    logger.info(f"Using device: {device}")
    dataset = CodeDataset(max_length=config["max_length"], max_samples=config["max_samples"])
    train_loader, validation_loader = load_train_and_validation(
        dataset, config["batch_size"], device
    )
    logger.info(f"Starting {args.command} procedure...")

    if args.command == "train":
        model = CodeAutocompleteRNN(
            dataset.tokenizer.vocab_size,
            config["embed_dimension"],
            config["hidden_dimension"],
            config["num_layers"],
            pad_token_id=dataset.tokenizer.pad_token_id,
        )
        train_and_evaluate(
            model,
            train_loader,
            validation_loader,
            dataset.tokenizer,
            config["epochs"],
            config["lr"],
            device,
        )

    logger.info("Goodbye from the python autocomplete tool!")


if __name__ == "__main__":
    main()
