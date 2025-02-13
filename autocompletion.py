import argparse
import os
from pathlib import Path

import src.utils as utils
from src.dataset import CodeDataset
from src.model import CodeAutocompleteRNN
from src.train import train_and_validate


os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # disable parallelism early, needed to prevent deadlock warning (Huggingface)
)

MODEL_DIRECTORY = Path("models")
MODEL_PATH = MODEL_DIRECTORY / "RNN_autocompletion.pt"


def main() -> None:
    logger = utils.new_logger(__name__)
    logger.info("Welcome to the python autocomplete tool!")
    parser = argparse.ArgumentParser(description="Python autocompletion")
    parser.add_argument("command", choices=["train", "inference"])
    parser.add_argument("--snippet", type=str, required=False)
    args = parser.parse_args()

    config = utils.load_config("config.yaml")
    utils.set_default_device()
    tokenizer = utils.get_tokenizer()
    pad_token_id = tokenizer.pad_token_id

    model = CodeAutocompleteRNN(
        tokenizer.vocab_size,
        config["embed_dimension"],
        config["hidden_dimension"],
        config["num_layers"],
        config["dropout"],
        pad_token_id=pad_token_id,
    )
    logger.info(f"Starting {args.command} procedure...")

    if args.command == "train":
        train_dataset = CodeDataset(
            tokenizer, config["context_length"], config["max_snippets"], train=True
        )
        validation_dataset = CodeDataset(
            tokenizer, config["context_length"], config["max_snippets"], train=False
        )
        train_loader = train_dataset.get_loader(config["batch_size"])
        validation_loader = validation_dataset.get_loader(config["batch_size"])

        train_and_validate(
            model,
            train_loader,
            validation_loader,
            tokenizer,
            config["epochs"],
            config["lr"],
        )
        model.save_to(MODEL_PATH)

    elif args.command == "inference":
        if not args.snippet:
            raise ValueError("The '--snippet' argument is required for inference.")

        model.load_weights_from(MODEL_PATH)
        tokens = utils.predict_next_tokens(model, tokenizer, args.snippet)
        logger.info(f"Predicted tokens: {tokens}")
    logger.info("Goodbye from the python autocomplete tool!")


if __name__ == "__main__":
    main()
