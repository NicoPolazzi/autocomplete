import logging
from pathlib import Path
import yaml

import torch
from transformers import PreTrainedTokenizer, AutoTokenizer
import matplotlib.pyplot as plt


def new_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = new_logger(__name__)


def get_tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("microsoft/codebert-base")


def set_default_device() -> None:
    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")

    torch.set_default_device(device)
    logger.info(f"Default device set to {device}")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def predict_next_tokens(
    model: torch.nn.Module, tokenizer: PreTrainedTokenizer, snippet: str, num_tokens: int = 5
) -> list[str]:
    tokens = []

    for _ in range(num_tokens):
        inputs = tokenizer.encode(snippet, return_tensors="pt", max_length=32 - 1, truncation=True)
        predicted_token_id = model.predict_next_token_id(inputs)
        predicted_token = tokenizer.decode(predicted_token_id)
        snippet += predicted_token
        tokens.append(predicted_token)

    return tokens


def plot_training_metrics(
    train_losses: list[float],
    valid_losses: list[float],
    accuracies: list[float],
    perplexities: list[float],
) -> None:
    save_path = Path("plots")
    save_path.mkdir(exist_ok=True)
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_losses) + 1)

    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, valid_losses, "r-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(epochs, accuracies, "g-", label="Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(epochs, perplexities, "m-", label="Perplexity")
    plt.title("Model Perplexity")
    plt.xlabel("Epochs")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path / "training_metrics.png")
    plt.close()
