import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import PreTrainedTokenizer

from src.utils import new_logger, plot_training_metrics, predict_next_tokens

logger = new_logger(__name__)


def train_and_validate(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    epochs: int,
    lr: float,
) -> None:
    """Train and validate a neural network model for code autocompletion.

    Executes the training loop with validation after each epoch. Tracks metrics including
    training loss, validation loss, accuracy, and perplexity. Implements learning rate
    scheduling and generates example completions during training.

    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): DataLoader containing training batches
        validation_loader (DataLoader): DataLoader containing validation batches
        tokenizer (PreTrainedTokenizer): Tokenizer used for text processing
        epochs (int): Number of training epochs
        lr (float): Initial learning rate for Adam optimizer

    Returns:
        None: Results are logged and plots are generated as side effects
    """

    device = torch.get_default_device()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    time_start = time.time()

    test_prompts = [
        "def calculate_",
        "class User:",
        "for item in ",
        "if __name__ == ",
    ]

    train_losses = []
    valid_losses = []
    accuracies = []
    perplexities = []

    for epoch in range(epochs):
        train_loss = _compute_training_loss(model, optimizer, criterion, train_loader, device)
        validation_loss, accuracy = _compute_validation_loss_and_accuracy(
            model, criterion, validation_loader, device
        )
        perplexity = math.exp(validation_loss) if validation_loss < 100 else float("inf")

        train_losses.append(train_loss)
        valid_losses.append(validation_loss)
        accuracies.append(accuracy)
        perplexities.append(perplexity)

        scheduler.step(validation_loss)
        logger.info(f"Epoch {epoch+1:02d}/{epochs}")
        logger.info(
            f"Train Loss: {train_loss:8.4f} | Eval Loss: {validation_loss:8.4f} | Perplexity: {perplexity:10.4f} | Accuracy: {accuracy:6.4f}"
        )

        for prompt in test_prompts:
            completion = predict_next_tokens(model, tokenizer, prompt, num_tokens=3)
            logger.info(f"Prompt: {prompt} -> Completion: {completion}")

    plot_training_metrics(train_losses, valid_losses, accuracies, perplexities)
    total_time = time.time() - time_start
    logger.info(f"Total training time: {total_time:.2f} seconds")


def _compute_training_loss(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    train_loader: DataLoader,
    device: torch.device,
) -> float:
    training_loss = 0.0
    model.train()
    max_grad_norm = (
        1.0  # max gradient norm for gradient clipping for stability and avoid exploding gradients
    )

    for inputs, targets in train_loader:
        input_ids = inputs.to(device)
        target_ids = targets.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        training_loss += loss.item()

    return training_loss / len(train_loader)


def _compute_validation_loss_and_accuracy(
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    validation_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    eval_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    model.eval()

    with torch.no_grad():
        for inputs, targets in validation_loader:
            input_ids = inputs.to(device)
            target_ids = targets.to(device)

            outputs = model(input_ids)
            batch_loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            eval_loss += batch_loss.item()

            predictions = outputs.argmax(dim=-1)
            mask = target_ids != model.pad_token_id
            correct_predictions += ((predictions == target_ids) & mask).sum().item()
            total_predictions += target_ids.numel()

    avg_loss = eval_loss / len(validation_loader)
    accuracy = correct_predictions / total_predictions if total_predictions else 0
    return avg_loss, accuracy
