import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR

from src.logger import new_logger


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    tokenizer,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    logger = new_logger(__name__)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    time_start = time.time()

    for epoch in range(epochs):
        training_loss = _compute_training_loss(model, optimizer, criterion, train_loader, device)
        scheduler.step()
        eval_loss = _compute_validation_loss(model, criterion, validation_loader, device)
        logger.info(f"Epoch {epoch+1}, Train_loss: {training_loss:.4f}, Eval Loss: {eval_loss:.4f}")

    total_time = time.time() - time_start
    logger.info(f"Total training time: {total_time:.2f} seconds")


def _compute_training_loss(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.CrossEntropyLoss,
    train_loader: DataLoader,
    device: torch.device,
) -> float:
    training_loss = 0.0
    model.train()

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_ids = batch["target_ids"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        training_loss += loss.item()

    return training_loss


def _compute_validation_loss(
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    validation_loader: DataLoader,
    device: torch.device,
) -> float:
    eval_loss = 0.0
    model.eval()

    with torch.no_grad():
        for batch in validation_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            eval_loss += loss.item()

        return eval_loss
