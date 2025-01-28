import time
import torch.nn as nn
import torch
from torch.optim import Adam

from src.logger import get_logger

logger = get_logger(__name__)


def train_and_evaluate(model, dataloader, epochs=2, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu"):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    total_start = time.time()

    for epoch in range(epochs):
        total_loss = 0.0
        eval_loss = 0.0

        model.train()
        for input_ids, target_ids in dataloader:
            inputs, targets = input_ids.to(device), target_ids.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}")

        model.eval()
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
                eval_loss += loss.item()

        logger.info(f"Epoch {epoch+1}, Eval Loss: {eval_loss:.4f}")

    total_time = time.time() - total_start
    logger.info(f"Total training time: {total_time:.2f} seconds")
