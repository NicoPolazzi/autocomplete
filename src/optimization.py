import time
import torch.nn as nn
import torch
from torch.optim import Adam

from src.logger import get_logger

logger = get_logger(__name__)


def train_and_evaluate(loader, model, epochs=2, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu"):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    total_start = time.time()

    for epoch in range(epochs):
        total_loss = 0.0
        eval_loss = 0.0

        model.train()
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}")

        model.eval()
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = criterion(output, targets)
                eval_loss += loss.item()

        logger.info(f"Epoch {epoch+1}, Eval Loss: {eval_loss:.4f}")

    total_time = time.time() - total_start
    logger.info(f"Total training time: {total_time:.2f} seconds")
