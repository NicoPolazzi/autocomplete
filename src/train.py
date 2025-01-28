from logging import getLogger
import time
import torch

logger = getLogger(__name__)


def train_model(model, train_loader, criterion, optimizer, num_epochs=10, clip_grad_norm=1.0):
    device = torch.get_default_device()
    model.to(device)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Avg_loss: {avg_loss:.4f}")
