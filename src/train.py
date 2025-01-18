from logging import getLogger
import time
import torch.nn as nn
import torch.optim as optim

logger = getLogger(__name__)


def train_model(model, dataloader, criterion, optimizer, num_epochs=10, report_every=5):
    all_losses = []
    model.train()

    start_time = time.time()
    logger.info(f"training on dataset with n = {len(dataloader.dataset)}")

    for epoch in range(1, num_epochs + 1):
        current_loss = 0

        for input_sequence, label in dataloader:
            optimizer.zero_grad()
            output = model(input_sequence)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()

        average_loss = current_loss / len(dataloader)
        all_losses.append(average_loss)

        if epoch % report_every == 0:
            logger.info(
                f"{epoch} ({epoch/num_epochs:.0%}): average batch loss = {average_loss}"
            )

    logger.info(f"training completed in {time.time() - start_time:.2f} seconds")
    return all_losses


def get_optimizer(model, learning_rate=0.001):
    return optim.Adam(model.parameters(), lr=learning_rate)


def get_criterion():
    return nn.CrossEntropyLoss()
