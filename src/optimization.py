import time
import torch.nn as nn
import torch
from torch.optim import Adam

from src.logger import get_logger

logger = get_logger(__name__)


def train_and_evaluate(
    model,
    train_set,
    validation_set,
    epochs=2,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    total_start = time.time()

    for epoch in range(epochs):
        total_loss = 0.0
        eval_loss = 0.0

        model.train()
        for batch in train_set:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}")
        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}")

        model.eval()
        with torch.no_grad():
            for batch in validation_set:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_ids = batch["target_ids"].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                eval_loss += loss.item()

        logger.info(f"Epoch {epoch+1}, Eval Loss: {eval_loss:.4f}")
        print(f"Epoch {epoch+1}, Eval Loss: {eval_loss:.4f}")

    total_time = time.time() - total_start
    logger.info(f"Total training time: {total_time:.2f} seconds")
    print(f"Total training time: {total_time:.2f} seconds")
