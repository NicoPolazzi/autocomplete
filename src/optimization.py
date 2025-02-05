import time
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer

from src.logger import get_logger

logger = get_logger(__name__)

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")


def train_and_evaluate(
    model,
    train_set,
    validation_set,
    epochs=2,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    model.to(device)
    total_start = time.time()

    for epoch in range(epochs):
        total_loss = 0.0

        model.train()
        for batch in train_set:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Evaluation phase with qualitative prediction logging
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(validation_set):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_ids = batch["target_ids"].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                eval_loss += loss.item()

                if i < 2:  # only for the first 2 batches
                    # Assume model output has shape [B, vocab_size]
                    predictions = outputs.argmax(dim=-1)  # [B]
                    for j in range(min(3, input_ids.size(0))):

                        pred_token = tokenizer.convert_ids_to_tokens([predictions[j].item()])[0]
                        target_token = tokenizer.convert_ids_to_tokens([target_ids[j].item()])[0]
                        context_tokens = tokenizer.convert_ids_to_tokens(input_ids[j].tolist())
                        logger.info(f"Context: {' '.join(context_tokens)}")
                        logger.info(
                            f"Predicted next token: {pred_token} | Actual next token: {target_token}"
                        )

        logger.info(f"Epoch {epoch+1}, Train_loss: {total_loss:.4f}, Eval Loss: {eval_loss:.4f}")

    total_time = time.time() - total_start
    logger.info(f"Total training time: {total_time:.2f} seconds")
