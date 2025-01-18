import time
import torch
from torch.utils.data import DataLoader
from src.logger import get_logger
from src.model import CodeRNN
from src.preprocessing.dataset import CodeDataset, create_train_test_split
from src.train import train_model, get_optimizer, get_criterion

logger = get_logger(__name__)


def collate_fn(batch):
    input_sequences, labels = zip(*batch)
    input_sequences_padded = torch.nn.utils.rnn.pad_sequence(
        [seq.squeeze(0) for seq in input_sequences], batch_first=True
    )
    labels = torch.tensor(labels)
    return input_sequences_padded, labels


def main():
    start_time = time.time()
    logger.info("Starting the ML pipiline")
    alldata = CodeDataset()

    train_set, test_set = create_train_test_split(alldata)
    train_loader = DataLoader(
        train_set, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    logger.info(
        f"Data loaded successfully after {time.time() - start_time:.2f} seconds"
    )

    input_size = 768  # Embedding size from CodeBERT
    hidden_size = 256
    output_size = 10  # Assuming the same size for output tokens
    model = CodeRNN(input_size, hidden_size, output_size, model_type="LSTM")

    criterion = get_criterion()
    optimizer = get_optimizer(model)

    num_epochs = 25
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

    logger.info("End of the ML pipeline")


if __name__ == "__main__":
    main()
