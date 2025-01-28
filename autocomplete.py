import argparse

from torch import nn
import torch

from src.logger import get_logger
from src.dataset import CodeDataset
from src.train import train_model
from src.utils import load_processed_dataset

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Python autocompletion")
    parser.add_argument("command", choices=["preprocess", "train"])
    args = parser.parse_args()

    logger.info("Start of the ML pipeline")

    if args.command == "preprocess":
        alldata = CodeDataset()
        alldata.write_to_file("data/dataset.pkl")

    elif args.command == "train":
        dataset = load_processed_dataset("data/dataset.pkl")
        train_dataset, test_dataset = dataset.create_train_evaluation_split()

        input_size = 768
        hidden_size = 256
        output_size = len(set(dataset.next_tokens_ids))
        num_layer = 2

    logger.info("End of the ML pipeline")


if __name__ == "__main__":
    main()
