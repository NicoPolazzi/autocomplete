import torch
from src.model import AutocompleteModel
from src.dataset import CodeDataset, get_dataloader


def test_model_training():
    # 1. Create small dataset
    dataset = CodeDataset()
    train_dataset, _ = dataset.create_train_evaluation_split([0.8, 0.2])
    train_loader = get_dataloader(train_dataset, batch_size=32)

    # 2. Initialize model
    input_size = 768  # CodeBERT embedding size
    hidden_size = 256
    output_size = len(set(dataset.next_tokens_ids))
    model = AutocompleteModel(input_size, hidden_size, output_size)

    # 3. Setup training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. Single batch test
    inputs, targets = next(iter(train_loader))
    outputs = model(inputs)

    # 5. Assert shapes
    assert outputs.shape[0] == inputs.shape[0]  # Batch size matches
    assert outputs.shape[1] == output_size  # Output size matches vocabulary

    # 6. Test loss computation
    loss = criterion(outputs, targets)
    assert not torch.isnan(loss)

    # 7. Test backward pass
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
