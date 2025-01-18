import torch


def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for input_sequence, label in dataloader:
            output = model(input_sequence)
            loss = criterion(output, label)
            total_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return average_loss, accuracy
