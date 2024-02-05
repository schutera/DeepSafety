"""This is a script for evaluating your trained model.

This is just a starting point for your validation pipeline.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import mlflow
import torch
import torch.nn as nn
import torchvision

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


def get_sign_names() -> Dict[int, str]:
    """Gets the corresponding sign names for the classes."""
    sign_names_file_path = Path(__file__).parent / "signnames.csv"
    sign_names = {}
    with open(sign_names_file_path, mode="r") as sign_names_file:
        sign_names_reader = csv.reader(sign_names_file)
        next(sign_names_reader, None)  # skip the header
        for line in sign_names_reader:
            class_id = int(line[0])
            sign_name = line[1]
            sign_names[class_id] = sign_name

    return sign_names


def load_and_transform_data(
    data_directory_path: str,
    batch_size: int = 64,
    img_dimensions: tuple[int, int] = (32, 32),
) -> torch.utils.data.DataLoader:
    """Loads data from directory, resizes and rescales images to floats
    between 0 and 1.

    You may want to extend this.
    """
    data_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(img_dimensions),
            torchvision.transforms.ToTensor(),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(
        data_directory_path, transform=data_transforms
    )

    batch_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
    )

    return batch_loader


def evaluate(
    model: nn.Module,
    loss_function: nn.modules.loss,
    batch_loader: torch.utils.data.DataLoader,
) -> List[int]:
    """Evaluates the model on the validation batch.

    You may want to extend this to report more metrics. However, it is not about
    how many metrics you crank out, it is about whether you find the meaningful
    ones and report. Think thoroughly about which metrics to go for.
    """
    model.eval()
    batch_loss = 0
    correct = 0
    predictions = []
    for data, target in batch_loader:
        with torch.no_grad():
            output = model(data)
            batch_loss += loss_function(output, target).item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            predictions.extend(predicted.tolist())

    batch_loss /= len(batch_loader.dataset)
    batch_accuracy = correct / len(batch_loader.dataset)
    print(
        f"Safety batch: Average loss: {batch_loss:.4f}, Accuracy: {100.0 * batch_accuracy:.1f} %"
    )

    return predictions


if __name__ == "__main__":
    # you may want to use different parameters than the default ones
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent / "safetyBatches" / "Batch_0"),
        help="Directory path where evaluation batch is located.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="MLFlow Run ID which contains a logged model to evaluate.",
    )

    args = parser.parse_args()

    # Load the logged model and evaluate it
    model_uri = f"runs:/{args.run_id}/model"
    loaded_model = mlflow.pytorch.load_model(model_uri)

    criterion = nn.CrossEntropyLoss()

    batch_loader = load_and_transform_data(data_directory_path=args.data_dir)

    predictions = evaluate(loaded_model, criterion, batch_loader)

    # Output incorrect classifications
    ground_truth = batch_loader.dataset.targets
    sign_names = get_sign_names()
    wrong_predictions_idx = [
        idx
        for idx, (y_pred, y) in enumerate(zip(predictions, ground_truth))
        if y_pred != y
    ]
    for idx in wrong_predictions_idx:
        print(
            f"Traffic sign {sign_names[ground_truth[idx]]} incorrectly "
            f"classified as {sign_names[predictions[idx]]}"
        )
