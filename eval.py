"""This is a script for evaluating your trained model.

This is just a starting point for your validation pipeline.
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import mlflow
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets.folder import default_loader
from torchvision.transforms import v2

ROOT_DIR = Path(__file__).parent

# Set the tracking server to be localhost with sqlite as tracking store
mlflow.set_tracking_uri(uri="sqlite:///mlruns.db")


class SafetyBatchDataset(torchvision.datasets.ImageFolder):
    """Custom dataset for safety batch."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(SafetyBatchDataset, self).__init__(
            root=root,
            loader=loader,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # convert target index to class label
        target_label = int(
            [label for label, idx in self.class_to_idx.items() if idx == target][0]
        )

        return sample, target_label


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
    data_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(img_dimensions),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    dataset = SafetyBatchDataset(data_directory_path, transform=data_transforms)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
    )

    return data_loader


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
        default=str(Path(__file__).parent / "safetyBatches" / "Batch_1"),
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
    ground_truth = []
    for _, target in batch_loader:
        ground_truth.extend(target.tolist())
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
