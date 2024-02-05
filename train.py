"""This is a script for training a neural network on traffic sign classification.

The original project was part of the deep learning safety lecture at RWU
by Mark Schutera (https://github.com/schutera/DeepSafety).

The data to train and validate the model can be downloaded here:
https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/download

Make sure you run a local MLFlow tracking server before running the script.
From a terminal, run:
mlflow server --host 127.0.0.1 --port 8080

Feel free to use a different tool to track you experiments if you want.
"""
import argparse
from pathlib import Path
from typing import Tuple

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# TODO
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, 5)
        self.conv1_bn = nn.BatchNorm2d(100)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(100, 150, 3)
        self.conv2_bn = nn.BatchNorm2d(150)

        self.conv3 = nn.Conv2d(150, 250, 1)
        self.conv3_bn = nn.BatchNorm2d(250)

        self.fc1 = nn.Linear(250 * 3 * 3, 350)
        self.fc1_bn = nn.BatchNorm1d(350)

        self.fc2 = nn.Linear(350, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.dropout(self.conv1_bn(x))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.dropout(self.conv2_bn(x))
        x = self.pool(F.elu(self.conv3(x)))
        x = self.dropout(self.conv3_bn(x))
        x = x.view(-1, 250 * 3 * 3)
        x = F.elu(self.fc1(x))
        x = self.dropout(self.fc1_bn(x))
        x = self.fc2(x)

        return x


def load_and_transform_data(
    batch_size: int,
    img_dimensions: Tuple[int, int] = (32, 32),
    train_data_split=0.8,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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

    dataset = torchvision.datasets.GTSRB(
        root="gtsrb",
        split="train",
        transform=data_transforms,
        download=True,
    )

    train_length = int(train_data_split * len(dataset))
    val_length = len(dataset) - train_length

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, (train_length, val_length)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def train(
    model: nn.Module,
    loss_function: nn.modules.loss,
    optimizer: torch.optim,
    train_loader: torch.utils.data.DataLoader,
    epoch: int,
) -> None:
    """Train model for one epoch."""
    model.train()
    correct = 0
    training_loss = 0
    for _, (data, target) in enumerate(
        progress_bar := tqdm(
            train_loader, unit=" batch", total=len(train_loader), desc="Training"
        ),
        0,
    ):
        progress_bar.set_description(f"Epoch {epoch}")
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == target).sum().item()
        training_loss += loss.item()
        progress_bar.set_postfix(
            loss=loss.item(),
            accuracy=f"{100.0 * correct / len(train_loader.dataset):.1f} %",
        )

    training_loss /= len(train_loader.dataset)
    training_accuracy = correct / len(train_loader.dataset)

    mlflow.log_metric("training loss", training_loss, step=epoch)
    mlflow.log_metric("training accuracy", training_accuracy, step=epoch)


def validation(
    model: nn.Module,
    loss_function: nn.modules.loss,
    lr_scheduler: optim.lr_scheduler,
    val_loader: torch.utils.data.DataLoader,
    epoch: int,
) -> None:
    """Evaluates the model on the validation dataset."""
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        with torch.no_grad():
            output = model(data)
            validation_loss += loss_function(output, target).item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

    validation_loss /= len(val_loader.dataset)
    lr_scheduler.step(round(validation_loss, 2))
    val_accuracy = correct / len(val_loader.dataset)
    print(
        f"Validation set: Average loss: {validation_loss:.4f}, Accuracy: {100.0 * val_accuracy:.1f} %"
    )
    mlflow.log_metric("validation loss", validation_loss, step=epoch)
    mlflow.log_metric("validation accuracy", val_accuracy, step=epoch)


if __name__ == "__main__":
    # you may want to use different parameters than the default ones
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Input batch size (default: 64).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train (default: 100).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001).",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1).")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path(__file__).parent / "saved_models"),
        help="Directory path where the trained model should be saved.",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    NUM_CLASSES = 43  # GTSRB as 43 classes

    # you may want to experiment with different models, loss function or other parameters
    model = Net(num_classes=NUM_CLASSES)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5, verbose=True
    )

    train_loader, val_loader = load_and_transform_data(batch_size=args.batch_size)

    mlflow.set_experiment("Deep Safety")

    with mlflow.start_run() as run:
        # Log the hyperparameters, add more if needed
        mlflow.log_params(
            {
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "random_seed": args.seed,
                "loss_function": criterion,
                "optimizer": optimizer,
            }
        )

        for epoch in range(1, args.epochs + 1):
            train(model, criterion, optimizer, train_loader, epoch)
            validation(model, criterion, scheduler, val_loader, epoch)

        # Infer the model signature
        X_train = next(iter(train_loader))[0]
        signature = mlflow.models.infer_signature(
            X_train.numpy(), model(X_train).detach().numpy()
        )

        # Save your model for later use. Early enough you should think about a model versioning
        # system and which information you will need to link with the model when doing so.
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.numpy(),
            registered_model_name="gtsrb",
        )