"""This is a script for training a neural network on traffic sign classification.

The original project was part of the deep learning safety lecture at RWU
by Mark Schutera (https://github.com/schutera/DeepSafety).

The German Traffic Sign Recognition Benchmark (GTSRB) dataset is used to train 
and validate the model (https://benchmark.ini.rub.de/).

MLFlow (https://mlflow.org/docs/latest/index.html) is used for experiment
tracking. Feel free to use a different tool to track your experiments if you want.
"""
import argparse
from typing import Tuple

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm

# Set the MLFlow tracking server to be localhost with sqlite as tracking store
mlflow.set_tracking_uri(uri="sqlite:///mlruns.db")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Some decent neural network for traffic sign classification.

    Feel free to adapt or change the architecture.
    """

    def __init__(self, num_classes: int):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(100)

        self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(150)

        self.conv3 = nn.Conv2d(150, 250, kernel_size=1)
        self.conv3_bn = nn.BatchNorm2d(250)

        self.fc1 = nn.Linear(250 * 3 * 3, 350)
        self.fc1_bn = nn.BatchNorm1d(350)

        self.fc2 = nn.Linear(350, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1_bn(F.max_pool2d(F.leaky_relu(self.conv1(x)), 2))
        x = self.dropout(x)

        x = self.conv2_bn(F.max_pool2d(F.leaky_relu(self.conv2(x)), 2))
        x = self.dropout(x)

        x = self.conv3_bn(F.max_pool2d(F.leaky_relu(self.conv3(x)), 2))
        x = self.dropout(x)

        x = x.view(-1, 250 * 3 * 3)
        x = F.relu(self.fc1(x))

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

    The German Traffic Sign Recognition Benchmark (GTSRB) dataset is loaded
    from torchvision and splitted into a training and validation set.

    You may want to extend this function.
    """
    data_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(img_dimensions),
            v2.ToDtype(torch.float32, scale=True),
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
    """Trains a model for one epoch."""
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
        # Pass inputs and labels to device (CPU/GPU)
        data, target = data.to(DEVICE), target.to(DEVICE)
        # Zero gradients for each batch
        optimizer.zero_grad()
        # Output predictions for each batch
        output = model(data)
        # Compute loss and gradients
        loss = loss_function(output, target)
        loss.backward()
        # Adjust weights
        optimizer.step()

        # Get data and report them
        # The class with the highest value is what we chose as prediction
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == target).sum().item()
        training_loss += loss.item()
        progress_bar.set_postfix(
            loss=loss.item(),
            accuracy=f"{100.0 * correct / len(train_loader.dataset):.1f} %",
        )

    training_loss /= len(train_loader.dataset)
    training_accuracy = correct / len(train_loader.dataset)

    # Log the loss and accuracy for each training epoch
    mlflow.log_metric("training loss", training_loss, step=epoch)
    mlflow.log_metric("training accuracy", training_accuracy, step=epoch)


def validate(
    model: nn.Module,
    loss_function: nn.modules.loss,
    lr_scheduler: optim.lr_scheduler,
    val_loader: torch.utils.data.DataLoader,
    epoch: int,
) -> None:
    """Evaluates the model on the validation dataset."""
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        # We don't need to calculate the gradients for our output since
        # we are not training here, so we can reduce memory consumption
        with torch.no_grad():
            output = model(data)
            validation_loss += loss_function(output, target).item()
            # The class with the highest value is what we chose as prediction
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

    validation_loss /= len(val_loader.dataset)
    # Adjust learning rate based on validation loss
    lr_scheduler.step(round(validation_loss, 2))
    # Gather data and report them
    val_accuracy = correct / len(val_loader.dataset)
    print(
        f"Validation set: Average loss: {validation_loss:.4f}, Accuracy: {100.0 * val_accuracy:.1f} %"
    )
    # Log the loss and accuracy for each validation epoch
    mlflow.log_metric("validation loss", validation_loss, step=epoch)
    mlflow.log_metric("validation accuracy", val_accuracy, step=epoch)


if __name__ == "__main__":
    # You may want to use different parameters than the default ones
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
        default=50,
        help="Number of epochs to train (default: 50).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001).",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1).")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # GTSRB as 43 classes, this parameter is just used for the number of outputs neurons
    # of our neural network
    NUM_CLASSES = 43

    # You may want to experiment with different models, loss function or other parameters
    model = Net(num_classes=NUM_CLASSES)
    model.to(DEVICE)

    # The CrossEntropyLoss is equivalent to applying LogSoftmax and NLLLoss, thus
    # our neural network doesn't contain a softmax layer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    # Scheduling to dynamically reduce learning rate based on validation measurements
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5, verbose=True
    )

    train_loader, val_loader = load_and_transform_data(batch_size=args.batch_size)

    # define an experiment that will group each (training) run together
    mlflow.set_experiment("Deep Safety")

    # Initiate a run context to record our model, the hyperparameter, as well as
    # metrics and so on
    with mlflow.start_run():
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
            validate(model, criterion, scheduler, val_loader, epoch)

        # Infer the model signature for logging
        model.to("cpu")
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
