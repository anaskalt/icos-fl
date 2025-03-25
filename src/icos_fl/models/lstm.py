"""LSTM model for time series prediction in ICOS-FL.

This module defines the LSTM model architecture and training/testing functions.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class LSTMModel(nn.Module):
    """LSTM model for time series prediction.

    Args:
        hidden_layer_size: Size of the LSTM hidden layer
        time_step: Number of time steps (sequence length) for LSTM input
        num_layers: Number of LSTM layers
        output_size: Size of the output layer (default is 1 for single value prediction)
    """

    def __init__(
        self,
        hidden_layer_size: int,
        time_step: int,
        num_layers: int,
        output_size: int = 1,
    ) -> None:
        """Initialize the LSTM model."""
        super().__init__()

        self.hidden_layer_size = hidden_layer_size
        self.time_step = time_step
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(time_step, hidden_layer_size, num_layers, batch_first=True)

        # Linear layer to produce output prediction
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            input_seq: Input tensor

        Returns:
            Output tensor of predictions
        """
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions


def get_weights(model: nn.Module) -> List[np.ndarray]:
    """Extract model weights as a list of NumPy arrays.

    Args:
        model: PyTorch model

    Returns:
        List of NumPy arrays containing model weights
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    """Set model weights from a list of NumPy arrays.

    Args:
        model: PyTorch model
        weights: List of NumPy arrays containing weights
    """
    params_dict = zip(model.state_dict().keys(), weights, strict=False)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def train(
    model: LSTMModel,
    train_dataloader: DataLoader,
    epochs: int,
    lr: float = 0.001,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
) -> float:
    """Train the LSTM model.

    Args:
        model: The LSTM model to train
        train_dataloader: DataLoader containing training data
        epochs: Number of training epochs
        lr: Learning rate
        criterion: Loss function (defaults to MSE if None)
        optimizer: Optimizer (defaults to Adam if None)

    Returns:
        Average training loss
    """
    # Set model to training mode
    model.train()

    # Set default criterion and optimizer if not provided
    if criterion is None:
        criterion = nn.MSELoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    total_loss = 0.0
    for _ in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in train_dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Average loss for this epoch
        total_loss += epoch_loss / len(train_dataloader)

    # Return average loss over all epochs
    return total_loss / epochs


def test(model: LSTMModel, test_dataloader: DataLoader, device: torch.device) -> float:
    """Evaluate model on the test dataset.

    Args:
        model: The LSTM model to evaluate
        test_dataloader: DataLoader containing test data
        device: PyTorch device for computation

    Returns:
        Average test loss
    """
    # Set model to evaluation mode
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            # Move input and targets to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Accumulate loss
            total_loss += loss.item()

    # Calculate average loss
    avg_loss = total_loss / len(test_dataloader)

    return avg_loss
