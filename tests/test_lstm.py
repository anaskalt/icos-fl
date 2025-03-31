"""Test module for the LSTM model.

This module validates the implementation of the LSTM model architecture and
its associated training and testing functions. It demonstrates key functionality
including model initialization, forward passes, weight management, and the
training/evaluation pipeline.
"""

import os
import sys
from typing import Tuple

import numpy as np
import torch

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from icos_fl.models.lstm import LSTMModel, get_weights, set_weights, test, train
from icos_fl.utils.colors import BCYA, BGRN, BMAG, BYEL, WHT, paint
from icos_fl.utils.logger import Logger

# We want to have logger as a module object accessible from all functions
log = None

# Global device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SyntheticDataset(torch.utils.data.Dataset):
    """A simple synthetic dataset for testing the LSTM model."""

    def __init__(self, size: int = 100, time_step: int = 10) -> None:
        """Initialize synthetic dataset with random sequences."""
        self.size = size
        self.time_step = time_step

        # Generate synthetic sequences and targets
        self.data = []
        for i in range(size):
            # Generate input with shape [1, time_step]
            input_seq = torch.randn(1, time_step)
            # Generate target with shape [1] to match model output
            target = torch.tensor([float(i % 10) / 10.0])
            self.data.append((input_seq, target))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        return self.data[index]

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.size


def create_dataloaders(
    batch_size: int = 16,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and test dataloaders with synthetic time series data."""
    train_dataset = SyntheticDataset(size=100)
    test_dataset = SyntheticDataset(size=50)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    train_msg = f"Created train dataloader with {paint(BGRN, str(len(train_dataset)))} samples"
    test_msg = f"Created test dataloader with {paint(BGRN, str(len(test_dataset)))} samples"
    log.info(train_msg)
    log.info(test_msg)

    return train_dataloader, test_dataloader


def initialize_models() -> None:
    """Initialize LSTM models with different configurations."""
    log.info("")
    log.info(paint(BCYA, "╔══════════════════════════════╗"))
    log.info(paint(BCYA, "║    Model Initialization      ║"))
    log.info(paint(BCYA, "╚══════════════════════════════╝"))
    log.info("")

    # Create a model with default output size
    model1 = LSTMModel(hidden_layer_size=10, time_step=10, num_layers=1)

    log.info(paint(BYEL, "▶ Model 1 Configuration:"))

    hidden_size_msg = f"  • Hidden layer size: {paint(WHT, str(model1.hidden_layer_size))}"
    time_step_msg = f"  • Time step: {paint(WHT, str(model1.time_step))}"
    num_layers_msg = f"  • Number of layers: {paint(WHT, str(model1.num_layers))}"

    log.info(hidden_size_msg)
    log.info(time_step_msg)
    log.info(num_layers_msg)
    log.info("")

    # Create a model with custom output size
    model2 = LSTMModel(
        hidden_layer_size=20,
        time_step=15,
        num_layers=2,
        output_size=2,
    )

    log.info(paint(BYEL, "▶ Model 2 Configuration:"))

    m2_hidden_size_msg = f"  • Hidden layer size: {paint(WHT, str(model2.hidden_layer_size))}"
    m2_time_step_msg = f"  • Time step: {paint(WHT, str(model2.time_step))}"
    m2_num_layers_msg = f"  • Number of layers: {paint(WHT, str(model2.num_layers))}"
    m2_output_size_msg = f"  • Output size: {paint(WHT, str(model2.linear.out_features))}"

    log.info(m2_hidden_size_msg)
    log.info(m2_time_step_msg)
    log.info(m2_num_layers_msg)
    log.info(m2_output_size_msg)
    log.info("")


def check_model_structure() -> None:
    """Examine the internal structure of the LSTM model components."""
    log.info(paint(BCYA, "╔══════════════════════════════╗"))
    log.info(paint(BCYA, "║       Model Structure        ║"))
    log.info(paint(BCYA, "╚══════════════════════════════╝"))
    log.info("")

    # Create a model
    hidden_size = 20
    time_step = 15
    num_layers = 2

    model = LSTMModel(
        hidden_layer_size=hidden_size,
        time_step=time_step,
        num_layers=num_layers,
    )

    # Show LSTM parameters
    log.info(paint(BYEL, "▶ LSTM Configuration:"))

    input_size_msg = f"  • Input size: {paint(WHT, str(model.lstm.input_size))}"
    hidden_size_msg = f"  • Hidden size: {paint(WHT, str(model.lstm.hidden_size))}"
    num_layers_msg = f"  • Number of layers: {paint(WHT, str(model.lstm.num_layers))}"

    log.info(input_size_msg)
    log.info(hidden_size_msg)
    log.info(num_layers_msg)
    log.info("")

    # Show linear layer parameters
    log.info(paint(BYEL, "▶ Linear Layer Configuration:"))

    in_features_msg = f"  • Input features: {paint(WHT, str(model.linear.in_features))}"
    out_features_msg = f"  • Output features: {paint(WHT, str(model.linear.out_features))}"

    log.info(in_features_msg)
    log.info(out_features_msg)
    log.info("")


def run_forward_pass() -> None:
    """Run a forward pass through the LSTM model with sample data."""
    log.info(paint(BCYA, "╔══════════════════════════════╗"))
    log.info(paint(BCYA, "║         Forward Pass         ║"))
    log.info(paint(BCYA, "╚══════════════════════════════╝"))
    log.info("")

    model = LSTMModel(hidden_layer_size=10, time_step=10, num_layers=1).to(DEVICE)

    # Create a batch of synthetic input data
    batch_size = 8
    input_data = torch.randn(batch_size, 1, 10).to(DEVICE)

    input_shape_msg = f"Input data shape: {paint(WHT, str(input_data.shape))}"
    log.info(input_shape_msg)

    # Perform forward pass
    with torch.no_grad():
        output = model(input_data)

    output_shape_msg = f"Output data shape: {paint(WHT, str(output.shape))}"
    log.info(output_shape_msg)
    log.info(paint(BYEL, "▶ Output values (first 3 samples):"))

    for i in range(min(3, batch_size)):
        sample_msg = f"  • Sample {i + 1}: {paint(BGRN, f'{output[i].item():.6f}')}"
        log.info(sample_msg)

    log.info("")


def transfer_model_weights() -> None:
    """Transfer weights between two LSTM model instances."""
    log.info(paint(BCYA, "╔══════════════════════════════╗"))
    log.info(paint(BCYA, "║       Weight Transfer        ║"))
    log.info(paint(BCYA, "╚══════════════════════════════╝"))
    log.info("")

    # Create two models with the same architecture
    model1 = LSTMModel(hidden_layer_size=10, time_step=10, num_layers=1)

    model2 = LSTMModel(hidden_layer_size=10, time_step=10, num_layers=1)

    # Get initial weights
    weights1 = get_weights(model1)
    weights2 = get_weights(model2)

    log.info(paint(BYEL, "▶ Initial weights comparison:"))

    for i, (w1, w2) in enumerate(zip(weights1, weights2, strict=False)):
        w1_sum = float(np.sum(w1))
        w2_sum = float(np.sum(w2))

        # Create formatted strings separately
        w1_fmt = paint(WHT, f"{w1_sum:.6f}")
        w2_fmt = paint(WHT, f"{w2_sum:.6f}")

        # Compose the final message
        weights_msg = f"  • Layer {i + 1} weights sum - Model 1: {w1_fmt}, Model 2: {w2_fmt}"
        log.info(weights_msg)

    # Set model2's weights to model1's weights
    log.info("")
    log.info(paint(BMAG, "► Transferring weights from Model 1 to Model 2..."))
    set_weights(model2, weights1)

    # Get updated weights
    weights2_updated = get_weights(model2)

    log.info("")
    log.info(paint(BYEL, "▶ Updated weights comparison:"))

    for i, (w1, w2) in enumerate(zip(weights1, weights2_updated, strict=False)):
        w1_sum = float(np.sum(w1))
        w2_sum = float(np.sum(w2))

        # Create formatted strings separately
        w1_fmt = paint(WHT, f"{w1_sum:.6f}")
        w2_fmt = paint(WHT, f"{w2_sum:.6f}")

        # Compose the final message
        updated_weights_msg = (
            f"  • Layer {i + 1} weights sum - Model 1: {w1_fmt}, Model 2: {w2_fmt}"
        )
        log.info(updated_weights_msg)

    log.info("")


def train_and_evaluate_model() -> None:
    """Train and evaluate the LSTM model on synthetic data."""
    log.info(paint(BCYA, "╔══════════════════════════════╗"))
    log.info(paint(BCYA, "║  Model Training & Evaluation ║"))
    log.info(paint(BCYA, "╚══════════════════════════════╝"))
    log.info("")

    # Create dataloaders once
    train_dataloader, test_dataloader = create_dataloaders()
    log.info("")

    # Create a model
    model = LSTMModel(hidden_layer_size=10, time_step=10, num_layers=1).to(DEVICE)

    # Train for a few epochs
    epochs = 2
    log.info(paint(BMAG, f"► Training model for {epochs} epochs..."))

    try:
        avg_loss = train(model=model, train_dataloader=train_dataloader, epochs=epochs, lr=0.01)

        train_result_msg = (
            f"Training completed with average loss: {paint(BGRN, f'{avg_loss:.6f}')}"
        )
        log.info(train_result_msg)
        log.info("")

        # Evaluate the model
        log.info(paint(BMAG, "► Evaluating model..."))

        test_loss = test(model=model, test_dataloader=test_dataloader, device=DEVICE)

        eval_result_msg = f"Evaluation completed with test loss: {paint(BGRN, f'{test_loss:.6f}')}"
        log.info(eval_result_msg)
        log.info("")
    except Exception as e:
        error_msg = f"Error during model training or evaluation: {e!s}"
        log.exception(error_msg)


def test_lstm() -> None:
    """Run all LSTM model tests."""
    global log

    # Initialize logger
    log = Logger(useconsole=True, usecolor=True)
    log.info(log.report())

    # Display header
    log.info("")
    log.info(paint(BCYA, "╔══════════════════════════════════════════════╗"))
    log.info(paint(BCYA, "║              LSTM MODEL TESTING              ║"))
    log.info(paint(BCYA, "╚══════════════════════════════════════════════╝"))
    log.info("")

    device_msg = f"Using device: {paint(BGRN, str(DEVICE))}"
    log.info(device_msg)

    # Run test functions
    try:
        initialize_models()
        check_model_structure()
        run_forward_pass()
        transfer_model_weights()
        train_and_evaluate_model()

        # Display footer
        log.info(paint(BCYA, "╔══════════════════════════════════════════════╗"))
        log.info(paint(BCYA, "║          LSTM MODEL TESTING COMPLETE         ║"))
        log.info(paint(BCYA, "╚══════════════════════════════════════════════╝"))
        log.info("")
    except Exception as e:
        error_msg = f"An error occurred during LSTM testing: {e!s}"
        log.exception(error_msg)


if __name__ == "__main__":
    test_lstm()
