"""Test module for the Processor and TimeSeriesDataset classes.

This module validates the implementation of the Processor class and its
associated TimeSeriesDataset class, testing data normalization, sequence
creation, and dataloader generation for time series prediction.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from icos_fl.utils.colors import BCYA, BGRN, BMAG, BYEL, RED, WHT, paint
from icos_fl.utils.processor import Processor, TimeSeriesDataset

# Global device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_synthetic_data(rows: int = 100, cols: int = 3) -> pd.DataFrame:
    """Create synthetic time series data for testing.

    Args:
        rows: Number of time steps in the dataset
        cols: Number of metrics/features in the dataset

    Returns:
        DataFrame with synthetic time series data
    """
    np.random.seed(42)  # For reproducibility

    # Create time index
    date_range = pd.date_range(start="2023-01-01", periods=rows, freq="h")

    # Create data with different patterns
    data = {}

    # CPU consumption: Sine wave with noise and daily pattern
    cpu = np.sin(np.linspace(0, 4 * np.pi, rows)) * 0.4 + 0.5
    cpu += np.random.normal(0, 0.05, rows)  # Add noise
    # Add daily pattern (higher during day, lower at night)
    daily_pattern = np.sin(np.linspace(0, 2 * np.pi, 24))
    for i in range(rows):
        hour = i % 24
        cpu[i] += daily_pattern[hour] * 0.2
    data["cpu_consumption"] = np.clip(cpu, 0.1, 0.9)  # Keep values in reasonable range

    # Memory consumption: Gradual increase with resets
    memory = np.linspace(0.3, 0.8, rows)
    # Add periodic resets
    for i in range(0, rows, 20):
        if i + 3 < rows:
            memory[i : i + 3] = 0.3
    memory += np.random.normal(0, 0.03, rows)  # Add noise
    data["memory_consumption"] = np.clip(memory, 0.2, 0.9)

    # Energy consumption: Correlated with CPU but with lag
    energy = np.roll(cpu, 2) * 0.7 + np.random.normal(0, 0.02, rows) + 0.2
    data["energy_consumption"] = np.clip(energy, 0.1, 0.8)

    # Create DataFrame
    df = pd.DataFrame(data, index=date_range)

    rows_msg = (
        f"Created synthetic DataFrame with {paint(BGRN, str(rows))} rows "
        f"and {paint(BGRN, str(cols))} columns"
    )
    cols_msg = f"Columns: {paint(WHT, ', '.join(df.columns))}"
    range_msg = f"Data range: {paint(WHT, str(df.index[0]))} to {paint(WHT, str(df.index[-1]))}"

    print(rows_msg)
    print(cols_msg)
    print(range_msg)

    return df


def initialize_processor() -> None:
    """Initialize a Processor instance and validate its configuration."""
    print("")
    print(paint(BCYA, "╔══════════════════════════════╗"))
    print(paint(BCYA, "║    Processor Initialization  ║"))
    print(paint(BCYA, "╚══════════════════════════════╝"))
    print("")

    # Initialize with default parameters
    processor1 = Processor(time_step=10, metric="cpu_consumption")

    print(paint(BYEL, "▶ Processor 1 Configuration:"))

    p1_step = f"  • Time step: {paint(WHT, str(processor1.time_step))}"
    p1_metric = f"  • Metric: {paint(WHT, processor1.metric)}"
    p1_batch = f"  • Batch size: {paint(WHT, str(processor1.batch_size))}"
    p1_ratio = f"  • Train ratio: {paint(WHT, str(processor1.train_ratio))}"
    p1_device = f"  • Device: {paint(WHT, str(processor1.device))}"

    print(p1_step)
    print(p1_metric)
    print(p1_batch)
    print(p1_ratio)
    print(p1_device)

    # Initialize with custom parameters
    processor2 = Processor(
        time_step=15, metric="memory_consumption", batch_size=32, train_ratio=0.75, device=DEVICE
    )

    print("")
    print(paint(BYEL, "▶ Processor 2 Configuration:"))

    p2_step = f"  • Time step: {paint(WHT, str(processor2.time_step))}"
    p2_metric = f"  • Metric: {paint(WHT, processor2.metric)}"
    p2_batch = f"  • Batch size: {paint(WHT, str(processor2.batch_size))}"
    p2_ratio = f"  • Train ratio: {paint(WHT, str(processor2.train_ratio))}"
    p2_device = f"  • Device: {paint(WHT, str(processor2.device))}"

    print(p2_step)
    print(p2_metric)
    print(p2_batch)
    print(p2_ratio)
    print(p2_device)
    print("")


def test_data_normalization() -> None:
    """Test the data normalization functionality."""
    print(paint(BCYA, "╔══════════════════════════════╗"))
    print(paint(BCYA, "║      Data Normalization      ║"))
    print(paint(BCYA, "╚══════════════════════════════╝"))
    print("")

    # Create test data
    df = create_synthetic_data(rows=100)

    # Create processor
    processor = Processor(time_step=10, metric="cpu_consumption")

    # Normalize data
    normalized_df = processor._normalize_data(df)

    # Check statistics of original data
    print(paint(BYEL, "▶ Original Data Statistics:"))

    for col in df.columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        stat_msg = f"  • {paint(WHT, col)}: mean={paint(BGRN, f'{mean_val:.4f}')}, "
        stat_msg += f"std={paint(BGRN, f'{std_val:.4f}')}"
        print(stat_msg)

    print("")

    # Check statistics of normalized data
    print(paint(BYEL, "▶ Normalized Data Statistics:"))

    for col in normalized_df.columns:
        mean_val = normalized_df[col].mean()
        std_val = normalized_df[col].std()
        norm_msg = f"  • {paint(WHT, col)}: mean={paint(BGRN, f'{mean_val:.4f}')}, "
        norm_msg += f"std={paint(BGRN, f'{std_val:.4f}')}"
        print(norm_msg)

    print("")
    print(paint(BMAG, "► Normalization should result in mean ≈ 0 and std ≈ 1 for each column"))
    print("")


def test_train_test_split() -> None:
    """Test the train/test splitting functionality."""
    print(paint(BCYA, "╔══════════════════════════════╗"))
    print(paint(BCYA, "║      Train/Test Splitting    ║"))
    print(paint(BCYA, "╚══════════════════════════════╝"))
    print("")

    # Create test data
    df = create_synthetic_data(rows=100)

    # Create processor
    processor = Processor(time_step=10, metric="cpu_consumption")

    # Test different split ratios
    ratios = [0.5, 0.7, 0.8, 0.9]

    for ratio in ratios:
        train_size, test_size = processor._train_test_split(df, ratio)

        ratio_title = f"▶ Split Ratio: {paint(BGRN, str(ratio))}"
        print(paint(BYEL, ratio_title))

        dataset_size = f"  • Dataset size: {paint(WHT, str(len(df)))}"
        print(dataset_size)

        train_pct = train_size / len(df) * 100
        train_msg = f"  • Training set size: {paint(BGRN, str(train_size))} "
        train_msg += f"({paint(WHT, f'{train_pct:.1f}')}%)"
        print(train_msg)

        test_pct = test_size / len(df) * 100
        test_msg = f"  • Test set size: {paint(BGRN, str(test_size))} "
        test_msg += f"({paint(WHT, f'{test_pct:.1f}')}%)"
        print(test_msg)

        print("")


def test_time_series_dataset() -> None:
    """Test the TimeSeriesDataset class."""
    print(paint(BCYA, "╔══════════════════════════════╗"))
    print(paint(BCYA, "║     TimeSeriesDataset        ║"))
    print(paint(BCYA, "╚══════════════════════════════╝"))
    print("")

    # Create test data
    df = create_synthetic_data(rows=100)

    # Normalize data for TimeSeriesDataset
    processor = Processor(time_step=10, metric="cpu_consumption")
    normalized_df = processor._normalize_data(df)

    # Create dataset
    dataset = TimeSeriesDataset(
        df=normalized_df,
        start_index=0,
        population=80,
        time_step=10,
        metric="cpu_consumption",
        device=DEVICE,
    )

    dataset_msg = f"TimeSeriesDataset created with {paint(BGRN, str(len(dataset)))} sequences"
    time_step_msg = f"  • Time step: {paint(WHT, str(dataset.time_step))}"
    metric_msg = f"  • Metric: {paint(WHT, dataset.metric)}"

    print(dataset_msg)
    print(time_step_msg)
    print(metric_msg)

    # Get a sample sequence
    sequence, target = dataset[0]

    print("")
    print(paint(BYEL, "▶ Sample Sequence and Target:"))

    seq_shape_msg = f"  • Sequence shape: {paint(WHT, str(sequence.shape))}"
    target_shape_msg = f"  • Target shape: {paint(WHT, str(target.shape))}"

    print(seq_shape_msg)
    print(target_shape_msg)

    # Check sequence values (first 5)
    print("")
    print(paint(BYEL, "▶ First 5 Values of the Sequence:"))

    for i in range(min(5, dataset.time_step)):
        value = sequence[0][i].item()
        value_msg = f"  • Value {i + 1}: {paint(BGRN, f'{value:.4f}')}"
        print(value_msg)

    target_value = target.item()
    target_msg = f"  • Target value: {paint(BGRN, f'{target_value:.4f}')}"
    print(target_msg)
    print("")


def test_data_loader_creation() -> None:
    """Test the complete data loader creation pipeline."""
    print(paint(BCYA, "╔══════════════════════════════╗"))
    print(paint(BCYA, "║     DataLoader Creation      ║"))
    print(paint(BCYA, "╚══════════════════════════════╝"))
    print("")

    # Create test data
    df = create_synthetic_data(rows=100)

    # Create processor
    processor = Processor(
        time_step=10, metric="cpu_consumption", batch_size=16, train_ratio=0.8, device=DEVICE
    )

    # Create dataloaders
    train_dataloader, val_dataloader, train_dataset, val_dataset = processor.create_data_loaders(
        df
    )

    print(paint(BMAG, "► DataLoaders created successfully"))

    train_size_msg = f"  • Training set size: {paint(BGRN, str(len(train_dataset)))} sequences"
    val_size_msg = f"  • Validation set size: {paint(BGRN, str(len(val_dataset)))} sequences"
    train_batches_msg = f"  • Training batches: {paint(WHT, str(len(train_dataloader)))}"
    val_batches_msg = f"  • Validation batches: {paint(WHT, str(len(val_dataloader)))}"

    print(train_size_msg)
    print(val_size_msg)
    print(train_batches_msg)
    print(val_batches_msg)

    # Get a batch from the training loader
    for batch_inputs, batch_targets in train_dataloader:
        print("")
        print(paint(BYEL, "▶ Sample Training Batch:"))

        inputs_msg = f"  • Batch inputs shape: {paint(WHT, str(batch_inputs.shape))}"
        targets_msg = f"  • Batch targets shape: {paint(WHT, str(batch_targets.shape))}"

        print(inputs_msg)
        print(targets_msg)
        break

    # Test with different parameters
    print("")
    print(paint(BMAG, "► Testing DataLoader creation with overridden parameters"))

    train_dataloader, val_dataloader, train_dataset, val_dataset = processor.create_data_loaders(
        df, time_step=15, metric="memory_consumption", batch_size=8, train_ratio=0.7
    )

    print(paint(BYEL, "▶ DataLoaders created with custom parameters:"))

    time_step_msg = f"  • Time step: {paint(WHT, '15')}"
    metric_msg = f"  • Metric: {paint(WHT, 'memory_consumption')}"
    batch_size_msg = f"  • Batch size: {paint(WHT, '8')}"
    ratio_msg = f"  • Train ratio: {paint(WHT, '0.7')}"
    train_size_msg = f"  • Training set size: {paint(BGRN, str(len(train_dataset)))} sequences"
    val_size_msg = f"  • Validation set size: {paint(BGRN, str(len(val_dataset)))} sequences"

    print(time_step_msg)
    print(metric_msg)
    print(batch_size_msg)
    print(ratio_msg)
    print(train_size_msg)
    print(val_size_msg)
    print("")


def test_processor() -> None:
    """Run all Processor and TimeSeriesDataset tests."""
    # Display header
    print("")
    print(paint(BCYA, "╔══════════════════════════════════════════════╗"))
    print(paint(BCYA, "║         PROCESSOR MODULE TESTING             ║"))
    print(paint(BCYA, "╚══════════════════════════════════════════════╝"))
    print("")

    device_msg = f"Using device: {paint(BGRN, str(DEVICE))}"
    print(device_msg)

    # Run test functions
    try:
        initialize_processor()
        test_data_normalization()
        test_train_test_split()
        test_time_series_dataset()
        test_data_loader_creation()

        # Display footer
        print(paint(BCYA, "╔══════════════════════════════════════════════╗"))
        print(paint(BCYA, "║       PROCESSOR MODULE TESTING COMPLETE      ║"))
        print(paint(BCYA, "╚══════════════════════════════════════════════╝"))
        print("")
    except Exception as e:  # noqa: BLE001
        error_msg = f"An error occurred during Processor testing: {e!s}"
        print(paint(RED, error_msg))


if __name__ == "__main__":
    test_processor()
