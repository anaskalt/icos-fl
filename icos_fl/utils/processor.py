"""Data processing utilities for ICOS-FL time series prediction.

This module provides the Processor class for preparing time series data
for LSTM model training and evaluation, along with a TimeSeriesDataset
class for handling sequence creation through sliding windows.
"""

from typing import Optional, Tuple

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    """Dataset for time series prediction with sliding window approach.

    Creates sequences of consecutive time steps as inputs and
    uses the next value as the prediction target.

    Args:
        df: Input DataFrame containing the time series data
        start_index: Starting index in the DataFrame to create sequences
        population: Total number of samples to include from start_index
        time_step: Number of time steps (sequence length) for LSTM input
        metric: Column name in the DataFrame to use as target
        device: PyTorch device to place tensors on
    """

    def __init__(
        self,
        df: pd.DataFrame,
        start_index: int,
        population: int,
        time_step: int,
        metric: str,
        device: torch.device,
    ) -> None:
        """Initialize the TimeSeriesDataset."""
        # Calculate overlap needed for first sequences
        previous_overlap = max(start_index - time_step, 0)

        # Extract the target metric column
        metric_series = df[metric]

        # Select the required slice of data including overlap
        self.data = metric_series.iloc[previous_overlap : start_index + population]
        self.metric = metric
        self.time_step = time_step
        self.device = device
        self.length = len(self.data) - self.time_step - 1

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence and its target.

        Args:
            index: Index of the sequence to retrieve

        Returns:
            Tuple of (input_sequence, target_value)
        """
        # Extract sequence of time_step values as input
        sequence_values = self.data.iloc[index : index + self.time_step].values

        # Convert to tensor and reshape to [1, sequence_length]
        input_tensor = torch.tensor(sequence_values).unsqueeze(0)
        input_tensor = input_tensor.float().to(self.device)

        # Extract the next value as target
        target_value = self.data.iloc[index + self.time_step]

        # Convert to tensor and reshape to [1]
        target_tensor = torch.tensor(target_value).float().to(self.device)
        target_tensor = target_tensor.unsqueeze(0)

        return input_tensor, target_tensor

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return self.length


class Processor:
    """Processor for time series data preparation in ICOS-FL.

    This class handles data preprocessing for time series forecasting,
    providing methods for data normalization, sequence creation,
    and DataLoader generation.

    Args:
        time_step: Number of time steps (sequence length) for LSTM input
        metric: Default column name in the DataFrame to use as target
        batch_size: Default batch size for DataLoaders
        train_ratio: Default ratio for train/test split
        device: PyTorch device to place tensors on
    """

    def __init__(
        self,
        time_step: int,
        metric: str,
        batch_size: int = 64,
        train_ratio: float = 0.8,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the Processor with required parameters.

        Args:
            time_step: Number of time steps (sequence length) for LSTM input
            metric: Default column name in the DataFrame to use as target
            batch_size: Default batch size for DataLoaders
            train_ratio: Default ratio for train/test split
            device: PyTorch device to place tensors on
        """
        self.time_step = time_step
        self.metric = metric
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.device = device if device is not None else torch.device("cpu")

    def create_data_loaders(
        self,
        df: pd.DataFrame,
        time_step: Optional[int] = None,
        metric: Optional[str] = None,
        batch_size: Optional[int] = None,
        train_ratio: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[DataLoader, DataLoader, TimeSeriesDataset, TimeSeriesDataset]:
        """Create DataLoaders for training and validation.

        This method handles the complete data preparation pipeline:
        1. Normalizes the data
        2. Splits into training and validation sets
        3. Creates appropriate datasets with sliding window sequences
        4. Wraps datasets in DataLoaders

        Args:
            df: DataFrame containing the time series data
            time_step: Sequence length (uses instance default if None)
            metric: Column name to use as target (uses instance default if None)
            batch_size: Batch size for DataLoaders (uses instance default if None)
            train_ratio: Ratio of data for training (uses instance default if None)
            device: PyTorch device (uses instance default if None)

        Returns:
            Tuple of (train_dataloader, val_dataloader, train_dataset, val_dataset)
        """
        # Use instance defaults if parameters not provided
        time_step = time_step if time_step is not None else self.time_step
        metric = metric if metric is not None else self.metric
        batch_size = batch_size if batch_size is not None else self.batch_size
        train_ratio = train_ratio if train_ratio is not None else self.train_ratio
        device = device if device is not None else self.device

        # Normalize data
        normalized_df = self._normalize_data(df)

        # Split data into train and validation sets
        train_size, val_size = self._train_test_split(normalized_df, train_ratio)

        # Create datasets
        train_dataset = TimeSeriesDataset(
            df=normalized_df,
            start_index=0,
            population=train_size,
            time_step=time_step,
            metric=metric,
            device=device,
        )

        val_dataset = TimeSeriesDataset(
            df=normalized_df,
            start_index=train_size,
            population=val_size,
            time_step=time_step,
            metric=metric,
            device=device,
        )

        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, val_dataloader, train_dataset, val_dataset

    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the dataset using standardization (zero mean, unit variance).

        Args:
            df: Input DataFrame containing the time series data

        Returns:
            Normalized DataFrame with the same structure
        """
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(df)
        return pd.DataFrame(normalized_data, columns=df.columns, index=df.index)

    def _train_test_split(self, df: pd.DataFrame, train_ratio: float) -> Tuple[int, int]:
        """Split the dataset into training and testing sets.

        Args:
            df: Input DataFrame containing the time series data
            train_ratio: Ratio for splitting data into train and test sets (0-1)

        Returns:
            Tuple containing the number of training and testing samples
        """
        train_count = int(len(df) * train_ratio)
        test_count = len(df) - train_count
        return train_count, test_count
