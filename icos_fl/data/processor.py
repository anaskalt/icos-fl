"""Data preprocessing and dataset creation module.

This module handles the preprocessing of time series data and creation of 
training/testing datasets for the LSTM model.

Key Components:
    - DataProcessor: Main class for data preprocessing
    - TimeSeriesDataset: Custom Dataset for time series data
"""

import pandas as pd
import numpy as np
import torch
from typing import Tuple
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from icos_fl.utils.logger import ICOSLogger as LOGGER

LOG = LOGGER("DataProcessor")

class TimeSeriesDataset(Dataset):
    """Custom time series dataset for LSTM model.
    
    This dataset creates sequences of fixed length from time series data,
    where each sequence is used to predict the next value.
    
    Attributes:
        sequences: Tensor of input sequences
        targets: Tensor of target values
    """

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """Initialize dataset.
        
        Args:
            sequences: Array of input sequences
            targets: Array of target values
        """
        self.sequences = torch.FloatTensor(sequences).unsqueeze(-1)
        self.targets = torch.FloatTensor(targets).unsqueeze(-1)

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item.
        
        Args:
            idx: Index of item
            
        Returns:
            Tuple of (sequence, target)
        """
        return self.sequences[idx], self.targets[idx]

class DataProcessor:
    """Data preprocessing and dataset creation.
    
    This class handles:
        - Data cleaning (handling missing values)
        - Normalization
        - Sequence creation
        - Dataset splitting
        
    Attributes:
        fill_nan: Maximum consecutive NaN values to fill
        time_step: Length of input sequences
        train_test_split: Ratio for train/test split
        scaler: StandardScaler instance for normalization
    """
    
    def __init__(self, fill_nan: int = 10,
                 time_step: int = 10,
                 train_test_split: float = 0.8):
        """Initialize processor.
        
        Args:
            fill_nan: Maximum consecutive NaN values to fill
            time_step: Length of input sequences
            train_test_split: Ratio for train/test split
        """
        self.fill_nan = fill_nan
        self.time_step = time_step
        self.train_test_split = train_test_split
        self.scaler = StandardScaler()

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize input data.
        
        Args:
            df: Input DataFrame with time series data
            
        Returns:
            Preprocessed DataFrame
            
        Raises:
            ValueError: If DataFrame is empty or all NaN
        """
        try:
            if df.empty:
                raise ValueError("Empty DataFrame provided")
                
            # Handle missing values
            df = df.ffill(limit=self.fill_nan)
            df = df.bfill(limit=self.fill_nan)
            
            if df.isna().any().any():
                raise ValueError("DataFrame contains NaN values after filling")
            
            # Normalize data
            normalized = self.scaler.fit_transform(df)
            return pd.DataFrame(normalized, columns=df.columns, index=df.index)
            
        except Exception as e:
            LOG.error(f"Preprocessing failed: {str(e)}")
            raise

    def create_dataset(self, df: pd.DataFrame, metric: str) -> Tuple[Dataset, Dataset]:
        """Create train/test datasets from dataframe.
        
        Args:
            df: Preprocessed DataFrame
            metric: Target metric column name
            
        Returns:
            Tuple of (train_dataset, test_dataset)
            
        Raises:
            ValueError: If insufficient data for sequences
        """
        try:
            if len(df) < self.time_step + 1:
                raise ValueError(
                    f"Insufficient data points ({len(df)}) "
                    f"for sequence length {self.time_step}"
                )
            
            # Split data
            train_size = int(len(df) * self.train_test_split)

            # Create sequences
            train_data = self._create_sequences(
                df[metric][:train_size], 
                self.time_step
            )
            test_data = self._create_sequences(
                df[metric][train_size:], 
                self.time_step
            )
            
            LOG.info(
                f"Created datasets - Train: {len(train_data)}, "
                f"Test: {len(test_data)}"
            )
            return train_data, test_data
            
        except Exception as e:
            LOG.error(f"Dataset creation failed: {str(e)}")
            raise

    def _create_sequences(self, series: pd.Series, 
                         sequence_length: int) -> TimeSeriesDataset:
        """Create sequences for time series prediction.
        
        Creates overlapping sequences where each sequence is used to
        predict the next value in the series.
        
        Args:
            series: Time series data
            sequence_length: Length of input sequences
            
        Returns:
            TimeSeriesDataset with sequences and targets
            
        Note:
            For a series [1,2,3,4,5] and sequence_length=2,
            creates sequences [[1,2],[2,3],[3,4]] with targets [3,4,5]
        """
        data = series.values
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            seq = data[i:(i + sequence_length)]
            target = data[i + sequence_length]
            sequences.append(seq)
            targets.append(target)
            
        return TimeSeriesDataset(
            sequences=np.array(sequences),
            targets=np.array(targets)
        )