#!/usr/bin/env python3
"""Test module for DataProcessor"""

import sys
import os
import pandas as pd
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from icos_fl.data.processor import DataProcessor, TimeSeriesDataset
from icos_fl.utils.colors import paint, BCYA, FGRY, BWHT, ERR_CLR, INF_CLR

class ProcessorTester:
    def __init__(self):
        self.total = 0
        self.failures = 0
        
        # Test parameters (from config)
        self.fill_nan = 10          # Maximum consecutive NaN values to fill
        self.time_step = 10         # Sequence length for time series
        self.train_test_split = 0.8 # Train/test data split ratio
        self.batch_size = 32        # Training batch size
        
    def _print_header(self):
        print(paint(BWHT, "\n┌────────────────────────────────────────┐"))
        print(paint(BWHT, "│       Testing DataProcessor Module      │"))
        print(paint(BWHT, "└────────────────────────────────────────┘"))
        
    def _print_footer(self):
        print(paint(BWHT, "\n►► Test Summary:"))
        print(paint(FGRY, f"Total tests: {self.total}"))
        print(paint(INF_CLR, f"Passed: {self.total - self.failures}"))
        print(paint(ERR_CLR if self.failures > 0 else INF_CLR, 
                  f"Failed: {self.failures}"))
    
    def _generate_test_data(self, size=200):  # Larger dataset to match real usage
        """Generate sample time series data with some NaN values."""
        dates = pd.date_range(start='2025-01-01', periods=size)
        data = np.sin(np.linspace(0, 4*np.pi, size)) + np.random.normal(0, 0.1, size)
        
        # Insert some NaN values
        nan_indices = np.random.choice(size, size=int(size*0.1), replace=False)
        data[nan_indices] = np.nan
        
        df = pd.DataFrame({
            'timestamp': dates,
            'value': data
        }).set_index('timestamp')
        
        return df
        
    def _test_initialization(self):
        self.total += 1
        try:
            print(paint(BCYA, "\n● Test 1: Processor Initialization"))
            
            processor = DataProcessor(
                fill_nan=self.fill_nan,
                time_step=self.time_step,
                train_test_split=self.train_test_split
            )
            
            assert processor.fill_nan == self.fill_nan
            assert processor.time_step == self.time_step
            assert processor.train_test_split == self.train_test_split
            assert processor.scaler is not None
            
            return paint(INF_CLR, "✓ PASS")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ FAIL: {str(e)}")
    
    def _test_preprocessing(self):
        self.total += 1
        try:
            print(paint(BCYA, "\n● Test 2: Data Preprocessing"))
            
            # Test with sample data
            df = self._generate_test_data()
            processor = DataProcessor(fill_nan=self.fill_nan)
            
            # Check preprocessing
            processed_df = processor.preprocess(df)
            
            # Verify preprocessing results
            assert not processed_df.isna().any().any(), "NaN values still present"
            assert processed_df.shape == df.shape, "Shape changed during preprocessing"
            assert processed_df.index.equals(df.index), "Index changed during preprocessing"
            
            # Test with empty DataFrame
            try:
                processor.preprocess(pd.DataFrame())
                assert False, "Should raise ValueError for empty DataFrame"
            except ValueError:
                pass
            
            return paint(INF_CLR, "✓ PASS")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ FAIL: {str(e)}")
    
    def _test_sequence_creation(self):
        self.total += 1
        try:
            print(paint(BCYA, "\n● Test 3: Sequence Creation"))
            
            # Generate test data
            df = self._generate_test_data(size=200)
            processor = DataProcessor(
                time_step=self.time_step,
                train_test_split=self.train_test_split
            )
            
            # Preprocess and create datasets
            processed_df = processor.preprocess(df)
            train_data, test_data = processor.create_dataset(processed_df, 'value')
            
            # Verify datasets
            assert isinstance(train_data, TimeSeriesDataset)
            assert isinstance(test_data, TimeSeriesDataset)
            
            # Check sequence dimensions
            sample_seq, sample_target = train_data[0]
            assert sample_seq.shape == (self.time_step, 1)
            assert sample_target.shape == (1,)
            
            # Check split ratio
            total_sequences = len(train_data) + len(test_data)
            assert abs(len(train_data)/total_sequences - self.train_test_split) < 0.1
            
            return paint(INF_CLR, "✓ PASS")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ FAIL: {str(e)}")
    
    def _test_dataset_functionality(self):
        self.total += 1
        try:
            print(paint(BCYA, "\n● Test 4: Dataset Functionality"))
            
            # Create small test dataset
            # Create test sequence matching the config time_step
            sequences = np.array([range(i, i+self.time_step) for i in range(3)])
            targets = np.array([i+self.time_step for i in range(3)])
            dataset = TimeSeriesDataset(sequences, targets)
            
            # Test dataset size
            assert len(dataset) == 3
            
            # Test item retrieval
            seq, target = dataset[0]
            assert isinstance(seq, torch.Tensor)
            assert isinstance(target, torch.Tensor)
            assert seq.shape == (self.time_step, 1), f"Expected shape ({self.time_step}, 1), got {seq.shape}"
            assert target.shape == (1,)
            
            # Test all items accessible
            for i in range(len(dataset)):
                seq, target = dataset[i]
                assert not torch.isnan(seq).any()
                assert not torch.isnan(target).any()
            
            return paint(INF_CLR, "✓ PASS")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ FAIL: {str(e)}")
    
    def run(self):
        self._print_header()
        
        print(self._test_initialization())
        print(self._test_preprocessing())
        print(self._test_sequence_creation())
        print(self._test_dataset_functionality())
        
        self._print_footer()
        sys.exit(1 if self.failures > 0 else 0)

if __name__ == "__main__":
    tester = ProcessorTester()
    tester.run()