#!/usr/bin/env python3
"""Test module for LSTMModel"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from icos_fl.models.lstm import LSTMModel
from icos_fl.utils.colors import paint, BCYA, FGRY, BWHT, ERR_CLR, INF_CLR

class LSTMTester:
    def __init__(self):
        self.test_dir = "/tmp/lstm_model_test"
        self.total = 0
        self.failures = 0
        
        # Model Architecture (matching production settings)
        self.input_size = 10        # Number of input features
        self.hidden_size = 32       # Number of hidden units in LSTM layers
        self.num_layers = 2         # Number of stacked LSTM layers
        self.output_size = 1        # Number of output features
        self.learning_rate = 0.001  # Learning rate
        
        # Training Settings
        self.batch_size = 32        # Training batch size
        self.seq_length = 10        # Time step/sequence length
        self.epochs = 100           # Number of training epochs
        self.train_test_split = 0.8 # Train/test split ratio
        self.fill_nan = 10          # Max consecutive NaN values to fill
        
    def _print_header(self):
        print(paint(BWHT, "\n┌────────────────────────────────────────┐"))
        print(paint(BWHT, "│          Testing LSTM Module           │"))
        print(paint(BWHT, "└────────────────────────────────────────┘"))
        
    def _print_footer(self):
        print(paint(BWHT, "\n►► Test Summary:"))
        print(paint(FGRY, f"Total tests: {self.total}"))
        print(paint(INF_CLR, f"Passed: {self.total - self.failures}"))
        print(paint(ERR_CLR if self.failures > 0 else INF_CLR, 
                  f"Failed: {self.failures}"))
    
    def _generate_dummy_data(self, num_samples):
        """Generate dummy sequential data for testing with NaN handling.
        
        Creates realistic test data with some NaN values to test robustness.
        """
        # Generate base data
        X = torch.randn(num_samples, self.seq_length, self.input_size)
        y = torch.randn(num_samples, self.output_size)
        
        # Insert some NaN values (randomly) to test NaN handling
        nan_mask = torch.rand(X.shape) > 0.95  # 5% NaN values
        X[nan_mask] = float('nan')
        
        # Fill NaN values (similar to production)
        X = self._fill_nan_values(X)
        
        return TensorDataset(X, y)
        
    def _fill_nan_values(self, data):
        """Fill NaN values using forward fill, similar to production."""
        # Convert to numpy for easier NaN handling
        data_np = data.numpy()
        
        # Forward fill NaN values
        for i in range(len(data_np)):
            for j in range(self.input_size):
                mask = np.isnan(data_np[i, :, j])
                
                # Only fill if consecutive NaNs are within limit
                nan_regions = np.split(mask, np.where(np.diff(mask))[0] + 1)
                for region in nan_regions:
                    if sum(region) <= self.fill_nan:  # Check against max fill limit
                        idx = np.where(mask)[0]
                        if len(idx) > 0:
                            # Forward fill
                            last_valid = data_np[i, :, j][~mask][-1] if any(~mask) else 0
                            data_np[i, idx, j] = last_valid
        
        return torch.from_numpy(data_np).float()
    
    def _test_initialization(self):
        self.total += 1
        try:
            print(paint(BCYA, "\n● Test 1: Model Initialization"))
            model = LSTMModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=self.output_size,
                learning_rate=self.learning_rate
            )
            
            # Check model attributes
            assert model.input_size == self.input_size
            assert model.hidden_size == self.hidden_size
            assert model.num_layers == self.num_layers
            assert model.output_size == self.output_size
            assert model.learning_rate == self.learning_rate
            
            # Check layer dimensions
            assert model.lstm.input_size == self.input_size
            assert model.lstm.hidden_size == self.hidden_size
            assert model.lstm.num_layers == self.num_layers
            
            return paint(INF_CLR, "✓ PASS")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ FAIL: {str(e)}")
    
    def _test_forward_pass(self):
        self.total += 1
        try:
            print(paint(BCYA, "\n● Test 2: Forward Pass"))
            model = LSTMModel(
                self.input_size, self.hidden_size,
                self.num_layers, self.output_size
            )
            
            # Test single sample
            x = torch.randn(1, self.seq_length, self.input_size)
            output = model(x)
            assert output.shape == (1, self.output_size)
            
            # Test batch
            x_batch = torch.randn(self.batch_size, self.seq_length, self.input_size)
            output_batch = model(x_batch)
            assert output_batch.shape == (self.batch_size, self.output_size)
            
            return paint(INF_CLR, "✓ PASS")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ FAIL: {str(e)}")
    
    def _test_training(self):
        self.total += 1
        try:
            print(paint(BCYA, "\n● Test 3: Training"))
            model = LSTMModel(
                self.input_size, self.hidden_size,
                self.num_layers, self.output_size
            )
            
            # Generate dummy training data
            # Generate larger dataset to match production scale
            total_samples = 1000  # Larger dataset for realistic testing
            train_size = int(total_samples * self.train_test_split)
            
            # Create train dataset
            train_data = self._generate_dummy_data(train_size)
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            
            # Train for 2 epochs and check loss decreases
            loss1 = model.train_epoch(train_loader)
            loss2 = model.train_epoch(train_loader)
            assert loss2 < loss1, "Training loss did not decrease"
            
            return paint(INF_CLR, "✓ PASS")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ FAIL: {str(e)}")
    
    def _test_testing(self):
        self.total += 1
        try:
            print(paint(BCYA, "\n● Test 4: Testing"))
            model = LSTMModel(
                self.input_size, self.hidden_size,
                self.num_layers, self.output_size
            )
            
            # Generate dummy test data
            test_data = self._generate_dummy_data(50)
            test_loader = DataLoader(test_data, batch_size=self.batch_size)
            
            # Test evaluation
            test_loss, accuracy = model.test(test_loader)
            assert isinstance(test_loss, float)
            assert isinstance(accuracy, float)
            assert 0 <= accuracy <= 1
            
            return paint(INF_CLR, "✓ PASS")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ FAIL: {str(e)}")
    
    def _test_save_load(self):
        self.total += 1
        try:
            print(paint(BCYA, "\n● Test 5: Save/Load Model"))
            model = LSTMModel(
                self.input_size, self.hidden_size,
                self.num_layers, self.output_size
            )
            
            # Generate test input
            x_test = torch.randn(1, self.seq_length, self.input_size)
            output_before = model(x_test)
            
            # Save model
            save_path = os.path.join(self.test_dir, "model.pt")
            model.save(save_path)
            assert os.path.exists(save_path)
            
            # Create new model and load saved weights
            new_model = LSTMModel(
                self.input_size, self.hidden_size,
                self.num_layers, self.output_size
            )
            new_model.load(save_path)
            
            # Compare outputs
            output_after = new_model(x_test)
            assert torch.allclose(output_before, output_after)
            
            return paint(INF_CLR, "✓ PASS")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ FAIL: {str(e)}")
    
    def run(self):
        self._print_header()
        
        if os.path.exists(self.test_dir):
            os.system(f"rm -rf {self.test_dir}")
        os.makedirs(self.test_dir)
        
        print(self._test_initialization())
        print(self._test_forward_pass())
        print(self._test_training())
        print(self._test_testing())
        print(self._test_save_load())
        
        self._print_footer()
        sys.exit(1 if self.failures > 0 else 0)

if __name__ == "__main__":
    tester = LSTMTester()
    tester.run()