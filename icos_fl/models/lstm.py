"""LSTM Model implementation for ICOS-FL.

This module implements an LSTM neural network with built-in training
and testing capabilities. The model handles its own optimization,
loss calculation, and state management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from torch.utils.data import DataLoader
from icos_fl.utils.logger import ICOSLogger as LOGGER

LOG = LOGGER("LSTMModel")

class LSTMModel(nn.Module):
   """Long Short-Term Memory neural network with training capabilities.
   
   Attributes:
       input_size: Size of input features
       hidden_size: Number of features in hidden state
       num_layers: Number of stacked LSTM layers
       output_size: Size of output features
       learning_rate: Learning rate for optimizer
       criterion: Loss function (MSE)
       optimizer: Adam optimizer instance
   """

   def __init__(self, input_size: int, hidden_size: int, 
                num_layers: int, output_size: int,
                learning_rate: float = 0.001):
       """Initialize LSTM model architecture and training components.
       
       Args:
           input_size: Size of input features
           hidden_size: Number of features in hidden state
           num_layers: Number of stacked LSTM layers
           output_size: Size of output features
           learning_rate: Learning rate for optimizer
       """
       super().__init__()
       
       self.input_size = input_size
       self.hidden_size = hidden_size
       self.num_layers = num_layers
       self.output_size = output_size
       self.learning_rate = learning_rate
       
       self.lstm = nn.LSTM(
           input_size=input_size,
           hidden_size=hidden_size,
           num_layers=num_layers,
           batch_first=True
       )
       
       self.fc = nn.Linear(hidden_size, output_size)
       self.criterion = nn.MSELoss()
       self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
       
       LOG.info(f"Initialized LSTM: layers={num_layers}, hidden={hidden_size}")

   def forward(self, x: torch.Tensor) -> torch.Tensor:
       """Forward pass through network.
       
       Args:
           x: Input tensor of shape (batch_size, seq_len, input_size)
           
       Returns:
           Output tensor of shape (batch_size, output_size)
       """
       h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
       c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
       
       out, _ = self.lstm(x, (h0, c0))
       out = self.fc(out[:, -1, :])
       return out
       
   def train_epoch(self, train_loader: DataLoader) -> float:
       """Train model for one epoch.
       
       Args:
           train_loader: DataLoader with training batches
           
       Returns:
           Average loss over epoch
           
       Raises:
           RuntimeError: If training fails
       """
       self.train()
       total_loss = 0.0
       
       try:
           for inputs, targets in train_loader:
               self.optimizer.zero_grad()
               outputs = self(inputs)
               loss = self.criterion(outputs, targets)
               loss.backward()
               self.optimizer.step()
               total_loss += loss.item()

           avg_loss = total_loss / len(train_loader)
           LOG.debug(f"Training loss: {avg_loss:.4f}")
           return avg_loss
           
       except Exception as e:
           LOG.error(f"Training failed: {str(e)}")
           raise RuntimeError(f"Training failed: {str(e)}")

   def test(self, test_loader: DataLoader) -> Tuple[float, float]:
       """Evaluate model on test data.
       
       Args:
           test_loader: DataLoader with test batches
           
       Returns:
           Tuple of (average loss, accuracy)
           
       Raises:
           RuntimeError: If testing fails
       """
       self.eval()
       total_loss = 0.0
       predictions = []
       actuals = []

       try:
           with torch.no_grad():
               for inputs, targets in test_loader:
                   outputs = self(inputs)
                   loss = self.criterion(outputs, targets)
                   total_loss += loss.item()
                   predictions.extend(outputs.numpy())
                   actuals.extend(targets.numpy())

           avg_loss = total_loss / len(test_loader)
           accuracy = self._calculate_accuracy(predictions, actuals)
           LOG.info(f"Test loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")
           return avg_loss, accuracy
           
       except Exception as e:
           LOG.error(f"Testing failed: {str(e)}")
           raise RuntimeError(f"Testing failed: {str(e)}")

   def _calculate_accuracy(self, pred: list, actual: list) -> float:
       """Calculate prediction accuracy with 5% error margin.
       
       Args:
           pred: List of predicted values
           actual: List of actual values
           
       Returns:
           Accuracy as fraction of correct predictions
       """
       correct = 0
       total = len(pred)
       threshold = 0.05  # 5% error margin
       
       for p, a in zip(pred, actual):
           if abs(p - a) / a < threshold:
               correct += 1
               
       return correct / total

   def save(self, path: str):
       """Save model state to file.
       
       Args:
           path: Path to save model checkpoint
           
       Raises:
           RuntimeError: If save fails
       """
       try:
           torch.save({
               'state_dict': self.state_dict(),
               'optimizer': self.optimizer.state_dict()
           }, path)
           LOG.info(f"Model saved to {path}")
       except Exception as e:
           LOG.error(f"Failed to save model: {str(e)}")
           raise RuntimeError(f"Failed to save model: {str(e)}")

   def load(self, path: str):
       """Load model state from file.
       
       Args:
           path: Path to model checkpoint
           
       Raises:
           RuntimeError: If load fails
       """
       try:
           checkpoint = torch.load(path)
           self.load_state_dict(checkpoint['state_dict'])
           self.optimizer.load_state_dict(checkpoint['optimizer'])
           LOG.info(f"Model loaded from {path}")
       except Exception as e:
           LOG.error(f"Failed to load model: {str(e)}")
           raise RuntimeError(f"Failed to load model: {str(e)}")