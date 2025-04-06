=================
LSTM Fundamentals
=================

This page explains the Long Short-Term Memory (LSTM) neural network architecture used in ICOS-FL.

Introduction to LSTM Networks
-----------------------------

Long Short-Term Memory (LSTM) networks are a type of Recurrent Neural Network (RNN) specifically designed to address the vanishing gradient problem that affects standard RNNs.

.. figure:: ../../_static/images/lstm_cell.png
   :alt: LSTM Cell Structure
   :align: center

   LSTM Cell Architecture

LSTMs are particularly well-suited for time series prediction because they can:

1. **Capture Long-term Dependencies**: Remember patterns over extended sequences
2. **Handle Variable-length Sequences**: Process sequences of different lengths
3. **Maintain Context**: Carry forward relevant information while forgetting irrelevant details
4. **Avoid Gradient Problems**: Overcome vanishing/exploding gradients with a specialized gating mechanism

LSTM Architecture
-----------------

Core Components of an LSTM Cell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An LSTM cell consists of several key components:

1. **Cell State (c_t)**: The "memory" of the LSTM, passing information through the sequence
2. **Hidden State (h_t)**: The output for the current time step
3. **Forget Gate (f_t)**: Controls what information to discard from the cell state
4. **Input Gate (i_t)**: Controls what new information to add to the cell state
5. **Output Gate (o_t)**: Controls what information from the cell state to output

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

The LSTM cell performs these operations:

.. math::

   f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \\
   i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
   \tilde{c}_t &= \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \\
   c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
   o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
   h_t &= o_t \odot \tanh(c_t)

Where:
- :math:`\sigma` is the sigmoid activation function
- :math:`\odot` is element-wise multiplication
- :math:`W_f, W_i, W_c, W_o` are weight matrices
- :math:`b_f, b_i, b_c, b_o` are bias vectors
- :math:`x_t` is the input at time step t
- :math:`h_{t-1}` is the hidden state from the previous time step
- :math:`c_{t-1}` is the cell state from the previous time step

LSTM Implementation in ICOS-FL
------------------------------

ICOS-FL uses PyTorch's LSTM implementation:

.. code-block:: python

   class LSTMModel(nn.Module):
       def __init__(
           self,
           hidden_layer_size: int,
           time_step: int,
           num_layers: int,
           output_size: int = 1,
       ) -> None:
           super().__init__()

           self.hidden_layer_size = hidden_layer_size
           self.time_step = time_step
           self.num_layers = num_layers

           # LSTM layer
           self.lstm = nn.LSTM(time_step, hidden_layer_size, num_layers, batch_first=True)

           # Linear layer to produce output prediction
           self.linear = nn.Linear(hidden_layer_size, output_size)

       def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
           lstm_out, _ = self.lstm(input_seq)
           predictions = self.linear(lstm_out[:, -1, :])
           return predictions

Key hyperparameters include:

1. **hidden_layer_size**: Number of units in the LSTM hidden layer
2. **time_step**: Length of input sequences
3. **num_layers**: Number of stacked LSTM layers
4. **output_size**: Dimension of the output prediction (default: 1)

Input and Output Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

ICOS-FL's LSTM model expects inputs and produces outputs with these dimensions:

- **Input**: ``[batch_size, 1, time_step]``
  - batch_size: Number of sequences in a batch
  - 1: Number of features per time step
  - time_step: Length of each sequence

- **Output**: ``[batch_size, output_size]``
  - batch_size: Same as input
  - output_size: Prediction dimension (typically 1 for single-value forecasting)

The model extracts the final hidden state (``lstm_out[:, -1, :]``) and passes it through a linear layer to produce the prediction.

Why LSTM for Time Series Prediction
-----------------------------------

LSTMs are ideal for time series prediction because they can:

1. **Identify Temporal Patterns**: Recognize recurring patterns in system metrics
2. **Remember Past States**: Maintain information about previous system conditions
3. **Handle Variable Sampling**: Work with irregularly sampled data
4. **Detect Anomalies**: Identify unusual patterns in resource usage
5. **Forecast Future Values**: Predict upcoming resource demands

In ICOS-FL, these capabilities are applied to system metrics like CPU usage, memory consumption, and power usage.

Batching and Sequence Creation
------------------------------

ICOS-FL processes time series data into LSTM-compatible batches:

1. **Sliding Window**: Creates overlapping sequences from the time series
2. **Sequence Formatting**: Each sequence consists of `time_step` consecutive data points
3. **Target Creation**: The target is the next value after the sequence
4. **Batching**: Multiple sequences are grouped into batches

This is handled by the ``TimeSeriesDataset`` class:

.. code-block:: python

   class TimeSeriesDataset(Dataset):
       def __init__(
           self,
           df: pd.DataFrame,
           start_index: int,
           population: int,
           time_step: int,
           metric: str,
           device: torch.device,
       ) -> None:
           # Initialize dataset
           # ...

       def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
           # Extract sequence and target
           sequence_values = self.data.iloc[index : index + self.time_step].values
           input_tensor = torch.tensor(sequence_values).unsqueeze(0)

           # Target is the next value after the sequence
           target_value = self.data.iloc[index + self.time_step]
           target_tensor = torch.tensor(target_value).float().unsqueeze(0)

           return input_tensor, target_tensor

Training Process
----------------

The LSTM training process in ICOS-FL follows these steps:

1. **Normalization**: Scale input data to zero mean and unit variance
2. **Sequence Creation**: Generate overlapping sequences from the time series
3. **Batch Formation**: Group sequences into batches
4. **Forward Pass**: Process batches through the LSTM network
5. **Loss Calculation**: Compute Mean Squared Error between predictions and targets
6. **Backward Pass**: Propagate the error gradient backward
7. **Parameter Update**: Update model weights using optimization algorithm
8. **Repeat**: Continue for specified number of epochs

This training process occurs locally at each node in the federated learning setup.

LSTM Advantages for ICOS-FL
---------------------------

LSTM networks provide several advantages in the ICOS-FL context:

1. **Compact Model Size**: LSTM models are relatively small, minimizing communication overhead
2. **Incremental Learning**: Can be updated with new data without complete retraining
3. **Adaptive Memory**: Focus on relevant patterns in varying system loads
4. **Robust to Noise**: Filter out noise in system metrics measurements
5. **Temporal Modeling**: Capture complex temporal dependencies in resource usage

Limitations and Considerations
------------------------------

When using LSTMs in ICOS-FL, consider these limitations:

1. **Training Data Requirements**: LSTMs need sufficient historical data
2. **Hyperparameter Sensitivity**: Performance depends on proper hyperparameter tuning
3. **Computational Cost**: Training can be resource-intensive
4. **Interpretability**: Internal representations are not easily interpretable
5. **Sequence Length Tradeoff**: Longer sequences capture more context but require more resources

ICOS-FL addresses these limitations through careful configuration and the federated learning approach.
