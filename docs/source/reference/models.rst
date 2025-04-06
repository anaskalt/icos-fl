======
Models
======

This page documents the model components of ICOS-FL, particularly the classes and functions defined in ``icos_fl.models.lstm``.

LSTM Model Components
---------------------

.. py:class:: icos_fl.models.lstm.LSTMModel(hidden_layer_size, time_step, num_layers, output_size=1)

   LSTM model for time series prediction.

   :param hidden_layer_size: Size of the LSTM hidden layer
   :type hidden_layer_size: int
   :param time_step: Number of time steps (sequence length) for LSTM input
   :type time_step: int
   :param num_layers: Number of LSTM layers
   :type num_layers: int
   :param output_size: Size of the output layer (default is 1 for single value prediction)
   :type output_size: int, optional

   .. py:method:: forward(input_seq)

      Forward pass through the network.

      :param input_seq: Input tensor
      :type input_seq: torch.Tensor
      :return: Output tensor of predictions
      :rtype: torch.Tensor

Model Utility Functions
-----------------------

.. py:function:: icos_fl.models.lstm.get_weights(model)

   Extract model weights as a list of NumPy arrays.

   :param model: PyTorch model
   :type model: nn.Module
   :return: List of NumPy arrays containing model weights
   :rtype: List[np.ndarray]

.. py:function:: icos_fl.models.lstm.set_weights(model, weights)

   Set model weights from a list of NumPy arrays.

   :param model: PyTorch model
   :type model: nn.Module
   :param weights: List of NumPy arrays containing weights
   :type weights: List[np.ndarray]

.. py:function:: icos_fl.models.lstm.train(model, train_dataloader, epochs, lr=0.001, criterion=None, optimizer=None)

   Train the LSTM model.

   :param model: The LSTM model to train
   :type model: LSTMModel
   :param train_dataloader: DataLoader containing training data
   :type train_dataloader: DataLoader
   :param epochs: Number of training epochs
   :type epochs: int
   :param lr: Learning rate, defaults to 0.001
   :type lr: float, optional
   :param criterion: Loss function (defaults to MSE if None)
   :type criterion: nn.Module, optional
   :param optimizer: Optimizer (defaults to Adam if None)
   :type optimizer: optim.Optimizer, optional
   :return: Average training loss
   :rtype: float

.. py:function:: icos_fl.models.lstm.test(model, test_dataloader, device)

   Evaluate model on the test dataset.

   :param model: The LSTM model to evaluate
   :type model: LSTMModel
   :param test_dataloader: DataLoader containing test data
   :type test_dataloader: DataLoader
   :param device: PyTorch device for computation
   :type device: torch.device
   :return: Average test loss
   :rtype: float

Model Architecture
------------------

The default LSTM model architecture in ICOS-FL consists of:

1. **LSTM Layer**: Processes input sequences
   - Input size: ``time_step`` (sequence length)
   - Hidden size: ``hidden_layer_size`` (configurable)
   - Number of layers: ``num_layers`` (configurable)

2. **Linear Layer**: Produces final predictions
   - Input size: ``hidden_layer_size``
   - Output size: ``output_size`` (defaults to 1)

The model configuration can be adjusted through the pyproject.toml file:

.. code-block:: toml

   [tool.flwr.app.config]
   # LSTM model configuration
   hidden-layer-size = 10
   time-step = 10
   num-layers = 1

Example Usage
-------------

.. code-block:: python

   from icos_fl.models.lstm import LSTMModel, train, test, get_weights, set_weights
   import torch

   # Create model
   model = LSTMModel(
       hidden_layer_size=10,
       time_step=10,
       num_layers=1
   )

   # Train model
   train_loss = train(
       model=model,
       train_dataloader=train_dataloader,
       epochs=100,
       lr=0.001
   )

   # Evaluate model
   test_loss = test(
       model=model,
       test_dataloader=test_dataloader,
       device=torch.device("cpu")
   )

   # Get model weights (for federated learning)
   weights = get_weights(model)

   # Set model weights (from federated learning)
   set_weights(model, weights)
