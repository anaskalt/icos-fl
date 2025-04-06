=============
Custom Models
=============

This guide shows how to create and integrate custom models with ICOS-FL.

Basic LSTM Model Structure
--------------------------

ICOS-FL uses an LSTM model defined in ``icos_fl/models/lstm.py``. The default model has this structure:

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

Creating a Custom Model
-----------------------

To create a custom model:

1. Subclass PyTorch's ``nn.Module``
2. Implement the required interface (init and forward methods)
3. Make sure your model accepts the expected input shape

Example: Adding Dropout
~~~~~~~~~~~~~~~~~~~~~~~

Here's an example of extending the base LSTM model with dropout for regularization:

.. code-block:: python

   import torch.nn as nn
   from icos_fl.models.lstm import LSTMModel

   class LSTMWithDropout(LSTMModel):
       def __init__(
           self,
           hidden_layer_size: int,
           time_step: int,
           num_layers: int,
           dropout_rate: float = 0.2,
           output_size: int = 1,
       ) -> None:
           super().__init__(
               hidden_layer_size=hidden_layer_size,
               time_step=time_step,
               num_layers=num_layers,
               output_size=output_size,
           )

           # Replace the existing LSTM with one that has dropout
           self.lstm = nn.LSTM(
               time_step,
               hidden_layer_size,
               num_layers,
               batch_first=True,
               dropout=dropout_rate
           )

           # Add dropout before the linear layer
           self.dropout = nn.Dropout(dropout_rate)

       def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
           lstm_out, _ = self.lstm(input_seq)
           lstm_out = self.dropout(lstm_out[:, -1, :])
           predictions = self.linear(lstm_out)
           return predictions

Integrating the Custom Model
----------------------------

To use your custom model in ICOS-FL, modify the client code to use your model instead of the default one.

1. Import your custom model:

   .. code-block:: python

      from path.to.your.model import LSTMWithDropout

2. Modify the model creation in the client:

   .. code-block:: python

      # In client_fn function in client.py
      model = LSTMWithDropout(
          hidden_layer_size=hidden_layer_size,
          time_step=time_step,
          num_layers=num_layers,
          dropout_rate=0.2
      ).to(device)

3. Ensure your model is serializable using Flower's model serialization:

   .. code-block:: python

      # These functions must work with your custom model
      from icos_fl.models.lstm import get_weights, set_weights

      weights = get_weights(model)  # Extract weights as numpy arrays
      set_weights(model, weights)   # Set weights from numpy arrays

Example: Bidirectional LSTM
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more advanced sequence modeling, here's a bidirectional LSTM implementation:

.. code-block:: python

   class BidirectionalLSTM(nn.Module):
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

           # Bidirectional LSTM
           self.lstm = nn.LSTM(
               time_step,
               hidden_layer_size,
               num_layers,
               batch_first=True,
               bidirectional=True
           )

           # The output size is doubled due to bidirectionality
           self.linear = nn.Linear(hidden_layer_size * 2, output_size)

       def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
           lstm_out, _ = self.lstm(input_seq)
           predictions = self.linear(lstm_out[:, -1, :])
           return predictions

Weight Handling for Custom Models
---------------------------------

When implementing custom models, ensure the ``get_weights`` and ``set_weights`` functions work with your model structure:

.. code-block:: python

   def get_weights_for_custom_model(model: nn.Module) -> List[np.ndarray]:
       """Extract model weights for custom model."""
       return [val.cpu().numpy() for _, val in model.state_dict().items()]

   def set_weights_for_custom_model(model: nn.Module, weights: List[np.ndarray]) -> None:
       """Set model weights for custom model."""
       params_dict = zip(model.state_dict().keys(), weights, strict=False)
       state_dict = {k: torch.tensor(v) for k, v in params_dict}
       model.load_state_dict(state_dict, strict=True)
