======
Client
======

This page documents the client-side components of ICOS-FL, particularly the classes and functions defined in ``icos_fl.client.client``.

Client Components
-----------------

.. py:class:: icos_fl.client.client.IcosClient(client_id, client_state, model, trainloader, valloader, local_epochs, metric)

   Client implementation for ICOS FL with time series prediction.

   This client trains an LSTM model for time series prediction in a federated learning setting.
   It handles local model training, evaluation, and parameter management as part of the federated
   learning process.

   :param client_id: Unique identifier for this client
   :type client_id: Union[int, str]
   :param client_state: RecordSet to maintain state across rounds
   :type client_state: RecordSet
   :param model: The LSTM model to train
   :type model: LSTMModel
   :param trainloader: DataLoader for training data
   :type trainloader: torch.utils.data.DataLoader
   :param valloader: DataLoader for validation data
   :type valloader: torch.utils.data.DataLoader
   :param local_epochs: Number of local training epochs per FL round
   :type local_epochs: int
   :param metric: Name of the metric being predicted
   :type metric: str

   .. py:method:: get_parameters(config)

      Get model parameters as a list of NumPy arrays.

      :param config: Configuration parameters
      :type config: Dict[str, Any]
      :return: List of model parameter arrays
      :rtype: List[NDArrays]

   .. py:method:: set_parameters(parameters)

      Set model parameters from a list of NumPy arrays.

      :param parameters: List of model parameter arrays
      :type parameters: List[NDArrays]

   .. py:method:: fit(parameters, config)

      Train the model on the local dataset.

      :param parameters: Current global model parameters
      :type parameters: List[NDArrays]
      :param config: Configuration parameters for training
      :type config: Dict[str, Any]
      :return: Tuple containing (updated model parameters, number of training samples, metrics)
      :rtype: Tuple[List[NDArrays], int, Dict[str, Any]]

   .. py:method:: evaluate(parameters, config)

      Evaluate the model on the local validation dataset.

      :param parameters: Current global model parameters
      :type parameters: List[NDArrays]
      :param config: Configuration parameters for evaluation
      :type config: Dict[str, Any]
      :return: Tuple containing (loss, number of validation samples, metrics)
      :rtype: Tuple[float, int, Dict[str, Any]]

.. py:function:: icos_fl.client.client.client_fn(context)

   Create and return a Flower client instance.

   This function is used by the ClientApp to create a client instance
   for each node in the federated learning system. It extracts configuration
   from the context, initializes the model, and prepares data loaders
   for training and evaluation.

   :param context: Flower client context with configuration and node info
   :type context: Context
   :return: Instantiated NumPyClient
   :rtype: NumPyClient

ClientApp Definition
--------------------

.. py:data:: icos_fl.client.client.app

   The Flower ClientApp instance used for federated learning.

   This is the main entry point for the client-side federated learning process.
   It uses the ``client_fn`` function to create client instances.

   :type: ClientApp

Client Workflow
---------------

The client workflow in ICOS-FL follows these steps:

1. **Initialization**: The client is initialized with a model, data loaders, and configuration
2. **Data Fetching**: The client fetches local system metrics data from DataClay
3. **Model Parameter Exchange**: The client receives the global model parameters
4. **Local Training**: The client trains the model on local data for a specified number of epochs
5. **Local Evaluation**: The client evaluates the model on a validation set
6. **Parameter Update**: The client sends the updated parameters back to the server

Example Usage
-------------

.. code-block:: python

   # Run the client with Flower CLI
   $ flwr run . remote-deployment --client-only --server-address=127.0.0.1:9092

   # Or programmatically
   from icos_fl.client.client import app

   app.start(server_address="127.0.0.1:9092")
