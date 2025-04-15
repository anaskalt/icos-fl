======
Server
======

This page documents the server-side components of ICOS-FL, particularly the classes and functions defined in ``icos_fl.server.server`` and ``icos_fl.server.strategy``.

Server Components
-----------------

.. py:function:: icos_fl.server.server.server_fn(context)

   Create and return a Flower server instance.

   This function is used by the ServerApp to create server components
   for the federated learning system.

   :param context: Flower server context
   :type context: Context
   :return: Instantiated ServerAppComponents
   :rtype: ServerAppComponents

.. py:function:: icos_fl.server.server.gen_evaluate_fn(model, metric, time_step, batch_size, device)

   Generate a centralized evaluation function.

   This function creates a callable that can evaluate the global model
   on a centralized test dataset.

   :param model: The LSTM model to evaluate
   :type model: LSTMModel
   :param metric: The metric being predicted
   :type metric: str
   :param time_step: Window size for time series prediction
   :type time_step: int
   :param batch_size: Batch size for data loaders
   :type batch_size: int
   :param device: Device to run evaluation on
   :type device: torch.device
   :return: Function that takes (round, parameters, config) and returns (loss, metrics)
   :rtype: Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]

.. py:function:: icos_fl.server.server.create_on_fit_config_fn(learning_rate)

   Create an on_fit_config function with the specified learning rate.

   :param learning_rate: Learning rate to use in the config
   :type learning_rate: float
   :return: A function that creates client configs with this learning rate
   :rtype: Callable[[int], Dict[str, Scalar]]

.. py:function:: icos_fl.server.server.on_evaluate_config(server_round)

   Return evaluation configuration dict for each round.

   :param server_round: Current federated learning round
   :type server_round: int
   :return: Configuration dictionary with server round
   :rtype: Dict[str, Scalar]

.. py:data:: icos_fl.server.server.app

   The Flower ServerApp instance used for federated learning.

   This is the main entry point for the server-side federated learning process.
   It uses the ``server_fn`` function to create server components.

   :type: ServerApp

Strategy Components
-------------------

.. py:class:: icos_fl.server.strategy.CustomFedAvg(*args, run_config, model, metric, use_wandb=True, save_dir="model", **kwargs)

   Custom FedAvg strategy for ICOS-FL.

   Extends the standard FedAvg strategy with additional functionality:
   - Tracks and logs metrics using Weights & Biases
   - Saves model checkpoints

   :param run_config: Configuration dictionary from context
   :type run_config: Dict[str, Any]
   :param model: The LSTM model used
   :type model: LSTMModel
   :param metric: The metric being predicted
   :type metric: str
   :param use_wandb: Whether to use Weights & Biases for logging, defaults to True
   :type use_wandb: bool, optional
   :param save_dir: Directory for saving model checkpoints, defaults to "model"
   :type save_dir: str, optional
   :param args: Variable length argument list for FedAvg parent class
   :param kwargs: Arbitrary keyword arguments for FedAvg parent class

   .. py:method:: _init_wandb()

      Initialize Weights & Biases project.

   .. py:method:: log_metrics(server_round, metrics_dict)

      Log metrics to Weights & Biases and optionally save them.

      :param server_round: Current federated learning round
      :type server_round: int
      :param metrics_dict: Dictionary containing metrics to log
      :type metrics_dict: Dict[str, Scalar]

   .. py:method:: aggregate_fit(server_round, results, failures)

      Aggregate model updates from clients and update the global model.

      :param server_round: Current round of federated learning
      :type server_round: int
      :param results: List of tuples of (client, fit result)
      :type results: List[Tuple[ClientProxy, FitRes]]
      :param failures: List of failures that occurred during fitting
      :type failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
      :return: Tuple of (new global parameters, aggregation metrics)
      :rtype: Tuple[Optional[Parameters], Dict[str, Scalar]]

   .. py:method:: aggregate_evaluate(server_round, results, failures)

      Aggregate evaluation results from clients.

      :param server_round: Current round of federated learning
      :type server_round: int
      :param results: List of tuples of (client, evaluation result)
      :type results: List[Tuple[ClientProxy, FitRes]]
      :param failures: List of failures that occurred during evaluation
      :type failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
      :return: Tuple of (aggregated loss, aggregation metrics)
      :rtype: Tuple[Optional[float], Dict[str, Scalar]]

   .. py:method:: evaluate(server_round, parameters)

      Evaluate the global model parameters on the server.

      :param server_round: Current round of federated learning
      :type server_round: int
      :param parameters: Current global model parameters
      :type parameters: Parameters
      :return: Tuple of (loss, metrics) or None
      :rtype: Optional[Tuple[float, Dict[str, Scalar]]]

.. py:function:: icos_fl.server.strategy.train_metrics_aggregation(metrics)

   Aggregate training metrics from multiple clients.

   :param metrics: List of tuples (num_examples, metrics_dict)
   :type metrics: List[Tuple[int, Metrics]]
   :return: Aggregated metrics dictionary
   :rtype: Metrics

.. py:function:: icos_fl.server.strategy.evaluate_metrics_aggregation(metrics)

   Aggregate evaluation metrics from multiple clients.

   :param metrics: List of tuples (num_examples, metrics_dict)
   :type metrics: List[Tuple[int, Metrics]]
   :return: Aggregated metrics dictionary
   :rtype: Metrics

Server Workflow
---------------

The server workflow in ICOS-FL follows these steps:

1. **Initialization**: The server is initialized with a strategy and configuration
2. **Client Selection**: The server selects a subset of available clients for the current round
3. **Global Model Distribution**: The server sends the current global model to selected clients
4. **Client Update Collection**: The server receives model updates from clients
5. **Update Aggregation**: The server aggregates client updates using the FedAvg strategy
6. **Global Model Update**: The server updates the global model
7. **Model Evaluation**: The server evaluates the global model
8. **Model Checkpointing**: The server saves the model checkpoint
9. **Iteration**: The process repeats for a configured number of rounds

Example Usage
-------------

.. code-block:: console

   # Run the server with Flower CLI
   $ flwr run . remote-deployment

   # Or programmatically
   from icos_fl.server.server import app

   app.start()
