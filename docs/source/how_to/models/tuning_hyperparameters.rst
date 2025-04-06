======================
Tuning Hyperparameters
======================

This guide explains how to optimize model hyperparameters in ICOS-FL.

Configuration Options
---------------------

ICOS-FL's hyperparameters can be configured in several ways:

1. **pyproject.toml** - For default settings
2. **Command line arguments** - For runtime overrides
3. **Environment variables** - For deployment-specific settings

Available Hyperparameters
-------------------------

LSTM Model Parameters
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - hidden-layer-size
     - 10
     - Size of the LSTM hidden layer
   * - time-step
     - 10
     - Number of time steps in input sequence
   * - num-layers
     - 1
     - Number of LSTM layers
   * - learning-rate
     - 0.001
     - Learning rate for model optimization

Training Parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - batch-size
     - 64
     - Batch size for training
   * - local-epochs
     - 100
     - Number of training epochs per round
   * - train-test-split
     - 0.8
     - Ratio for train/validation split

Federated Learning Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - num-server-rounds
     - 10
     - Number of federated learning rounds
   * - min-fit-clients
     - 2
     - Minimum clients for training
   * - min-evaluate-clients
     - 2
     - Minimum clients for evaluation
   * - min-available-clients
     - 2
     - Minimum available clients to start round

Setting Parameters in pyproject.toml
------------------------------------

Configure hyperparameters in the ``pyproject.toml`` file:

.. code-block:: toml

   [tool.flwr.app.config]
   # LSTM model configuration
   hidden-layer-size = 20
   time-step = 15
   num-layers = 2

   # Training configuration
   batch-size = 32
   local-epochs = 50
   learning-rate = 0.0005

   # Federated learning configuration
   num-server-rounds = 20
   min-fit-clients = 3

Configuring via Command Line
----------------------------

Override settings at runtime:

.. code-block:: bash

   flwr run . remote-deployment \
     --config.hidden-layer-size=30 \
     --config.local-epochs=75 \
     --config.learning-rate=0.0002

Hyperparameter Search with Weights & Biases
-------------------------------------------

ICOS-FL integrates with Weights & Biases for hyperparameter tuning:

1. Enable W&B integration:

   .. code-block:: toml

      [tool.flwr.app.config]
      use-wandb = true

2. Create a sweep configuration file (sweep.yaml):

   .. code-block:: yaml

      program: main.py
      method: bayes
      metric:
        name: val_loss
        goal: minimize
      parameters:
        hidden-layer-size:
          min: 10
          max: 50
        learning-rate:
          min: 0.0001
          max: 0.01
          distribution: log_uniform
        num-layers:
          values: [1, 2, 3]
        time-step:
          values: [5, 10, 15, 20]
        local-epochs:
          values: [50, 100, 150]

3. Initialize and run the sweep:

   .. code-block:: bash

      wandb sweep sweep.yaml
      wandb agent SWEEP_ID

Tuning for Different Metrics
----------------------------

Different resource metrics may require different hyperparameters:

CPU Usage
~~~~~~~~~

CPU usage patterns benefit from:
- Smaller time windows (time-step: 8-12)
- Fewer layers (num-layers: 1-2)
- Moderate hidden layer size (hidden-layer-size: 10-20)

Memory Usage
~~~~~~~~~~~~

Memory patterns benefit from:
- Longer time windows (time-step: 15-20)
- More layers (num-layers: 2-3)
- Larger hidden layer size (hidden-layer-size: 20-40)

Power Consumption
~~~~~~~~~~~~~~~~~

Power consumption patterns benefit from:
- Medium time windows (time-step: 10-15)
- More layers (num-layers: 2)
- Moderate hidden layer size (hidden-layer-size: 15-25)

Recommended Process
-------------------

1. Start with default hyperparameters
2. Monitor performance with consumer.py
3. Adjust parameters based on the specific metric
4. Use W&B for systematic hyperparameter search
5. Save best configuration to pyproject.toml

Best Practices
--------------

1. **Start small**: Begin with fewer clients and epochs to speed up experimentation
2. **One at a time**: Change one parameter at a time to understand its impact
3. **Cross-validation**: Use multiple random seeds for more robust results
4. **Resource constraints**: Consider hardware limitations when setting batch sizes
5. **Balance training/communication**: More local epochs means less communication but potential convergence issues
