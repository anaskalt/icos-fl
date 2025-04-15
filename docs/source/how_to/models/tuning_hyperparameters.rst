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
   :header-rows: 1
   :align: left

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
   :header-rows: 1
   :align: left

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
   :header-rows: 1
   :align: left

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

Override settings at runtime using the ``--run-config`` parameter:

.. code-block:: bash

   # Modify a single parameter
   flwr run . remote-deployment --run-config hidden-layer-size=30

   # Modify multiple parameters (use quotes)
   flwr run . remote-deployment --run-config "local-epochs=75 learning-rate=0.0002 num-server-rounds=15"

   # Using a different metric
   flwr run . remote-deployment --run-config "metric=power_consumption batch-size=32"

Hyperparameter Search with Weights & Biands
-------------------------------------------

ICOS-FL integrates with Weights & Biands (W&B) for hyperparameter tuning:

1. Enable W&B integration in your configuration:

   .. code-block:: bash

      flwr run . remote-deployment --run-config "use-wandb=true"

   Or in your ``pyproject.toml``:

   .. code-block:: toml

      [tool.flwr.app.config]
      use-wandb = true

2. Set up a W&B account and log in:

   .. code-block:: bash

      pip install wandb
      wandb login

3. ICOS-FL will automatically log metrics to W&B during training, allowing you to compare performance across different hyperparameter settings.

4. For systematic parameter tuning, you can use W&B Sweeps to explore the hyperparameter space:

   .. code-block:: bash

      # Create a sweep configuration
      wandb sweep sweep.yaml

      # Run sweep agents
      wandb agent SWEEP_ID

Tuning for Different Metrics
----------------------------

Different resource metrics have unique patterns that benefit from specific hyperparameter configurations:

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 25 75

   * - Metric
     - Recommended Hyperparameters
   * - **CPU Usage**
     - | • **Time Window**: 8-12 time steps (shorter sequences)
       | • **Layers**: 1-2 LSTM layers (simpler architecture)
       | • **Hidden Size**: 10-20 units (moderate complexity)
       | • **Batch Size**: 32-64 (standard)
   * - **Memory Usage**
     - | • **Time Window**: 15-20 time steps (longer sequences)
       | • **Layers**: 2-3 LSTM layers (deeper architecture)
       | • **Hidden Size**: 20-40 units (higher complexity)
       | • **Batch Size**: 16-32 (smaller for more updates)
   * - **Power Consumption**
     - | • **Time Window**: 10-15 time steps (medium sequences)
       | • **Layers**: 2 LSTM layers (balanced architecture)
       | • **Hidden Size**: 15-25 units (balanced complexity)
       | • **Batch Size**: 32 (balanced)

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
