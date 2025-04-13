==========
Quickstart
==========

This guide helps you get started with ICOS-FL quickly, demonstrating the basic workflow from installation to running your first federated learning session.

Prerequisites
-------------

Before you begin, make sure you have:

* Python 3.10 or newer
* Docker and Docker Compose
* Git

Installation
------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/anaskalt/icos-fl.git
      cd icos-fl

2. Install the package:

   .. code-block:: bash

      # Install in development mode with all extras
      pip install -e ".[all]"

      # Or install just the required dependencies
      pip install -e .

Deploying the Communication Layer
---------------------------------

ICOS-FL uses Flower's SuperLink/SuperNode architecture for federated learning communication. This forms the essential communication layer for distributed training:

.. code-block:: bash

   # Deploy SuperLink (controller)
   docker compose -f docker/simulation.yml up -d superlink

   # Deploy SuperNodes (clients)
   docker compose -f docker/simulation.yml up -d supernode-1 supernode-2

.. note::
   In a production environment, you would deploy the SuperNodes on separate machines. Each SuperNode
   simply needs to specify the IP address of the SuperLink to connect to it. The SuperLink machine
   must have the necessary ports open (9091, 9092, 9093) for communication. Check the docker
   compose file for detailed configuration options.

Network Communication
---------------------

Flower's federated learning system establishes two main types of network connections:

1. **CLI to SuperLink (Exec API)**: The ``flwr`` CLI command communicates with the SuperLink via port 9093. This is the only way for users to interact with the deployed federation.

2. **SuperNode to SuperLink (Fleet API)**: Each SuperNode connects to the SuperLink via port 9092. SuperNodes only initiate outgoing connections and don't respond to incoming requests.

For production deployments, TLS should be used, but ``insecure`` mode is supported for local testing.

For more details on Flower's network architecture, see the `Flower Network Communication <https://flower.ai/docs/framework/ref-flower-network-communication.html>`_ documentation.

Running Federated Learning
--------------------------

Starting a federated learning session with ICOS-FL is straightforward. Simply run:

.. code-block:: bash

   flwr run . local-deployment --stream

This command runs the ICOS-FL application using the configuration from your `pyproject.toml`. The ``--stream`` flag shows logs in real-time.

You can use either ``local-deployment`` or ``remote-deployment`` depending on your setup:

* ``local-deployment``: When SuperLink is on localhost
* ``remote-deployment``: When SuperLink is on a remote machine

The difference is just the SuperLink address configured in your `pyproject.toml`:

.. code-block:: toml

   [tool.flwr.federations.local-deployment]
   address = "127.0.0.1:9093"
   insecure = true

   [tool.flwr.federations.remote-deployment]
   address = "127.0.0.1:9093"  # Change to remote IP in production
   insecure = true

Customizing Configuration
-------------------------

You can override any configuration in `pyproject.toml` using the ``--run-config`` parameter without editing the file:

.. code-block:: bash

   # Change the metric to predict
   flwr run . --run-config "metric=power_consumption"

   # Set multiple configuration options
   flwr run . --run-config "num-server-rounds=20 metric=memory_usage min-fit-clients=3"

Common configuration options include:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - num-server-rounds
     - Number of federated learning rounds
   * - metric
     - Resource metric to monitor (cpu_usage, memory_usage, power_consumption)
   * - hidden-layer-size
     - Size of the LSTM hidden layer
   * - time-step
     - Number of time steps for sequence prediction
   * - num-layers
     - Number of LSTM layers
   * - batch-size
     - Batch size for training
   * - local-epochs
     - Number of training epochs per FL round
   * - learning-rate
     - Learning rate for optimization

Here's the relevant section from the `pyproject.toml` file:

.. code-block:: toml

   [tool.flwr.app.config]
   # Server configuration
   num-server-rounds = 10
   fraction-fit = 1.0
   fraction-evaluate = 1.0
   min-fit-clients = 2
   min-evaluate-clients = 2
   min-available-clients = 2
   server-device = "cpu"

   # LSTM model configuration
   hidden-layer-size = 10
   time-step = 10
   num-layers = 1

   # Resource metric to monitor and predict
   metric = "cpu_usage"
   batch-size = 64
   train-test-split = 0.8
   local-epochs = 100
   learning-rate = 0.001

   use-wandb = false

For more information on available CLI commands and options, see the `Flower CLI Reference <https://flower.ai/docs/framework/ref-api-cli.html>`_.

Monitoring Training Progress
----------------------------

During training, you'll see output showing:

1. Model training on each node
2. Model aggregation on the server
3. Evaluation metrics for each round

Trained models are saved in the ``model/`` directory, organized by metric type.

Next Steps
----------

- Explore the :doc:`/how_to/deployment/docker_setup` guide for production deployment
- Learn about :doc:`/how_to/models/custom_models` to extend the framework
- Read about :doc:`/explanation/architecture/overview` to understand the components
- Check out :doc:`/how_to/data/metrics_collection` to set up the metrics collection pipeline

Useful Resources
----------------

- `Flower Architecture <https://flower.ai/docs/framework/explanation-flower-architecture.html>`_: Understand the architecture of Flower federated learning framework
- `Flower CLI Reference <https://flower.ai/docs/framework/ref-api-cli.html>`_: Comprehensive guide to the Flower command line interface
- `Flower Network Communication <https://flower.ai/docs/framework/ref-flower-network-communication.html>`_: Detailed explanation of network communication in Flower
