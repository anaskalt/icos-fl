=============
Configuration
=============

This page documents the configuration options available in ICOS-FL.

Configuration Files
-------------------

ICOS-FL uses several configuration files:

1. **pyproject.toml**: Main project configuration
2. **otel-config.yaml**: OpenTelemetry collector configuration
3. **docker-compose.yml**: Container configuration
4. **bridgeConfig.py**: Bridge configuration script

pyproject.toml Configuration
----------------------------

The ``pyproject.toml`` file is the primary configuration file for ICOS-FL. It is divided into multiple sections:

Build System
~~~~~~~~~~~~

.. code-block:: toml

   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

Project Metadata
~~~~~~~~~~~~~~~~

.. code-block:: toml

   [project]
   name = "icos-fl"
   version = "0.1.0"
   description = "ICOS-FL: Flower-powered FL framework for real-time resource monitoring (LSTM) & predictions."
   license = "MIT"
   readme = "README.md"
   requires-python = ">=3.10"
   authors = [
     { name = "Anastasios Kaltakis", email = "anastasioskaltakis@gmail.com" },
   ]
   dependencies = [
     "flwr[simulation]>=1.17.0",
     "torch==2.5.1",
     "wandb==0.19.8",
     "pandas>=2.2.3",
     "scikit-learn>=1.6.1",
     "dataclay==4.0.0",
   ]

Flower Application Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [tool.flwr.app]
   publisher = "Anastasios Kaltakis"

   [tool.flwr.app.components]
   serverapp = "icos_fl.server.server:app"
   clientapp = "icos_fl.client.client:app"

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

Federation Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [tool.flwr.federations]
   default = "local-deployment"

   [tool.flwr.federations.local-deployment]
   address = "127.0.0.1:9093"
   insecure = true

   [tool.flwr.federations.remote-deployment]
   address = "127.0.0.1:9093"
   insecure = true

OpenTelemetry Configuration
---------------------------

The ``otel-config.yaml`` file configures the OpenTelemetry collector:

.. code-block:: yaml

   receivers:
     prometheus:
       config:
         scrape_configs:
           - job_name: 'scaphandre'
             scrape_interval: 3s  # Scrape metrics every 3 seconds
             static_configs:
               - targets: ['127.0.0.1:8080']

   processors:
     batch:
       timeout: 180s  # Batch metrics for 3 minutes

   exporters:
     otlp:
       endpoint: 127.0.0.1:4317
       tls:
         insecure: true

   service:
     pipelines:
       metrics:
         receivers: [prometheus]
         processors: [batch]
         exporters: [otlp]

Docker Compose Configuration
----------------------------

The ``docker-compose.yml`` file configures the Docker containers:

.. code-block:: yaml

   services:
     redis:
       image: redis:latest
       restart: unless-stopped

     scaphandre:
       image: docker.io/hubblo/scaphandre
       command: prometheus -p 8080 -a 0.0.0.0
       privileged: true
       # ...

     proxy:
       build: .
       ports:
         - 8676:8676
       depends_on:
         - metadata-service
         - backend
       environment:
         - DATACLAY_PROXY_MDS_HOST=metadata-service
         - DATACLAY_KV_HOST=redis
       command: python -m dataclay.proxy
       # ...

Bridge Configuration
--------------------

The bridge configuration is set through the ``bridgeConfig.py`` script:

.. code-block:: python

   # Create a ResourceConfiguration for Scaphandre
   rc_scaphandre = ResourceConfiguration("scaphandre-metrics", scaphandre_rules)

   # Add the specific metrics you want to track
   rc_scaphandre.add_metric("scaph_host_power_microwatts")
   rc_scaphandre.add_metric("scaph_host_load_avg_one")
   rc_scaphandre.add_metric("scaph_host_memory_total_bytes")
   rc_scaphandre.add_metric("scaph_host_memory_available_bytes")

   # Add this configuration to the bridge
   bc.set_res_config(rc_scaphandre)

Configuration Parameters
------------------------

Server Configuration
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - num-server-rounds
     - 10
     - Number of federated learning rounds
   * - fraction-fit
     - 1.0
     - Fraction of clients to select for training
   * - fraction-evaluate
     - 1.0
     - Fraction of clients to select for evaluation
   * - min-fit-clients
     - 2
     - Minimum number of clients for training
   * - min-evaluate-clients
     - 2
     - Minimum number of clients for evaluation
   * - min-available-clients
     - 2
     - Minimum clients before starting a round
   * - server-device
     - "cpu"
     - Device to use for server-side operations

LSTM Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 15 55
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

Training Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - batch-size
     - 64
     - Batch size for training
   * - train-test-split
     - 0.8
     - Ratio for train/validation split
   * - local-epochs
     - 100
     - Number of local training epochs per round
   * - metric
     - "cpu_usage"
     - Metric to predict (cpu_usage, memory_usage, power_consumption)
   * - use-wandb
     - false
     - Whether to use Weights & Biases for logging

Time Series Data Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - max_rows
     - 300
     - Maximum rows in the sliding window
   * - scrape_interval
     - 3s
     - Interval between metrics scrapes
   * - batch_timeout
     - 180s
     - Interval for batching metrics

Environment Variables
---------------------

ICOS-FL respects these environment variables:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Description
   * - DATACLAY_PROXY_HOST
     - Host address for the DataClay proxy
   * - DATACLAY_PROXY_PORT
     - Port for the DataClay proxy
   * - BRIDGE_CONFIGURATION_ALIAS
     - Alias for the bridge configuration
   * - TIMESERIES_ALIAS
     - Alias for the TimeSeriesData object
   * - LOG_LEVEL
     - Logging level (DEBUG, INFO, WARNING, ERROR)

Example Configuration
---------------------

Here's an example of a complete configuration for predicting memory usage with a larger model:

.. code-block:: toml

   [tool.flwr.app.config]
   # Server configuration
   num-server-rounds = 20
   min-fit-clients = 3
   min-evaluate-clients = 3
   min-available-clients = 3

   # LSTM model configuration
   hidden-layer-size = 20
   time-step = 15
   num-layers = 2

   # Training configuration
   metric = "memory_usage"
   batch-size = 32
   local-epochs = 150
   learning-rate = 0.0005

   # Logging
   use-wandb = true
