=======
Scaling
=======

This guide explains how to scale ICOS-FL across multiple machines.

Multi-Node Federated Deployment
-------------------------------

ICOS-FL is designed to operate in a distributed setting with multiple nodes participating in federated learning.

Architecture Overview
~~~~~~~~~~~~~~~~~~~~~

In a multi-node deployment:

- One node acts as the **Controller** (runs SuperLink)
- Multiple nodes act as **Workers** (run SuperNode)
- Each node runs its own DataClay and metrics collection stack

Prerequisites
~~~~~~~~~~~~~

- All nodes must be able to communicate over the network
- Each node should have Docker installed
- The Controller node must have a stable IP address or hostname

Controller Node Setup
---------------------

On the machine designated as the Controller:

1. Deploy the base infrastructure:

   .. code-block:: bash

      docker compose up -d

2. Deploy the SuperLink component:

   .. code-block:: bash

      docker compose -f docker/simulation.yml up -d superlink

3. Note the IP address of the Controller:

   .. code-block:: bash

      ip addr show

   You'll need this IP to configure the worker nodes.

Worker Node Setup
-----------------

On each worker machine:

1. Deploy the base infrastructure:

   .. code-block:: bash

      docker compose up -d

2. Deploy the SuperNode component, pointing to the Controller:

   .. code-block:: bash

      # Replace CONTROLLER_IP with the actual IP address of the controller
      docker compose -f docker/simulation.yml up -d supernode-1 \
        --superlink CONTROLLER_IP:9092

   For additional worker nodes, use supernode-2, supernode-3, etc., with unique ports:

   .. code-block:: bash

      docker compose -f docker/simulation.yml up -d supernode-2 \
        --superlink CONTROLLER_IP:9092 \
        --clientappio-api-address "0.0.0.0:9095"

Federation Configuration
------------------------

Configure the federation in the pyproject.toml file:

.. code-block:: toml

   [tool.flwr.federations.remote-deployment]
   address = "CONTROLLER_IP:9093"
   insecure = true

Scaling DataClay
----------------

For high-throughput deployments, you can scale the DataClay components:

1. Scale the backend service:

   .. code-block:: bash

      docker compose up -d --scale backend=3

2. Configure a load balancer (like Nginx) in front of multiple proxy instances

Optimizing Performance
----------------------

Batch Size Tuning
~~~~~~~~~~~~~~~~~

Adjust batch processing settings for higher throughput:

.. code-block:: yaml

   # In otel-config.yaml
   processors:
     batch:
       timeout: 180s  # Increase for less frequent but larger updates

Memory Optimization
~~~~~~~~~~~~~~~~~~~

Control the sliding window size to manage memory usage:

.. code-block:: python

   # In model/timeseries.py
   TimeSeriesData(max_rows=300)  # Adjust based on memory constraints

Network Bandwidth Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Federated learning can be bandwidth-intensive. Options to manage this:

1. Increase training rounds interval
2. Reduce model complexity
3. Use model compression techniques

Monitoring Scaled Deployments
-----------------------------

Use the consumer.py script to monitor each node's data collection:

.. code-block:: bash

   python consumer.py --host=NODE_IP

For cluster-wide monitoring:

1. Set up Prometheus and Grafana
2. Use Docker's built-in metrics
3. Export OpenTelemetry metrics to your monitoring system

Handling Node Failures
----------------------

ICOS-FL can handle nodes joining or leaving the federation:

- Set appropriate `min_available_clients` in your configuration
- Use checkpointing to save model state regularly
- Configure automatic restarts for Docker containers

.. code-block:: yaml

   restart: unless-stopped
