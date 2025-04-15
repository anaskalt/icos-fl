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
- Each node, including the Controller, runs its own DataClay and metrics collection stack

The Controller node requires DataClay because it not only coordinates the federated learning process but also collects its own data and performs evaluation of the aggregated model.

Prerequisites
~~~~~~~~~~~~~

- All nodes must be able to communicate over the network
- Each node should have Docker installed
- The Controller node must have a stable IP address or hostname

Controller Node Setup
---------------------

On the machine designated as the Controller:

1. Deploy the SuperLink component:

   .. code-block:: bash

      docker compose -f docker/simulation.yml up -d superlink

2. Note the IP address of the Controller:

   .. code-block:: bash

      ip addr show

   You'll need this IP to configure the worker nodes.

Worker Node Setup
-----------------

On each worker machine:

1. First, modify the ``docker/simulation.yml`` file to update the SuperLink address to point to the Controller:

   .. code-block:: yaml

      # Example for supernode-1
      command:
        - --insecure
        - --superlink
        - CONTROLLER_IP:9092  # Replace with actual controller IP
        - --clientappio-api-address
        - "0.0.0.0:9094"

2. Deploy the SuperNode component:

   .. code-block:: bash

      # Deploy the first supernode
      docker compose -f docker/simulation.yml up -d supernode-1

   For additional worker nodes on separate machines, update the port in the YAML and use a different service:

   .. code-block:: yaml

      # Example for supernode-2
      command:
        - --insecure
        - --superlink
        - CONTROLLER_IP:9092  # Replace with actual controller IP
        - --clientappio-api-address
        - "0.0.0.0:9095"  # Use different port for each node

   .. code-block:: bash

      # Deploy the second supernode
      docker compose -f docker/simulation.yml up -d supernode-2

Federation Configuration
------------------------

Configure the federation in the pyproject.toml file to point to the Controller:

.. code-block:: toml

   [tool.flwr.federations.remote-deployment]
   address = "CONTROLLER_IP:9093"  # Replace with actual controller IP
   insecure = true

Handling Node Failures
----------------------

ICOS-FL can handle nodes joining or leaving the federation:

- Set appropriate ``min_available_clients`` in your configuration to ensure the system can tolerate node failures:

  .. code-block:: toml

     [tool.flwr.app.config]
     min_available_clients = 3  # Federation continues if at least 3 nodes are available
     min_fit_clients = 2  # Training occurs with at least 2 nodes

- Use checkpointing to save model state regularly by enabling it in the SuperLink strategy
- Configure automatic restarts for Docker containers:

  .. code-block:: yaml

     supernode-1:
       # ... other configuration ...
       restart: unless-stopped
