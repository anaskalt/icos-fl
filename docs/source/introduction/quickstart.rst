==========
Quickstart
==========

This guide helps you get started with ICOS-FL quickly, demonstrating the basic workflow
from installation to running your first federated learning session.

Prerequisites
-------------

- Python 3.10 or newer
- Docker and Docker Compose
- Git

Setup
-----

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/anaskalt/icos-fl.git
      cd icos-fl

2. Create a virtual environment and install requirements:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -r requirements.txt

Starting the Infrastructure
---------------------------

ICOS-FL relies on several services that can be started using Docker Compose:

.. code-block:: bash

   docker compose up -d

This command starts:

- DataClay services (Redis, Metadata Service, Backend, Proxy)
- Scaphandre for hardware metrics collection
- OpenTelemetry collector for metrics processing
- OTLP-DataClay bridge

Configuring the Bridge
----------------------

The bridge configuration is handled automatically through the ``bridgeConfig.py`` script:

.. code-block:: bash

   python bridgeConfig.py

This sets up the metrics collection pipeline to track CPU usage, memory usage, and power consumption.

Running Federated Learning
--------------------------

To start a federated learning session:

1. Start the SuperLink (server):

   .. code-block:: bash

      flwr run . remote-deployment --server-only

2. Start SuperNodes (clients) on different machines or terminals:

   .. code-block:: bash

      # On client machine 1
      flwr run . remote-deployment --client-only --server-address=<server-ip>:9092

      # On client machine 2
      flwr run . remote-deployment --client-only --server-address=<server-ip>:9092

3. Monitor the training process:

   .. code-block:: bash

      python consumer.py

Next Steps
----------

- Read the :doc:`Installation Guide </introduction/installation>` for more detailed setup options
- Learn about :doc:`custom models </how_to/models/custom_models>` to extend the framework
- Explore the :doc:`architecture </explanation/architecture/overview>` to understand the components
- Check the :doc:`deployment guide </how_to/deployment/docker_setup>` for production setups
