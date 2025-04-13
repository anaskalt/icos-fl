============
Installation
============

This page provides detailed installation instructions for ICOS-FL and its dependencies.

Requirements
------------

ICOS-FL requires:

- Python 3.10 or newer
- Docker and Docker Compose (for deploying the federated learning infrastructure)
- NVIDIA drivers (optional, for GPU acceleration)

Python Dependencies
-------------------

ICOS-FL depends on:

- flwr>=1.17.0 (Flower federated learning framework)
- torch==2.5.1 (PyTorch for LSTM models)
- pandas>=2.2.3 (Data manipulation)
- scikit-learn>=1.6.1 (Data preprocessing)
- dataclay==4.0.0 (Distributed data storage)
- wandb==0.19.8 (optional, for experiment tracking)

Installing ICOS-FL
------------------

To install ICOS-FL, clone the repository and install the package in editable mode:

.. code-block:: bash

   git clone https://github.com/anaskalt/icos-fl.git
   cd icos-fl
   pip install -e .

For development, install with additional dependencies:

.. code-block:: bash

   pip install -e ".[all]"  # Installs all extras (dev, test, docs)

   # Or install specific extras
   pip install -e ".[dev]"   # Development tools
   pip install -e ".[test]"  # Testing tools
   pip install -e ".[docs]"  # Documentation tools

Setting Up Flower Infrastructure
--------------------------------

ICOS-FL uses Flower's SuperLink and SuperNode components for federated learning. These components need to be deployed separately.

Using Docker
^^^^^^^^^^^^

The easiest way to deploy the Flower infrastructure is using Docker:

.. code-block:: bash

   # Start the SuperLink (controller)
   docker compose -f docker/simulation.yml up -d superlink

   # Start the SuperNodes (clients)
   docker compose -f docker/simulation.yml up -d supernode-1 supernode-2

This sets up:

1. A SuperLink on ports 9091 (ServerAppIO), 9092 (Fleet), and 9093 (Exec)
2. Two SuperNodes (supernode-1 on port 9094 and supernode-2 on port 9095)

Manual Installation
^^^^^^^^^^^^^^^^^^^

For production environments or custom setups, you can manually install and run Flower components:

.. code-block:: bash

   # Install Flower CLI globally
   pip install flwr

   # Start SuperLink
   flower-superlink --insecure

   # Start SuperNodes (on separate machines/terminals)
   flower-supernode --insecure --superlink <controller-ip>:9092 --clientappio-api-address 0.0.0.0:9094

See the `Flower CLI Reference <https://flower.ai/docs/framework/ref-api-cli.html>`_ for all available options.

Setting Up Data Collection Pipeline
-----------------------------------

For resource monitoring, ICOS-FL requires a metrics collection pipeline. This is handled by the OTLP-DataClay Bridge:

1. Clone the OTLP-DataClay Bridge repository
2. Start the services:

   .. code-block:: bash

      cd /path/to/otlp-dataclay-bridge
      docker compose up -d

See :doc:`/how_to/data/metrics_collection` for detailed instructions on setting up the data collection pipeline.

Optional Components
-------------------

GPU Support
^^^^^^^^^^^

To enable GPU acceleration for the LSTM models:

1. Install NVIDIA drivers and CUDA toolkit
2. Install PyTorch with CUDA support:

   .. code-block:: bash

      pip install torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu118

In your configuration, set:

.. code-block:: toml

   # In pyproject.toml
   [tool.flwr.app.config]
   server-device = "cuda:0"  # or other GPU device

Weights & Biases Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For experiment tracking with Weights & Biands:

.. code-block:: bash

   pip install wandb==0.19.8
   wandb login

Then enable it in your configuration:

.. code-block:: bash

   # Enable via run config
   flwr run . --run-config "use-wandb=true"

   # Or set in pyproject.toml
   [tool.flwr.app.config]
   use-wandb = true

Verifying Installation
----------------------

To verify your ICOS-FL installation:

.. code-block:: bash

   python -c "import icos_fl; print(icos_fl.__version__)"

To verify the Flower infrastructure is running:

.. code-block:: bash

   # Check Docker containers (if using Docker)
   docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

   # Or check processes (if running manually)
   ps aux | grep flower

Testing Your Installation
-------------------------

Run a simple federated learning job:

.. code-block:: bash

   # Make sure SuperLink and SuperNodes are running
   flwr run . local-deployment --stream

This should start a federated learning process using the default configuration.

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

- **Connection errors**: Ensure SuperLink is running and accessible from SuperNodes
- **Import errors**: Verify all dependencies are installed correctly
- **CUDA errors**: Check that your PyTorch installation matches your CUDA version
- **Data access issues**: Ensure the DataClay services are running and properly configured

Flower-Specific Issues
^^^^^^^^^^^^^^^^^^^^^^

- **SuperNode can't connect to SuperLink**: Check network connectivity and firewall settings
- **SuperLink not starting**: Verify ports 9091-9093 are available
- **ServerApp/ClientApp errors**: Check logs for specific error messages:

  .. code-block:: bash

     docker logs icos-fl_superlink_1
     docker logs icos-fl_supernode-1_1

For more detailed troubleshooting, see the :doc:`troubleshooting guide </how_to/troubleshooting>`.

Next Steps
----------

Once you have ICOS-FL installed, proceed to:

- :doc:`quickstart` to run your first federated learning job
- :doc:`/how_to/deployment/docker_setup` for production deployment options
- :doc:`/explanation/architecture/overview` to understand the system components
