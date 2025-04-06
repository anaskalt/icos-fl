============
Installation
============

This page provides detailed installation instructions for ICOS-FL and its dependencies.

Requirements
------------

ICOS-FL requires:

- Python 3.10 or newer
- Docker and Docker Compose
- NVIDIA drivers (optional, for GPU acceleration)

Python Dependencies
-------------------

ICOS-FL depends on:

- flwr>=1.17.0
- torch==2.5.1
- pandas>=2.2.3
- scikit-learn>=1.6.1
- dataclay==4.0.0
- wandb==0.19.8 (optional, for experiment tracking)

Installing from PyPI
--------------------

The recommended way to install ICOS-FL is from PyPI:

.. code-block:: bash

   pip install icos-fl

Installing from Source
----------------------

Alternatively, you can install from source:

.. code-block:: bash

   git clone https://github.com/anaskalt/icos-fl.git
   cd icos-fl
   pip install -e .

This installs ICOS-FL in development mode, allowing you to modify the code and immediately see the effects.

Docker Setup
------------

The full infrastructure can be deployed using Docker Compose:

1. Ensure Docker and Docker Compose are installed
2. Clone the repository if you haven't already
3. Start the services:

   .. code-block:: bash

      docker compose up -d

4. Verify the services are running:

   .. code-block:: bash

      docker compose ps

Optional Components
-------------------

**GPU Support**

To enable GPU acceleration for the LSTM models:

1. Install NVIDIA drivers and CUDA toolkit
2. Install PyTorch with CUDA support:

   .. code-block:: bash

      pip install torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu118

**Weights & Biases Integration**

For experiment tracking with Weights & Biases:

.. code-block:: bash

   pip install wandb==0.19.8
   wandb login

Then enable it in your configuration:

.. code-block:: bash

   # Enable in pyproject.toml
   use-wandb = true

   # Or set environment variable
   export ICOS_FL_USE_WANDB=1

Verifying Installation
----------------------

To verify your installation:

.. code-block:: bash

   python -c "import icos_fl; print(icos_fl.__version__)"

Troubleshooting
---------------

**Common Issues**

- **DataClay connection errors**: Check that Redis is running properly
- **Import errors**: Ensure all dependencies are installed
- **CUDA errors**: Verify your PyTorch installation matches your CUDA version

For more detailed troubleshooting, see the :doc:`troubleshooting guide </how_to/troubleshooting>`.
