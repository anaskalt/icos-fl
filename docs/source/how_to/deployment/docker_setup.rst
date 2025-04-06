=========================
Docker Setup & Deployment
=========================

This guide explains how to deploy ICOS-FL using Docker.

Basic Deployment
----------------

ICOS-FL provides a complete Docker Compose configuration that sets up all required components.

Prerequisites
~~~~~~~~~~~~~

- Docker Engine (20.10+)
- Docker Compose (2.0+)
- 4GB+ of available RAM
- Network access between nodes (for federated deployment)

Single-Machine Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~

For development or testing, deploy all components on a single machine:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/anaskalt/icos-fl.git
      cd icos-fl

2. Start all services:

   .. code-block:: bash

      docker compose up -d

   This command starts:

   - Redis for DataClay backend
   - DataClay services (Metadata Service, Backend, Proxy)
   - Scaphandre for metrics collection
   - OpenTelemetry collector
   - OTLP-DataClay Bridge
   - Bridge Configuration service

3. Verify the services are running:

   .. code-block:: bash

      docker compose ps

   All services should show as "Up" status.

Component Configuration
-----------------------

Each component can be configured through environment variables or by modifying the docker-compose.yml file.

DataClay Configuration
~~~~~~~~~~~~~~~~~~~~~~

DataClay services can be configured through environment variables:

.. code-block:: yaml

   environment:
     - DATACLAY_MEMORY_CHECK_INTERVAL=600  # Memory cleanup interval in seconds
     - DATACLAY_KV_HOST=redis              # Redis hostname

OTLP Collector Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The OpenTelemetry Collector is configured via the otel-config.yaml file:

.. code-block:: yaml

   # Adjust scrape interval (currently 3s)
   scrape_interval: 3s

   # Adjust batch timeout (currently 180s)
   timeout: 180s

Bridge Configuration
~~~~~~~~~~~~~~~~~~~~

The Bridge configuration is handled by the bridge-config container, which you can customize:

.. code-block:: yaml

   environment:
     - DATACLAY_PROXY_HOST=proxy
     - DATACLAY_PROXY_PORT=8676
     - BRIDGE_CONFIG_ALIAS=bridge_config  # Optional: change alias

Custom Docker Images
--------------------

You can build custom Docker images for each component:

.. code-block:: bash

   # Build all images
   docker compose build

   # Build specific service
   docker compose build bridge

Container Resource Limits
-------------------------

For production deployments, consider setting resource limits:

.. code-block:: yaml

   services:
     backend:
       deploy:
         resources:
           limits:
             cpus: '1.0'
             memory: 1G
           reservations:
             cpus: '0.5'
             memory: 512M

Persisting Data
---------------

To persist data between container restarts:

.. code-block:: yaml

   volumes:
     - ./dataclay/storage:/dataclay/storage:rw
     - ./dataclay/metadata:/dataclay/metadata:rw

Security Considerations
-----------------------

For production deployments:

1. Use network isolation with Docker networks
2. Set up proper firewall rules for exposed ports
3. Use environment files instead of hardcoding secrets
4. Consider using Docker Secrets for sensitive information

Cleanup
-------

To stop and remove all containers:

.. code-block:: bash

   docker compose down

To also remove volumes and networks:

.. code-block:: bash

   docker compose down -v
