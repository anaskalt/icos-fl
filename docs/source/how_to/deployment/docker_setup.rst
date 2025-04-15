=========================
Docker Setup & Deployment
=========================

This guide explains how to deploy ICOS-FL using Docker, focusing on the Flower-based federated learning infrastructure.

Prerequisites
-------------

- Docker Engine (20.10+)
- Docker Compose (2.0+)
- 4GB+ of available RAM
- Network access between nodes (for distributed deployment)

Understanding Docker Components
-------------------------------

ICOS-FL uses Docker containers for its federated learning infrastructure:

* **SuperLink Container**: Central coordinator for federated learning
* **SuperNode Containers**: Client nodes that train models on local data

Each component is defined in the ``docker/`` directory:

* ``docker/superlink.Dockerfile``: Defines the SuperLink container
* ``docker/supernode.Dockerfile``: Defines the SuperNode container
* ``docker/simulation.yml``: Docker Compose file for local deployment

Simulation Deployment
---------------------

For development or testing, you can run all components on a single machine:

1. Deploy the SuperLink (central coordinator):

   .. code-block:: bash

      docker compose -f docker/simulation.yml up -d superlink

2. Deploy SuperNodes (client nodes):

   .. code-block:: bash

      docker compose -f docker/simulation.yml up -d supernode-1 supernode-2

3. Verify the containers are running:

   .. code-block:: bash

      docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

This setup creates a simulated federated learning environment all on your local machine using host network mode, which enables containers to access local services like DataClay through localhost.

Production Deployment
---------------------

In a production environment, you'll deploy components across multiple machines:

1. **Controller Machine Setup**:

   Deploy the SuperLink container on your controller machine:

   .. code-block:: bash

      # On controller machine
      cd /path/to/icos-fl
      docker compose -f docker/simulation.yml up -d superlink

2. **Node Machine Setup**:

   On each node machine, edit the ``docker/simulation.yml`` file to update the SuperLink address:

   .. code-block:: yaml

      # Example for supernode-1
      command:
        - --insecure
        - --superlink
        - controller.ip.address:9092  # Replace with actual controller IP
        - --clientappio-api-address
        - "0.0.0.0:9094"

   Then deploy the SuperNode:

   .. code-block:: bash

      # On each node machine
      cd /path/to/icos-fl
      docker compose -f docker/simulation.yml up -d supernode-1  # Or supernode-2, etc.

3. **Start Federated Learning**:

   The following command can be run from any machine, not just the controller. Just make sure to configure the controller's IP address in your ``pyproject.toml`` file under the ``[tool.flwr.federations.remote-deployment]`` section:

   .. code-block:: toml

      [tool.flwr.federations.remote-deployment]
      address = "controller.ip.address:9093"  # Replace with actual controller IP
      insecure = true

   Then start the federated learning process:

   .. code-block:: bash

      cd /path/to/icos-fl
      flwr run . remote-deployment --stream

Container Configuration
-----------------------

The containers are configured through the Docker Compose file and command-line parameters:

SuperLink Container
~~~~~~~~~~~~~~~~~~~

The SuperLink container accepts these parameters:

.. code-block:: yaml

   superlink:
     build:
       context: ..
       dockerfile: docker/superlink.Dockerfile
     network_mode: "host"  # Uses host network mode
     volumes:
       - ..:/app/model:rw  # To save model checkpoints
     command:
       - --insecure

Key parameters:

* ``--insecure``: Disables TLS (remove for production)
* ``network_mode: "host"``: Uses host networking for optimal local service access
* Default ports:
  * 9091: ServerAppIO API
  * 9092: Fleet API
  * 9093: Exec API

SuperNode Container
~~~~~~~~~~~~~~~~~~~

The SuperNode container accepts these parameters:

.. code-block:: yaml

   supernode-1:
     build:
       context: ..
       dockerfile: docker/supernode.Dockerfile
     network_mode: "host"
     command:
       - --insecure
       - --superlink
       - localhost:9092  # In production, use the controller's IP
       - --clientappio-api-address
       - "0.0.0.0:9094"

Key parameters:

* ``--insecure``: Disables TLS (remove for production)
* ``--superlink``: Address of the SuperLink to connect to
* ``--clientappio-api-address``: Address for ClientApp communication
* ``network_mode: "host"``: Uses host networking for optimal local service access

Building Custom Images
----------------------

The project already includes custom Docker images that extend the base Flower images with project-specific dependencies:

1. **SuperLink Image**:

   If you need to modify the SuperLink image:

   .. code-block:: bash

      docker build -f docker/superlink.Dockerfile -t custom-icos-superlink:latest .

2. **SuperNode Image**:

   If you need to modify the SuperNode image:

   .. code-block:: bash

      docker build -f docker/supernode.Dockerfile -t custom-icos-supernode:latest .

Container Resource Limits
-------------------------

For production deployments, add resource limits to your containers:

.. code-block:: yaml

   # In docker-compose.yml or simulation.yml
   superlink:
     # ... other configuration ...
     deploy:
       resources:
         limits:
           cpus: '2.0'
           memory: 2G
         reservations:
           cpus: '1.0'
           memory: 1G

   supernode-1:
     # ... other configuration ...
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

The simulation.yml file already includes volume mounts for persisting model data:

.. code-block:: yaml

   superlink:
     # ... other configuration ...
     volumes:
       - ..:/app/model:rw  # Mount the project directory for model storage

Security Considerations
-----------------------

For production deployments:

1. **Enable TLS**:

   * Remove the ``--insecure`` flag
   * Add SSL certificates and configuration:

   .. code-block:: yaml

      superlink:
        # ... other configuration ...
        command:
          - --ssl-certfile=/path/to/cert.pem
          - --ssl-keyfile=/path/to/key.pem
          - --ssl-ca-certfile=/path/to/ca.pem
        volumes:
          - /path/to/certs:/path/to:ro

2. **Network Isolation**:

   * When not using ``network_mode: "host"``, create dedicated networks
   * Restrict port exposure to the minimum necessary

3. **Authentication**:

   * Enable authentication for SuperLink and SuperNodes:

   .. code-block:: yaml

      superlink:
        # ... other configuration ...
        command:
          # ... other parameters ...
          - --auth-list-public-keys=/path/to/keys.csv
        volumes:
          - /path/to/keys.csv:/path/to/keys.csv:ro

Troubleshooting
---------------

1. **Container Fails to Start**:

   Check logs for errors:

   .. code-block:: bash

      docker logs icos-fl_superlink_1

2. **Network Connectivity Issues**:

   Ensure ports are accessible:

   .. code-block:: bash

      nc -zv localhost 9092

3. **Resource Constraints**:

   Check if containers are being killed due to OOM (Out of Memory):

   .. code-block:: bash

      docker stats

Cleanup
-------

To stop and remove all containers:

.. code-block:: bash

   docker compose -f docker/simulation.yml down
