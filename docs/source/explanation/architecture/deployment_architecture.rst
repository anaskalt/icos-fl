=======================
Deployment Architecture
=======================

This page explains the deployment architecture of ICOS-FL, focusing on how components are organized and deployed across machines.

Deployment Patterns
-------------------

ICOS-FL supports several deployment patterns:

1. **Development Mode**: All components on a single machine
2. **Federated Mode**: Components distributed across multiple machines
3. **Hybrid Mode**: Mix of centralized and distributed components

.. figure:: ../../_static/images/deployment_patterns.png
   :alt: ICOS-FL Deployment Patterns
   :align: center

   ICOS-FL deployment patterns

Single-Machine Deployment
-------------------------

In development mode, all components run on a single machine:

.. code-block:: text

    ┌─────────────────────────── Single Machine ───────────────────────────┐
    │                                                                      │
    │  ┌─────────┐  ┌───────────┐  ┌───────────┐  ┌─────────────────────┐  │
    │  │ Redis   │  │ DataClay  │  │ DataClay  │  │ DataClay Proxy      │  │
    │  └─────────┘  │ Metadata  │  │ Backend   │  └─────────────────────┘  │
    │               └───────────┘  └───────────┘                           │
    │                                                                      │
    │  ┌─────────-┐  ┌───────────┐  ┌───────────┐  ┌─────────────────────┐ │
    │  │Scaphandre│  │ OTEL      │  │ OTLP      │  │ Bridge Config       │ │
    │  │          │  │ Collector │  │ Bridge    │  └─────────────────────┘ │
    │  └──────-───┘  └───────────┘  └───────────┘                          │
    │                                                                      │
    │  ┌─────────────────────┐     ┌─────────────────────────────────────┐ │
    │  │ SuperLink (Server)  │     │ SuperNode (Client)                  │ │
    │  └─────────────────────┘     └─────────────────────────────────────┘ │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

All components are deployed as Docker containers, managed by Docker Compose.

Federated Deployment
--------------------

In a federated deployment, components are distributed across multiple machines:

.. code-block:: text

    ┌─────────────────────── Controller Machine ────────────────────────┐
    │                                                                   │
    │  ┌─────────┐  ┌───────────┐  ┌───────────┐  ┌─────────────────┐   │
    │  │ Redis   │  │ DataClay  │  │ DataClay  │  │ DataClay Proxy  │   │
    │  └─────────┘  │ Metadata  │  │ Backend   │  └─────────────────┘   │
    │               └───────────┘  └───────────┘                        │
    │                                                                   │
    │  ┌─────────-┐  ┌───────────┐  ┌───────────┐  ┌─────────────────┐  │
    │  │Scaphandre│  │ OTEL      │  │ OTLP      │  │ Bridge Config   │  │
    │  │          │  │ Collector │  │ Bridge    │  └─────────────────┘  │
    │  └────────-─┘  └───────────┘  └───────────┘                       │
    │                                                                   │
    │  ┌─────────────────────────────────────────────────────────────┐  │
    │  │ SuperLink (Server)                                          │  │
    │  └─────────────────────────────────────────────────────────────┘  │
    │                                                                   │
    └────────────────────────────────┬──────────────────────────────────┘
                                     │
                                     │ Network Communication
                                     │
    ┌────────────────────────────────┼───────────────────────────────────┐
    │                                │                                   │
    │  ┌─────────────────────────────▼───────────────────────────────┐   │
    │  │ Node Machine 1                                              │   │
    │  │                                                             │   │
    │  │  ┌─────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐   │   │
    │  │  │ Redis   │  │ DataClay  │  │ DataClay  │  │ DataClay  │   │   │
    │  │  │         │  │ Metadata  │  │ Backend   │  │ Proxy     │   │   │
    │  │  └─────────┘  └───────────┘  └───────────┘  └───────────┘   │   │
    │  │                                                             │   │
    │  │  ┌────────-─┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  │   │
    │  │  │Scaphandre│  │ OTEL      │  │ OTLP      │  │ Bridge    │  │   │
    │  │  │          │  │ Collector │  │ Bridge    │  │ Config    │  │   │
    │  │  └──────-───┘  └───────────┘  └───────────┘  └───────────┘  │   │
    │  │                                                             │   │
    │  │  ┌─────────────────────────────────────────────────────┐    │   │
    │  │  │ SuperNode (Client)                                  │    │   │
    │  │  └─────────────────────────────────────────────────────┘    │   │
    │  │                                                             │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │                                                                    │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │ Node Machine 2                                              │   │
    │  │ (Same structure as Node Machine 1)                          │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘

Key aspects of the federated deployment:

1. **Controller Machine**: Hosts the SuperLink server
2. **Node Machines**: Each runs a SuperNode client
3. **Local Data Collection**: Each node collects and processes its own metrics
4. **Federation**: SuperLink coordinates learning across nodes

Component Communication
-----------------------

ICOS-FL components communicate through several protocols:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Components
     - Protocol
     - Description
   * - Scaphandre → OTEL Collector
     - HTTP
     - Metrics scraping (Prometheus format)
   * - OTEL Collector → OTLP Bridge
     - gRPC
     - Metrics streaming (OpenTelemetry protocol)
   * - OTLP Bridge → DataClay
     - Custom Protocol
     - DataClay's internal communication
   * - SuperNode → SuperLink
     - gRPC
     - Flower's federation protocol
   * - Consumer → DataClay
     - Custom Protocol
     - DataClay client API

Docker Container Architecture
-----------------------------

Each component runs in its own Docker container:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Container
     - Responsibility
     - Dependencies
   * - redis
     - DataClay backend storage
     - None
   * - metadata-service
     - DataClay metadata management
     - redis
   * - backend
     - DataClay execution environment
     - redis
   * - proxy
     - DataClay client access point
     - metadata-service, backend
   * - scaphandre
     - Hardware metrics collection
     - Host access (privileged)
   * - otel-collector
     - Metrics processing and forwarding
     - scaphandre
   * - bridge
     - Connect OTLP to DataClay
     - proxy, otel-collector
   * - bridge-config
     - Configure bridge settings
     - proxy
   * - superlink
     - Federated learning server
     - proxy
   * - supernode-X
     - Federated learning clients
     - proxy, superlink

Network Configuration
---------------------

ICOS-FL uses several network configurations:

1. **Docker Network**: Internal communication between containers
2. **Host Network**: For components requiring direct access to host interfaces
3. **External Network**: For federation across machines

Port allocations:

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Port
     - Protocol
     - Usage
   * - 8676
     - TCP
     - DataClay Proxy
   * - 8080
     - HTTP
     - Scaphandre metrics endpoint
   * - 4317
     - gRPC
     - OpenTelemetry collector
   * - 9091
     - gRPC
     - SuperLink ServerAppIo
   * - 9092
     - gRPC
     - SuperLink Fleet (federation)
   * - 9093
     - gRPC
     - SuperLink Exec
   * - 9094+
     - gRPC
     - SuperNode ClientAppIo (one per node)

Resource Requirements
---------------------

Minimum requirements for each component:

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Component
     - CPU
     - Memory
     - Notes
   * - DataClay Stack
     - 1 core
     - 1GB
     - Scales with data volume
   * - Metrics Collection
     - 0.5 core
     - 512MB
     - Minimal overhead
   * - SuperLink
     - 1 core
     - 1GB
     - Scales with number of clients
   * - SuperNode
     - 1 core
     - 1GB
     - Scales with model complexity
   * - LSTM Training
     - 2+ cores
     - 2GB+
     - GPU beneficial but optional

Deployment Considerations
-------------------------

When deploying ICOS-FL in production, consider:

1. **Data Persistence**: Configure volume mounts for Redis and DataClay
2. **Resource Allocation**: Set appropriate resource limits for containers
3. **Network Security**: Use firewalls to restrict access to internal services
4. **Monitoring**: Implement health checks and container monitoring
5. **High Availability**: Use container orchestration for critical components
6. **Backup Strategy**: Regular backups for DataClay and model checkpoints
