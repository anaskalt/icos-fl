=====================
Architecture Overview
=====================

This page provides a high-level overview of the ICOS-FL architecture.

Core Components
---------------

ICOS-FL consists of several interconnected components:

.. figure:: ../../_static/images/architecture_overview.png
   :alt: ICOS-FL Architecture Overview
   :align: center

   High-level architecture of ICOS-FL

The architecture is divided into three main layers:

1. **Data Collection Layer**: Captures system metrics from nodes
2. **Storage Layer**: Persists time series data in DataClay
3. **Learning Layer**: Implements federated learning with Flower

Component Interactions
----------------------

.. code-block:: text

    +----------------------+     +----------------------+     +----------------------+
    |   Data Collection    |     |       Storage        |     |    Learning Layer    |
    |       Layer          |     |        Layer         |     |                      |
    |                      |     |                      |     |                      |
    | ┌──────────────────┐ |     | ┌──────────────────┐ |     | ┌──────────────────┐ |
    | │    Scaphandre    │ |     | │    DataClay      │ |     | │    SuperLink     │ |
    | │  Hardware Metrics│ |     | │  Distributed     │ |     | │  Central Server  │ |
    | └────────┬─────────┘ |     | │  Object Store    │ |     | └────────┬─────────┘ |
    |          │           |     | └────────┬─────────┘ |     |          │           |
    | ┌────────▼─────────┐ |     |          │           |     | ┌────────▼─────────┐ |
    | │  OpenTelemetry   │ |     | ┌────────▼─────────┐ |     | │    SuperNodes    │ |
    | │     Collector    │ |     | │  TimeSeriesData  │ |     | │   Client Nodes   │ |
    | └────────┬─────────┘ |     | │  Sliding Window  │ |     | └────────┬─────────┘ |
    |          │           |     | └────────┬─────────┘ |     |          │           |
    | ┌────────▼─────────┐ |     |          │           |     | ┌────────▼─────────┐ |
    | │    OTLP-Bridge   ├─┼─────►          │           |     | │   LSTM Models    │ |
    | │   DataClay Link  │ |     |          │           |     | │ Sequence Models  │ |
    | └──────────────────┘ |     | ┌────────▼─────────┐ |     | └────────┬─────────┘ |
    |                      |     | │    Processor     │◄┼─────┐          │           |
    |                      |     | │  Data Pipeline   │ |     │ ┌────────▼─────────┐ |
    |                      |     | └──────────────────┘ |     | │ FedAvg Strategy  │ |
    +----------------------+     +----------------------+     | │ Model Aggregation│ |
                                                              | └──────────────────┘ |
                                                              +----------------------+

The ICOS-FL framework is organized around these components working together:

Data Collection Layer
~~~~~~~~~~~~~~~~~~~~~

1. **Scaphandre**: Collects hardware metrics from the system
   - Monitors CPU, memory, and power consumption
   - Exposes metrics through a Prometheus-compatible HTTP endpoint

2. **OpenTelemetry Collector**: Processes and batches metrics
   - Scrapes metrics from Scaphandre at configurable intervals
   - Applies transformations and batching
   - Forwards metrics to the OTLP-Bridge via gRPC

3. **OTLP-Bridge**: Connects OpenTelemetry to DataClay
   - Receives metrics via gRPC
   - Transforms metrics into a format suitable for storage
   - Stores metrics in DataClay using the TimeSeriesData object

Storage Layer
~~~~~~~~~~~~~

1. **DataClay**: Distributed object store
   - Provides persistent storage for metrics and model state
   - Consists of multiple services (Redis, Metadata Service, Backend, Proxy)
   - Enables efficient data access across nodes

2. **TimeSeriesData**: Manages metric storage
   - Implements a sliding window approach for time series data
   - Maintains a configurable number of recent data points
   - Provides methods for accessing and waiting for new data

3. **Processor**: Prepares data for model training
   - Normalizes raw metrics data
   - Creates sequences for LSTM input
   - Splits data into training and validation sets
   - Generates DataLoaders for model training

Learning Layer
~~~~~~~~~~~~~~

1. **SuperLink**: Central federated learning server
   - Coordinates the federated learning process
   - Aggregates model updates from clients
   - Distributes global model to clients
   - Manages training rounds and evaluation

2. **SuperNodes**: Federated learning clients
   - Run on each node in the federation
   - Train local LSTM models on local data
   - Send model updates to the server
   - Apply global model updates

3. **LSTM Models**: Sequence prediction models
   - Predict future resource usage based on historical patterns
   - Train on local data without sharing raw metrics
   - Support custom architectures and hyperparameters

4. **FedAvg Strategy**: Federated averaging algorithm
   - Aggregates model updates from clients
   - Weights updates based on data quantity
   - Configurable parameters for client selection and participation

Cross-Cutting Concerns
----------------------

Several aspects span multiple components:

1. **Configuration Management**
   - Central configuration via pyproject.toml
   - Component-specific configuration files
   - Runtime parameter overrides

2. **Monitoring and Logging**
   - Component-level logging
   - Metrics validation via consumer.py
   - Optional integration with Weights & Biases

3. **Containerization**
   - Docker-based deployment
   - Container orchestration via Docker Compose
   - Resource management and isolation

4. **Security**
   - Data privacy through federated learning
   - Network isolation options
   - Configurable authentication

Deployment Models
-----------------

ICOS-FL supports multiple deployment models:

1. **Single-node**: All components on one machine (for development)
2. **Multi-node Federation**: Components distributed across machines
3. **Hybrid**: Mix of centralized and distributed components

The architecture is designed to be modular, allowing components to be replaced or extended as needed.
