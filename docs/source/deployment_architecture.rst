ICOS-FL Deployment Architecture
===============================

::

                    +──────────────────────────────────────+
                    │           ICOS Controller            │
                    │   (Central Coordinator for ICOS-FL)  │
                    +────────────────────┬─────────────────+
                                         │ Hosts SuperLink and DataClay
                                         │ Coordinates federated training
                                         ▼
    +──────────────────────────────────────────────────────────────────────+
    │                     Controller Docker Setup                          │
    │                                                                      │
    │  ┌────────────────────────────┐    ┌────────────────────────────┐    │
    │  │ OTLP-DataClay Bridge       │    │ SuperLink Container        │    │
    │  │ (Data collection pipeline) │    │ (FL coordination)          │    │
    │  │                            │    │                            │    │
    │  │  ┌──────────┐ ┌──────────┐ │    │  ┌─────────────────────┐   │    │
    │  │  │Scaphandre│ │ OTLP     │ │    │  │ SuperLink Service   │   │    │
    │  │  │(metrics) │ │Collector │ │    │  │                     │   │    │
    │  │  └────┬─────┘ └────┬─────┘ │    │  │ • Connects nodes    │   │    │
    │  │       │            │       │    │  │ • Aggregates models │   │    │
    │  │  ┌────▼────────────▼─────┐ │    │  │ • Coordinates rounds│   │    │
    │  │  │ Bridge + DataClay     │ │    │  └─────────┬───────────┘   │    │
    │  │  │ (Stores time series)  │ │    │            │               │    │
    │  │  └───────────┬───────────┘ │    │  ┌─────────▼─────────-───┐ │    │
    │  └──────────────┼─────────────┘    │  │ ServerApp (subprocess)│ │    │
    │                 │                  │  │                       │ │    │
    │                 │                  │  │ • FedAvg Strategy     │ │    │
    │                 │                  │  │ • Model checkpointing │ │    │
    │                 │                  │  │ • Evaluation          │ │    │
    │                 │                  │  └───────────────────────┘ │    │
    │                 │                  │                            │    │
    │                 └──────────────────┼────────────────────────────┘    │
    │                                    │                                 │
    │         Host Network (localhost:8676)                                │
    │                                                                      │
    └────────────────────────────────────┼─────────────────────────────────┘
                                         │
                                         │ Network Communication
                                         │ Controller.IP:9092
                                         ▼
    +──────────────────────────────────────────────────────────────────────+
    │                        ICOS Node Machine 1                           │
    │                                                                      │
    │  ┌────────────────────────────┐    ┌────────────────────────────┐    │
    │  │ OTLP-DataClay Bridge       │    │ SuperNode Container        │    │
    │  │                            │    │                            │    │
    │  │  ┌──────────┐ ┌──────────┐ │    │  ┌─────────────────────┐   │    │
    │  │  │Scaphandre│ │ OTLP     │ │    │  │ SuperNode Service   │   │    │
    │  │  │(metrics) │ │Collector │ │    │  │                     │   │    │
    │  │  └────┬─────┘ └────┬─────┘ │    │  │ • Connects to       │   │    │
    │  │       │            │       │    │  │   SuperLink         │   │    │
    │  │  ┌────▼────────────▼─────┐ │    │  │ • Manages local     │   │    │
    │  │  │ Bridge + DataClay     │ │    │  │   training          │   │    │
    │  │  │ (Local time series)   │ │    │  └─────────┬───────────┘   │    │
    │  │  └───────────┬───────────┘ │    │            │               │    │
    │  └──────────────┼─────────────┘    │    ┌───────▼─────────────┐ │    │
    │                 │                  │    │ ClientApp           │ │    │
    │                 │                  │    │ (subprocess)        │ │    │
    │                 │                  │    │                     │ │    │
    │                 │                  │    │ • LSTM model        │ │    │
    │                 │                  │    │ • Local training    │ │    │
    │                 │                  │    └─────────────────────┘ │    │
    │                 └──────────────────┼────────────────────────────┘    │
    │                                    │                                 │
    │         Host Network (localhost:8676)                                │
    │                                                                      │
    └────────────────────────────────────┼─────────────────────────────────┘
                                         │
                                         │ Additional nodes follow same pattern
                                         ▼
    +──────────────────────────────────────────────────────────────────────+
    │                        ICOS Node Machine 2                           │
    │                                                                      │
    │  ┌────────────────────────────┐    ┌────────────────────────────┐    │
    │  │ OTLP-DataClay Bridge       │    │ SuperNode Container        │    │
    │  │                            │    │                            │    │
    │  │  ┌──────────┐ ┌──────────┐ │    │  ┌─────────────────────┐   │    │
    │  │  │Scaphandre│ │ OTLP     │ │    │  │ SuperNode Service   │   │    │
    │  │  │(metrics) │ │Collector │ │    │  │                     │   │    │
    │  │  └────┬─────┘ └────┬─────┘ │    │  │ • Connects to       │   │    │
    │  │       │            │       │    │  │   SuperLink         │   │    │
    │  │  ┌────▼────────────▼─────┐ │    │  │ • Manages local     │   │    │
    │  │  │ Bridge + DataClay     │ │    │  │   training          │   │    │
    │  │  │ (Local time series)   │ │    │  └─────────┬───────────┘   │    │
    │  │  └───────────┬───────────┘ │    │            │               │    │
    │  └──────────────┼─────────────┘    │    ┌───────▼─────────────┐ │    │
    │                 │                  │    │ ClientApp           │ │    │
    │                 │                  │    │ (subprocess)        │ │    │
    │                 │                  │    │                     │ │    │
    │                 │                  │    │ • LSTM model        │ │    │
    │                 │                  │    │ • Local training    │ │    │
    │                 │                  │    └─────────────────────┘ │    │
    │                 └──────────────────┼────────────────────────────┘    │
    │                                    │                                 │
    │         Host Network (localhost:8676)                                │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘


    +─────────-──────────────── DEPLOYMENT STEPS ─────────────────────────────+
    │                                                                         │
    │  1. Deploy otlp-dataclay-bridge on all machines (controller and nodes)  │
    │     $ cd /path/to/otlp-dataclay-bridge                                  │
    │     $ docker compose up -d                                              │
    │                                                                         │
    │  2. Deploy SuperLink on controller machine                              │
    │     $ cd /path/to/icos-fl                                               │
    │     $ docker compose -f docker/simulation.yml up -d superlink           │
    │                                                                         │
    │  3. Deploy SuperNodes on each node machine                              │
    │     $ cd /path/to/icos-fl                                               │
    │     $ docker compose -f docker/simulation.yml up -d supernode-X         │
    │     (replace X with node number and update --superlink to Controller IP)│
    │                                                                         │
    │  4. Update federation address in pyproject.toml on the controller       │
    │     [tool.flwr.federations.remote-deployment]                           │
    │     address = "127.0.0.1:9093"                                          │
    │     insecure = true                                                     │
    │                                                                         │
    │  5. Run federated learning                                              │
    │     $ cd /path/to/icos-fl                                               │
    │     $ flwr run . remote-deployment --stream                             │
    │                                                                         │
    └───────────────────────────────────────────────────-─────────────────────┘


Components
----------

Controller Machine
~~~~~~~~~~~~~~~~~~

- **SuperLink Container**: Core coordination component
    - *ServerApp subprocess*: Runs federated learning algorithms
    - Exposes ports 9091 (ServerAppIo), 9092 (Fleet), 9093 (Exec)
    - Uses host networking to access local DataClay services
    - Mounts volume for model checkpoint storage

- **OTLP-DataClay Bridge**: Data collection and storage
    - *Scaphandre*: Hardware monitoring service
    - *OTLP Collector*: Batches and forwards metrics
    - *DataClay Services*: Stores time-series metrics data
    - Maintains DataFrame with system metrics (CPU, RAM, power)

Node Machines
~~~~~~~~~~~~~

- **SuperNode Container**: Local training component
    - *ClientApp subprocess*: Runs local model training
    - Connects to SuperLink via Controller IP:9092
    - Exposes ClientAppIo port (9094, 9095, etc.)
    - Uses host networking to access local DataClay services

- **OTLP-DataClay Bridge**: Per-node monitoring
    - Identical setup to Controller's DataClay system
    - Collects node-specific hardware metrics
    - Stores up to 15 minutes of local time-series data
    - Serves data to ClientApp for local training

Communication Channels
----------------------

- **Internal Communication**: Uses host network mode
  - SuperLink/SuperNode -> DataClay: localhost:8676
  - ClientApp/ServerApp -> DataClay: localhost:8676

- **External Communication**: Over network
  - SuperNodes -> SuperLink: Controller.IP:9092
  - CLI -> SuperLink: Controller.IP:9093

Summary of Key Components
-------------------------

- **SuperLink**: Orchestrates federated learning process
- **SuperNodes**: Manage local model training on each node
- **DataClay**: Stores time-series data for training
- **ClientApp**: Trains LSTM models on local data
- **ServerApp**: Aggregates models and coordinates training rounds
