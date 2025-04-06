=================
Dataflow Pipeline
=================

This page explains the data flow through ICOS-FL, from metrics collection to model training.

Overview
--------

ICOS-FL's data pipeline processes system metrics through several stages:

.. code-block:: text

    +───────────────────+
    │    Scaphandre     │
    │  (Metrics Source) │
    +──────────┬────────+
               │ Exposes metrics (HTTP server :8080)
               │ Note: Metrics refresh interval ≥ 2 seconds
               ▼
    +───────────────────────────────────────────────────────────────+
    │                 OpenTelemetry Collector                       │
    │                                                               │
    │  ┌───────────────────┐           ┌───────────────────┐        │
    │  │ Prometheus        │ Scrapes   │ Batch Processor   │        │
    │  │ Receiver          ├──────────►│                   │        │
    │  │                   │ every 3s  │ (Batches Metrics) │        │
    │  └───────────────────┘           │ Interval: 180s    │        │
    │                                  └───────────┬───────┘        │
    │                                              │ batches metrics│
    │                                              ▼                │
    │                                  ┌───────────────────┐        │
    │                                  │ OTLP Exporter     │        │
    │                                  │ (gRPC :4317)      │        │
    │                                  └───────────┬───────┘        │
    +──────────────────────────────────────────────┼────────────────+
                                                   │ Sends metrics
                                                   ▼
    +───────────────────────────────────────────────────────────────+
    │                  OTLP–DataClay Bridge                         │
    │                                                               │
    │  ┌───────────────────┐           ┌───────────────────┐        │
    │  │ MetricsService    │ Processes │ BridgeConfig      │        │
    │  │ (gRPC Server)     ├──────────►│ (Configured via   │        │
    │  └───────────────────┘ metrics   │   dataClay)       │        │
    │                                  └───────────┬───────┘        │
    │                                              │ Stores metrics │
    │                                              ▼                │
    │                                  ┌───────────────────┐        │
    │                                  │ TimeSeriesData    │        │
    │                                  │   (dataClay)      │        │
    │                                  └───────────┬───────┘        │
    +──────────────────────────────────────────────┼────────────────+
                                                   │ Updates
                                                   ▼
    +───────────────────────────────────────────────────────────────+
    │                TimeSeriesData (dataClay)                      │
    │                                                               │
    │ ┌───────────────────────────────────────────────────────────┐ │
    │ │ Unified DataFrame (Sliding Window: Max 300 rows)          │ │
    │ │                                                           │ │
    │ │  ┌─────┬─────────────────┬───────────┬───────────┐        │ │
    │ │  │time │power_consumption│cpu_usage  │ram_usage  │        │ │
    │ │  ├─────┼─────────────────┼───────────┼───────────┤        │ │
    │ │  │ ... │        ...      │    ...    │    ...    │        │ │
    │ │  └─────┴─────────────────┴───────────┴───────────┘        │ │
    │ │                                                           │ │
    │ │ • FIFO (oldest entries removed first)                     │ │
    │ │ • Maintains approximately 15 minutes of historical data   │ │
    │ │ • Each batch introduces about 60 new data points          │ │
    │ └───────────────────────────────────────────────────────────┘ │
    +──────────────────────────────────────────────┬────────────────+
                                                   │ Fetches data
                                                   ▼
    +───────────────────────────────────────────────────────────────+
    │                          Fetcher                              │
    │                                                               │
    │ ┌───────────────────────────┐       ┌───────────────────────┐ │
    │ │ get_dataframe()           │──────►│ process_dataframe()   │ │
    │ │ • Retrieves unified data  │       │ • Prepares data for   │ │
    │ │ • All 300 data points     │       │   LSTM modeling       │ │
    │ └───────────────────────────┘       └───────────┬───────────┘ │
    │                                                 │             │
    │                                                 ▼             │
    │                                    ┌───────────────────────┐  │
    │                                    │ LSTM-ready DataFrame  │  │
    │                                    │ • Time step = 10      │  │
    │                                    │   (10 points per seq) │  │
    │                                    └───────────┬───────────┘  │
    +───────────────────────────────-────────────────┼──────────────+
                                                     │ Used for
                                                     ▼
    +───────────────────────────────────────────────────────────────+
    │                  LSTM Model Training                          │
    │                                                               │
    │ • Time step = 10: Each sequence covers ~30 seconds (10×3s)    │
    │ • Predicts next 5 minutes ahead (100 data points)             │
    │ • Utilizes 15-min historical data (300 points) for training   │
    └───────────────────────────────────────────────────────────────+

Stage 1: Metrics Collection
---------------------------

Scaphandre
~~~~~~~~~~

Scaphandre is responsible for collecting hardware metrics:

1. Runs as a privileged container with access to system stats
2. Collects CPU, memory, and power consumption metrics
3. Exposes metrics via an HTTP endpoint (port 8080)
4. Updates metrics at minimum 2-second intervals

Important metrics collected:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Metric Name
     - Description
   * - scaph_host_power_microwatts
     - Power consumption in microwatts
   * - scaph_host_load_avg_one
     - 1-minute CPU load average
   * - scaph_host_memory_total_bytes
     - Total system memory in bytes
   * - scaph_host_memory_available_bytes
     - Available system memory in bytes

These metrics are exposed in Prometheus format:

.. code-block:: text

   # HELP scaph_host_power_microwatts Power consumption in microwatts
   # TYPE scaph_host_power_microwatts gauge
   scaph_host_power_microwatts{sensor="package-0"} 52428800

Stage 2: Metrics Processing
---------------------------

OpenTelemetry Collector
~~~~~~~~~~~~~~~~~~~~~~~

The OpenTelemetry Collector processes the raw metrics:

1. **Prometheus Receiver**: Scrapes metrics from Scaphandre every 3 seconds
2. **Batch Processor**: Collects metrics for 180 seconds (3 minutes)
3. **OTLP Exporter**: Sends batched metrics via gRPC to the OTLP-Bridge

Configuration in otel-config.yaml:

.. code-block:: yaml

   receivers:
     prometheus:
       config:
         scrape_configs:
           - job_name: 'scaphandre'
             scrape_interval: 3s
             static_configs:
               - targets: ['127.0.0.1:8080']

   processors:
     batch:
       timeout: 180s

   exporters:
     otlp:
       endpoint: 127.0.0.1:4317
       tls:
         insecure: true

   service:
     pipelines:
       metrics:
         receivers: [prometheus]
         processors: [batch]
         exporters: [otlp]

This configuration results in:
- Approximately 60 data points per batch (180s ÷ 3s)
- One batch sent to the OTLP-Bridge every 3 minutes

Stage 3: Bridge Layer
---------------------

OTLP-Bridge
~~~~~~~~~~~

The OTLP-Bridge connects OpenTelemetry to DataClay:

1. Implements a gRPC server listening on port 4317
2. Receives batched metrics from the OpenTelemetry Collector
3. Filters metrics based on the BridgeConfiguration
4. Transforms metrics into a DataFrame format
5. Stores the DataFrame in DataClay via TimeSeriesData

Bridge Configuration
~~~~~~~~~~~~~~~~~~~~

The bridge is configured through the BridgeConfiguration class:

.. code-block:: python

   # From bridgeConfig.py
   rc_scaphandre = ResourceConfiguration("scaphandre-metrics", scaphandre_rules)

   # Add specific metrics to track
   rc_scaphandre.add_metric("scaph_host_power_microwatts")
   rc_scaphandre.add_metric("scaph_host_load_avg_one")
   rc_scaphandre.add_metric("scaph_host_memory_total_bytes")
   rc_scaphandre.add_metric("scaph_host_memory_available_bytes")

   # Add configuration to bridge
   bc.set_res_config(rc_scaphandre)

Stage 4: Data Storage
---------------------

TimeSeriesData
~~~~~~~~~~~~~~

The TimeSeriesData class manages the storage of time series data:

1. Maintains a sliding window of metrics data (default: 300 rows)
2. Implements first-in-first-out (FIFO) behavior for the window
3. Stores data as a pandas DataFrame in DataClay
4. Provides methods for accessing and waiting for new data

Implementation:

.. code-block:: python

   # From icos_fl/utils/fetcher.py
   class TimeSeriesData(DataClayObject):
       """Class for managing time series data with a sliding window approach."""

       def __init__(self, max_rows: int = 300) -> None:
           self.dataframe = None
           self.max_rows = max_rows
           self.waiters = list()

       @activemethod
       def add_dataframe(self, df: pd.DataFrame) -> None:
           """Add new data to the unified dataframe, maintaining the sliding window."""
           if self.dataframe is None:
               self.dataframe = df
           else:
               # Append new data
               self.dataframe = pd.concat([self.dataframe, df])

               # Maintain sliding window by removing oldest entries
               if len(self.dataframe) > self.max_rows:
                   self.dataframe = self.dataframe.iloc[-self.max_rows:]

At default settings (300 rows with 3-second intervals), this maintains approximately 15 minutes of historical data.

DataFrame Format
~~~~~~~~~~~~~~~~

The DataFrame has this structure:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Column
     - Description
   * - timestamp
     - Unix timestamp in nanoseconds (index)
   * - scaphandre-metrics.scaph_host_power_microwatts
     - Power consumption
   * - scaphandre-metrics.scaph_host_load_avg_one
     - CPU load average
   * - scaphandre-metrics.scaph_host_memory_total_bytes
     - Total memory
   * - scaphandre-metrics.scaph_host_memory_available_bytes
     - Available memory

Stage 5: Data Preprocessing
---------------------------

Fetcher
~~~~~~~

The Fetcher class retrieves and processes the raw data:

1. Connects to DataClay and retrieves TimeSeriesData
2. Processes raw metrics into a standardized format
3. Converts metrics to appropriate units (watts, MB)
4. Creates derived metrics (e.g., memory_usage from total and available)

The post-processing creates a clean DataFrame with these columns:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Column
     - Description
   * - timestamp
     - Unix timestamp in seconds
   * - power_consumption
     - Power in watts (microwatts ÷ 1,000,000)
   * - cpu_usage
     - CPU load average (unchanged)
   * - memory_usage
     - Memory usage in MB ((total - available) ÷ (1024 * 1024))

Processor
~~~~~~~~~

The Processor class prepares data for LSTM training:

1. Normalizes data using StandardScaler (zero mean, unit variance)
2. Creates sequences of length time_step for LSTM input
3. Splits data into training and validation sets
4. Creates DataLoaders with appropriate batch sizes

Implementation details:

.. code-block:: python

   # From icos_fl/utils/processor.py
   class Processor:
       def __init__(
           self,
           time_step: int,
           metric: str,
           batch_size: int = 64,
           train_ratio: float = 0.8,
           device: Optional[torch.device] = None,
       ) -> None:
           # ...

       def create_data_loaders(
           self,
           df: pd.DataFrame,
           # ...
       ) -> Tuple[DataLoader, DataLoader, TimeSeriesDataset, TimeSeriesDataset]:
           # ...

Stage 6: Model Training
-----------------------

LSTM Model
~~~~~~~~~~

The LSTM model is trained on the preprocessed data:

1. Receives batches of sequences from the DataLoader
2. Predicts the next value in the sequence
3. Updates weights based on prediction error
4. Returns updated weights to the federated learning system

Training parameters:

- **Sequence Length**: 10 time steps (~30 seconds with 3s scrape interval)
- **Prediction Horizon**: Next value (can be extended to multiple steps)
- **Batch Size**: Typically 64 sequences
- **Local Epochs**: Usually 50-100 per federated round

The model is trained to minimize Mean Squared Error (MSE) between predicted and actual values.

Federated Learning
~~~~~~~~~~~~~~~~~~

In the federated learning setup:

1. Each node trains on its local data
2. Model updates are sent to the central server
3. The server aggregates updates using FedAvg
4. Updated global model is distributed back to nodes
5. Process repeats for multiple rounds

Complete Data Lifecycle
-----------------------

To summarize the complete lifecycle:

1. **Collection**: Scaphandre collects metrics every 2+ seconds
2. **Processing**: OpenTelemetry scrapes every 3s and batches for 180s
3. **Storage**: OTLP-Bridge stores batches in TimeSeriesData (sliding window of 300 rows)
4. **Preprocessing**: Fetcher and Processor prepare data for model training
5. **Training**: LSTM models train on local data in a federated learning setup
6. **Prediction**: Trained models predict future resource usage

Key Intervals
~~~~~~~~~~~~~

- **Metrics collection**: Every 3 seconds
- **Metrics batching**: Every 180 seconds (3 minutes)
- **Sliding window**: 300 data points (~15 minutes)
- **LSTM sequences**: 10 time steps (~30 seconds)
- **Federated rounds**: Variable (typically minutes to hours)
