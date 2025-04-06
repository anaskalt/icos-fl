==================
Metrics Collection
==================

This guide explains how metrics are collected and processed in ICOS-FL.

Metrics Collection Pipeline
---------------------------

ICOS-FL uses a pipeline to collect, process, and store system metrics:

1. **Scaphandre**: Collects hardware metrics from the host system
2. **OpenTelemetry Collector**: Scrapes and processes metrics
3. **OTLP-Bridge**: Receives batched metrics and converts them for storage
4. **DataClay**: Stores processed metrics as time series data

.. figure:: ../../_static/images/metrics_flow.png
   :alt: Metrics Collection Pipeline
   :align: center

   Metrics Collection Pipeline

Collected Metrics
-----------------

By default, ICOS-FL collects these system metrics:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Metric
     - Description
   * - scaph_host_power_microwatts
     - Power consumption in microwatts
   * - scaph_host_load_avg_one
     - 1-minute CPU load average
   * - scaph_host_memory_total_bytes
     - Total system memory in bytes
   * - scaph_host_memory_available_bytes
     - Available system memory in bytes

These metrics are transformed into more user-friendly values:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Source Metric
     - Transformed Metric
     - Transformation
   * - scaph_host_power_microwatts
     - power_consumption
     - Converted to watts (divided by 1,000,000)
   * - scaph_host_load_avg_one
     - cpu_usage
     - Used directly
   * - memory_total - memory_available
     - memory_usage
     - Converted to MB (divided by 1,024*1,024)

Configuring Metrics Collection
------------------------------

Customize which metrics are collected by modifying the Bridge Configuration:

.. code-block:: python

   # In bridgeConfig.py
   rc_scaphandre = ResourceConfiguration("scaphandre-metrics", scaphandre_rules)

   # Add or remove metrics
   rc_scaphandre.add_metric("scaph_host_power_microwatts")
   rc_scaphandre.add_metric("scaph_host_load_avg_one")
   rc_scaphandre.add_metric("scaph_host_memory_total_bytes")
   rc_scaphandre.add_metric("scaph_host_memory_available_bytes")

   # Add new metric
   rc_scaphandre.add_metric("scaph_host_memory_cached_bytes")

OpenTelemetry Configuration
---------------------------

Adjust the OpenTelemetry scraping interval and batch settings in `otel-config.yaml`:

.. code-block:: yaml

   receivers:
     prometheus:
       config:
         scrape_configs:
           - job_name: 'scaphandre'
             scrape_interval: 3s  # Adjust collection frequency
             static_configs:
               - targets: ['127.0.0.1:8080']

   processors:
     batch:
       timeout: 180s  # Adjust batching interval

Time Series Data Storage
------------------------

Metrics are stored in a sliding window in DataClay:

.. code-block:: python

   # In icos_fl/utils/fetcher.py
   class TimeSeriesData(DataClayObject):
       """Class for managing time series data with a sliding window approach."""

       def __init__(self, max_rows: int = 300) -> None:
           self.dataframe = None
           self.max_rows = max_rows
           self.waiters = list()

The default configuration maintains 300 most recent data points (approximately 15 minutes with 3-second intervals).

Accessing Collected Metrics
---------------------------

To access the collected metrics programmatically:

.. code-block:: python

   from dataclay import Client
   from icos_fl.utils.fetcher import TimeSeriesData

   # Connect to DataClay
   client = Client(proxy_host="127.0.0.1", dataset="admin")
   client.start()

   # Get TimeSeriesData
   tsd = TimeSeriesData.get_by_alias("timeseries")

   # Get current dataframe
   df = tsd.get_dataframe()

   # Display metrics
   print(df.head())

You can also use the included consumer.py script to monitor metrics:

.. code-block:: bash

   python consumer.py

Adding Custom Metrics
---------------------

To collect additional metrics:

1. Ensure the metrics are exposed by Scaphandre or another OpenTelemetry source
2. Update the OpenTelemetry configuration to scrape these metrics
3. Modify the Bridge Configuration to collect the new metrics:

   .. code-block:: python

      rc_custom = ResourceConfiguration("custom-metrics", custom_rules)
      rc_custom.add_metric("my_custom_metric_name")
      bc.set_res_config(rc_custom)

4. Update your data processing code to handle the new metrics

Data Preprocessing
------------------

When metrics are used for model training, they go through these preprocessing steps:

1. **Normalization**: Standardized to zero mean and unit variance
2. **Sequencing**: Converted to sequences of length `time_step`
3. **Train/Test Split**: Divided based on the `train_ratio` configuration
4. **Batching**: Grouped into batches of size `batch_size`

This preprocessing is handled by the Processor class:

.. code-block:: python

   from icos_fl.utils.processor import Processor

   processor = Processor(
       time_step=10,
       metric="cpu_usage",
       batch_size=64,
       train_ratio=0.8,
       device=torch.device("cpu")
   )

   train_dataloader, val_dataloader, _, _ = processor.create_data_loaders(df)
