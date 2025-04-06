==========================
Frequently Asked Questions
==========================

General Questions
-----------------

What is ICOS-FL?
~~~~~~~~~~~~~~~~

ICOS-FL is a federated learning framework built on Flower for resource monitoring and prediction across distributed nodes. It uses LSTM models to predict CPU, memory, and power usage patterns.

How does ICOS-FL compare to other federated learning frameworks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ICOS-FL is specifically optimized for time series prediction of system metrics. While general-purpose frameworks like TensorFlow Federated or PySyft offer broader ML capabilities, ICOS-FL provides a streamlined solution for resource monitoring with built-in DataClay integration.

Does ICOS-FL require specialized hardware?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No. ICOS-FL runs on standard hardware but can utilize GPUs if available for faster training. All components can run on CPU-only machines with reasonable performance.

Technical Questions
-------------------

How is data stored in ICOS-FL?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ICOS-FL uses DataClay as its persistent storage layer. Time series data is stored in a sliding window format, with the TimeSeriesData class handling automatic management of the buffer size.

What metrics does ICOS-FL collect?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, ICOS-FL collects CPU usage, memory usage, and power consumption through Scaphandre and OpenTelemetry. You can customize the metrics collection by modifying the BridgeConfiguration.

Can I deploy ICOS-FL without Docker?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, all components can be installed and run natively. However, Docker simplifies deployment and ensures consistent behavior across different environments.

Federated Learning Questions
----------------------------

How many clients does ICOS-FL support?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ICOS-FL has been tested with up to 20 client nodes. The practical limit depends on network bandwidth, server resources, and aggregation strategy.

How does ICOS-FL handle client failures?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ICOS-FL uses Flower's built-in fault tolerance mechanisms. If a client becomes unavailable during training, the server can continue with the remaining clients as long as the minimum number of available clients is satisfied.

Can I use my own custom models?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, ICOS-FL supports custom models as long as they implement the required interface. See :doc:`/how_to/models/custom_models` for details.

Can ICOS-FL work with non-time-series data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While ICOS-FL is optimized for time series forecasting, its architecture can be adapted for other federated learning tasks. However, you would need to modify the data processing pipeline.

Performance and Scaling
-----------------------

How much data does ICOS-FL need for accurate predictions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For reasonable forecasting accuracy, ICOS-FL typically requires at least a few hours of historical data. Prediction accuracy improves with more data and training rounds.

What is the network overhead of federated learning?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The network overhead is proportional to model size rather than data size. LSTM models in ICOS-FL are relatively compact (typically <1MB), making the communication efficient.

How far ahead can ICOS-FL predict resource usage?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, ICOS-FL is configured to predict 5 minutes ahead based on 15 minutes of historical data. This prediction horizon can be adjusted by modifying the time window configuration.
