===============
Troubleshooting
===============

This guide helps diagnose and resolve common issues with ICOS-FL.

Connection Issues
-----------------

DataClay Connection Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- ``DataClayException: Failed to connect to DataClay``
- ``RuntimeError: Failed to initialize TimeSeriesData``

**Solutions:**

1. Check if DataClay services are running:

   .. code-block:: bash

      docker compose ps

2. Verify Redis is operational:

   .. code-block:: bash

      docker compose logs redis

3. Ensure correct connection parameters:

   .. code-block:: python

      # Should match your environment
      client = Client(proxy_host="127.0.0.1", proxy_port=8676, dataset="admin")

4. Restart DataClay services:

   .. code-block:: bash

      docker compose restart redis metadata-service backend proxy

Federation Connection Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- ``UNAVAILABLE: failed to connect to all addresses``
- Clients unable to connect to server

**Solutions:**

1. Check server address configuration:

   .. code-block:: toml

      # In pyproject.toml
      [tool.flwr.federations.remote-deployment]
      address = "127.0.0.1:9093"  # Change to actual server IP

2. Verify network connectivity:

   .. code-block:: bash

      ping <server-ip>
      telnet <server-ip> 9093

3. Check firewall settings:

   .. code-block:: bash

      # On Linux
      sudo ufw status
      # On Windows
      netsh advfirewall show currentprofile

4. Ensure SuperLink is running:

   .. code-block:: bash

      docker compose -f docker/simulation.yml ps superlink

Data Collection Issues
----------------------

No Metrics Being Collected
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- ``No data available yet: DataFrame is None``
- Empty results in consumer.py

**Solutions:**

1. Check if Scaphandre is running:

   .. code-block:: bash

      docker compose logs scaphandre

2. Verify OpenTelemetry collector configuration:

   .. code-block:: bash

      # Check config
      cat otel-config.yaml

      # Check logs
      docker compose logs otel-collector

3. Ensure bridge configuration has been applied:

   .. code-block:: bash

      # Run bridge configuration
      python bridgeConfig.py

      # Check logs
      docker compose logs bridge-config

4. Restart the metrics pipeline:

   .. code-block:: bash

      docker compose restart scaphandre otel-collector bridge-config bridge

Missing Specific Metrics
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- Some columns missing from TimeSeriesData
- ``KeyError: 'expected_metric_name'``

**Solutions:**

1. Check bridge configuration:

   .. code-block:: python

      # In bridgeConfig.py, ensure metrics are added
      rc_scaphandre.add_metric("scaph_host_power_microwatts")

2. Verify metrics are available in Scaphandre:

   .. code-block:: bash

      curl http://localhost:8080/metrics | grep power_microwatts

3. Update bridge configuration and restart:

   .. code-block:: bash

      python bridgeConfig.py
      docker compose restart bridge

Training Issues
---------------

Out of Memory Errors
~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- ``RuntimeError: CUDA out of memory``
- ``MemoryError`` during training

**Solutions:**

1. Reduce batch size:

   .. code-block:: toml

      # In pyproject.toml
      batch-size = 32  # Decrease from default

2. Use smaller model:

   .. code-block:: toml

      hidden-layer-size = 10  # Decrease size
      num-layers = 1  # Reduce layers

3. Enable CPU fallback:

   .. code-block:: python

      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

4. Limit DataClay memory usage:

   .. code-block:: yaml

      # In docker-compose.yml
      environment:
        - DATACLAY_MEMORY_CHECK_INTERVAL=300  # More frequent cleanup

Model Convergence Issues
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- Steadily increasing or fluctuating loss values
- Poor prediction accuracy

**Solutions:**

1. Adjust learning rate:

   .. code-block:: toml

      learning-rate = 0.0005  # Try lower value

2. Increase local epochs:

   .. code-block:: toml

      local-epochs = 200  # More training per round

3. Normalize data properly:

   .. code-block:: python

      # Check normalization in Processor class
      # or use custom preprocessing

4. Try different model architecture:

   .. code-block:: python

      # See custom_models.rst for examples

Docker Issues
-------------

Container Startup Failures
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- Containers exit with non-zero status
- Services show as "unhealthy" or keep restarting

**Solutions:**

1. Check container logs:

   .. code-block:: bash

      docker compose logs <service-name>

2. Verify host resource availability:

   .. code-block:: bash

      # Check disk space
      df -h

      # Check memory
      free -m

3. Ensure proper dependency order:

   .. code-block:: yaml

      # In docker-compose.yml
      depends_on:
        - redis
        - metadata-service

4. Check for port conflicts:

   .. code-block:: bash

      # List used ports
      netstat -tulpn | grep LISTEN

Debugging Techniques
--------------------

Checking DataClay Objects
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dataclay import Client
   client = Client(proxy_host="127.0.0.1", dataset="admin")
   client.start()

   # List all aliases
   from dataclay.client.api import get_all_aliases
   print(get_all_aliases())

   # Get TimeSeriesData
   from icos_fl.utils.fetcher import TimeSeriesData
   tsd = TimeSeriesData.get_by_alias("timeseries")
   df = tsd.get_dataframe()
   print(df.shape if df is not None else "No data")

Enabling Verbose Logging
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Set environment variables for more detailed logs
   export LOG_LEVEL=DEBUG

   # Run with debug output
   python -m icos_fl.client.client --log-level=DEBUG

   # For Flower components
   flwr run . remote-deployment --log_level=DEBUG
