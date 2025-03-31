Data Flow Pipeline
==================

::

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
    +──────────────────────────────────────────────┼────────────────+
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
    +───────────────────────────────-────────────────┼───────────────+
                                                     │ Used for
                                                     ▼
    +───────────────────────────────────────────────────────────────+
    │                  LSTM Model Training                          │
    │                                                               │
    │ • Time step = 10: Each sequence covers ~30 seconds (10×3s)    │
    │ • Predicts next 5 minutes ahead (100 data points)             │
    │ • Utilizes 15-min historical data (300 points) for training   │
    └───────────────────────────────────────────────────────────────+

Summary of Key Intervals
------------------------

- **Scaphandre metrics minimum refresh:** every 2 seconds.
- **Prometheus scraping interval:** every 3 seconds.
- **Batch processor interval:** every 180 seconds.
- **Sliding window history:** maintains ~15 minutes (300 rows).
- **LSTM prediction:** sequences of 30 seconds (10 data points), predicts 5 minutes ahead.
