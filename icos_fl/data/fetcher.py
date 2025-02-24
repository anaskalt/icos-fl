"""Real-time metrics collection for ICOS-FL components.

This module handles the collection and processing of real-time metrics from
DataClay storage. It implements buffering and validation of time series data
for machine learning purposes.

Key Components:
   - MetricsMapping: Handles different types of metrics processing
   - DataFetcher: Main class for data collection and management
"""

from collections import deque
from threading import Event
from typing import Optional, List
import pandas as pd
from dataclay import Client, DataClayObject, activemethod
from dataclay.exceptions import DataClayException
from icos_fl.utils.logger import ICOSLogger as LOGGER

LOG = LOGGER("DataFetcher")

class TimeSeriesData(DataClayObject):
   """Persistent object for storing time series data.
   
   This class manages the storage and access of time series data in DataClay.
   It implements a circular buffer for DataFrames and provides methods for
   adding new data and waiting for updates.
   
   Attributes:
       columns: List of column names
       dataframes: Circular buffer of DataFrames
       waiters: List of waiting events
   """
   
   columns: list[str]
   dataframes: deque[pd.DataFrame]
   waiters: list[Event]

   def __init__(self):
       """Initialize empty time series storage."""
       self.dataframes = deque(maxlen=5)  # Keep last 5 DataFrames
       self.waiters = list()

   @activemethod
   def add_dataframe(self, df: pd.DataFrame) -> None:
       """Add a DataFrame to the circular buffer.
       
       Args:
           df: DataFrame to add
           
       Note:
           This method also notifies any waiting consumers.
       """
       self.dataframes.append(df)
       for waiter in self.waiters:
           waiter.set()
   
   @activemethod
   def get_last_dataframe(self) -> Optional[pd.DataFrame]:
       """Get the last DataFrame in the buffer.
       
       Returns:
           Most recent DataFrame or None if buffer is empty
       """
       if self.dataframes:
           return self.dataframes[-1]
       return None

   @activemethod
   def wait_for_dataframe(self) -> Optional[pd.DataFrame]:
       """Wait for new data to be added to the buffer.
       
       Returns:
           Most recent DataFrame after update
           
       Note:
           This method blocks until new data is available.
       """
       waiter = Event()
       self.waiters.append(waiter)
       waiter.wait()
       self.waiters.remove(waiter)
       return self.get_last_dataframe()

   @activemethod
   def get_all_dataframes(self) -> List[pd.DataFrame]:
       """Get all DataFrames in the buffer.
       
       Returns:
           List of all stored DataFrames
       """
       return list(self.dataframes)

class MetricsMapping:
   """Mapping and processing for different metrics types.
   
   This class handles the mapping between raw metrics from DataClay
   and processed metrics used for training. It supports three types
   of metrics:
       - power_consumption: Direct measurement from power sensors
       - cpu_usage: Aggregated CPU usage across all processes
       - ram_usage: Calculated from total and available memory
   """
   
   METRICS = {
       'power_consumption': 'scaph_host_power_microwatts',
       'cpu_usage': 'scaph_process_cpu_usage_percentage',
       'ram_usage': ['scaph_host_memory_total_bytes', 'scaph_host_memory_available_bytes']
   }
   
   @staticmethod
   def process_metric(df: pd.DataFrame, metric_type: str) -> float:
       """Process raw data into specific metric value.
       
       Args:
           df: Raw DataFrame from DataClay
           metric_type: Type of metric to process
           
       Returns:
           float: Processed metric value
           
       Raises:
           ValueError: If metric type is invalid
       """
       if metric_type not in MetricsMapping.METRICS:
           raise ValueError(f"Invalid metric type: {metric_type}")
           
       if metric_type == 'power_consumption':
           return float(df[MetricsMapping.METRICS[metric_type]].iloc[0])
           
       elif metric_type == 'cpu_usage':
           return float(df[MetricsMapping.METRICS[metric_type]].sum())
           
       elif metric_type == 'ram_usage':
           total, available = MetricsMapping.METRICS[metric_type]
           return float(df[total].iloc[0] - df[available].iloc[0])

   @staticmethod
   def get_required_columns() -> List[str]:
       """Get all required columns for metrics processing.
       
       Returns:
           List of column names required for all metrics
       """
       columns = []
       for metric in MetricsMapping.METRICS.values():
           if isinstance(metric, list):
               columns.extend(metric)
           else:
               columns.append(metric)
       return list(set(columns))

class DataFetcher:
   """Handles metrics fetching from persistent storage.
   
   This class manages the collection and buffering of metrics data
   from DataClay storage. It maintains a rolling buffer of measurements
   and provides access to collected data when sufficient samples are
   available.
   
   Attributes:
       host: DataClay host address
       dataset: Dataset identifier
       alias: Time series alias
       target_metric: Type of metric to collect
       poll_interval: Data collection interval
       timeout: Maximum wait time for data fetch
       min_samples: Minimum required samples for training
       metrics_buffer: Rolling buffer of collected measurements
   """

   def __init__(self, host: str, dataset: str, alias: str, 
                target_metric: str, poll_interval: int = 60,
                timeout: int = 300, min_samples: int = 100):
       """Initialize fetcher.
       
       Args:
           host: DataClay host address
           dataset: Dataset identifier
           alias: Time series alias
           target_metric: Metric to collect (power_consumption/cpu_usage/ram_usage)
           poll_interval: Data polling interval in seconds
           timeout: Maximum wait time for data fetch
           min_samples: Minimum number of samples required for training
       """
       self.host = host
       self.dataset = dataset
       self.alias = alias
       self.target_metric = target_metric
       self.poll_interval = poll_interval
       self.timeout = timeout
       self.min_samples = min_samples
       self.client = None
       self.tsd = None
       
       # Initialize metrics buffer (2x min_samples to allow for some history)
       self.metrics_buffer = deque(maxlen=min_samples * 2)
       LOG.info(f"Initialized DataFetcher with min_samples={min_samples}")

   def connect(self) -> bool:
       """Establish connection to metrics backend.
       
       Returns:
           bool: True if connection successful, False otherwise
       """
       try:
           self.client = Client(
               proxy_host=self.host,
               dataset=self.dataset
           )
           self.client.start()
           LOG.info("Connected to DataClay backend")
           return True
       except Exception as e:
           LOG.error(f"Connection failed: {str(e)}")
           return False

   def initialize_timeseries(self) -> bool:
       """Initialize or retrieve persistent time series data.
       
       Returns:
           bool: True if initialization successful, False otherwise
       """
       try:
           self.tsd = TimeSeriesData.get_by_alias(self.alias)
           LOG.info("Using existing TimeSeriesData")
           self.initialize_buffer()  # Pre-fill buffer with existing data
           return True
       except DataClayException:
           try:
               self.tsd = TimeSeriesData()
               self.tsd.make_persistent(alias=self.alias)
               LOG.info("Created new TimeSeriesData")
               return True
           except Exception as e:
               LOG.error(f"TS initialization failed: {str(e)}")
               return False

   def initialize_buffer(self) -> None:
       """Initialize buffer with existing data.
       
       This method attempts to pre-fill the buffer with existing measurements
       to reach min_samples faster.
       """
       try:
           # Get all available dataframes
           dataframes = self.tsd.get_all_dataframes()
           
           # Process each dataframe
           for df in dataframes:
               df = self._validate_data(df)
               metric_value = MetricsMapping.process_metric(df, self.target_metric)
               self.metrics_buffer.append(metric_value)
               
           LOG.info(f"Pre-filled buffer with {len(self.metrics_buffer)} measurements")
           
       except Exception as e:
           LOG.error(f"Failed to initialize buffer: {str(e)}")

   def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
       """Validate dataframe structure.
       
       Args:
           df: Raw DataFrame to validate
           
       Returns:
           Validated DataFrame
           
       Raises:
           ValueError: If required columns are missing
       """
       required_cols = MetricsMapping.get_required_columns()
       
       missing = [col for col in required_cols if col not in df.columns]
       if missing:
           raise ValueError(f"Missing required columns: {missing}")
           
       return df.dropna().reset_index(drop=True)

   def fetch_metrics(self) -> Optional[float]:
       """Fetch and process latest metric value.
       
       Returns:
           Processed metric value or None if fetch fails
           
       Note:
           First tries to get latest data without waiting,
           only waits if no new data is available.
       """
       try:
           # First try to get last dataframe without waiting
           df = self.tsd.get_last_dataframe()
           
           # If no new data, wait for update
           if df is None:
               df = self.tsd.wait_for_dataframe(timeout=self.timeout)
               if df is None:
                   return None
           
           # Process data
           df = self._validate_data(df)
           metric_value = MetricsMapping.process_metric(df, self.target_metric)
           self.metrics_buffer.append(metric_value)
           
           LOG.debug(f"Collected new {self.target_metric} value: {metric_value}")
           return metric_value
           
       except Exception as e:
           LOG.error(f"Fetch error: {str(e)}")
           return None
           
   def get_collected_data(self) -> Optional[pd.DataFrame]:
       """Get collected metrics if enough samples are available.
       
       Returns:
           DataFrame with collected metrics or None if insufficient data
           
       Note:
           Returns None if number of collected samples is less than min_samples
       """
       if len(self.metrics_buffer) < self.min_samples:
           LOG.info(f"Insufficient samples: {len(self.metrics_buffer)}/{self.min_samples}")
           return None
           
       # Create DataFrame with timestamps
       data = pd.DataFrame({
           self.target_metric: list(self.metrics_buffer),
           'timestamp': pd.date_range(
               end=pd.Timestamp.now(),
               periods=len(self.metrics_buffer),
               freq=f"{self.poll_interval}S"
           )
       }).set_index('timestamp')
       
       LOG.info(f"Returning dataset with {len(data)} samples")
       return data