"""Data fetching utilities for ICOS-FL time series prediction.

This module provides the Fetcher class for connecting to DataClay and
retrieving time series data for LSTM model training and evaluation.
It handles establishing connections, retrieving data, and processing
it into a suitable format for the Processor component.

Additionally, this module provides resource configuration management
classes that enable flexible matching and organization of resources
based on custom rules, facilitating integration with external bridge
applications and monitoring systems.
"""

import time
from threading import Event
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd
from dataclay import Client, DataClayObject, activemethod
from dataclay.exceptions import DataClayException

# Type alias for match rules: (key, matcher_function, expected_value)
MatchRule = Tuple[str, Callable[[Any, Any], bool], Any]


class ResourceConfiguration(DataClayObject):
    """Hold the configuration for a resource, including the rules to match it.

    The rules are provided as a list of tuples, where each tuple contains:
    - The key to match
    - A function to match the value (typically from operator module)
    - The expected value

    This enables flexible resource matching based on customizable criteria,
    supporting complex filtering and organization of monitoring resources.

    Example:
        >>> rc = ResourceConfiguration("cpu_intensive", [("cpu_usage", operator.gt, 0.8)])

        This configuration will match any resource where cpu_usage > 0.8

    Attributes:
        name: Unique identifier for this resource configuration
        rules: List of match rules to apply
        metric_names: Set of metric names being collected for matched resources
    """

    name: str
    rules: List[MatchRule]
    metric_names: set[str]

    def __init__(
        self,
        name: str,
        rules: Optional[List[MatchRule]] = None,
        metric_names: Optional[set[str]] = None,
    ) -> None:
        """Initialize a ResourceConfiguration with matching rules.

        Creates a new resource configuration with the specified name and optional
        matching rules. The configuration can be used to identify resources that
        meet specific criteria and determine which metrics should be collected.

        Args:
            name: Unique identifier for this configuration
            rules: List of (key, matcher_function, value) tuples defining match criteria
            metric_names: Initial set of metrics to collect for matched resources
        """
        self.name = name
        self.rules = rules or []
        self.metric_names = metric_names or set()

    @activemethod
    def add_metric(self, metric_name: str) -> None:
        """Add a metric to be collected for resources matching this configuration.

        Adds the specified metric name to the collection set. If the metric
        already exists in the set, no action is taken (sets naturally handle duplicates).

        Args:
            metric_name: Name of the metric to add to the collection set
        """
        self.metric_names.add(metric_name)

    @activemethod
    def remove_metric(self, metric_name: str) -> None:
        """Remove a metric from the collection set.

        Removes the specified metric from the set of metrics to be collected
        for resources matching this configuration.

        Args:
            metric_name: Name of the metric to remove from collection

        Raises:
            KeyError: If the metric_name doesn't exist in the metric_names set
        """
        self.metric_names.remove(metric_name)

    @activemethod
    def match(self, resource_kvs: dict[str, str]) -> bool:
        """Check if a resource matches all rules in this configuration.

        Evaluates all rules against the provided resource key-value pairs.
        A resource matches only if ALL rules evaluate to True. If a rule
        references a key that doesn't exist in the resource, the match fails.

        Args:
            resource_kvs: Dictionary of resource attributes to match against

        Returns:
            True if all rules match, False if any rule fails

        Example:
            >>> rc = ResourceConfiguration("test", [("type", operator.eq, "server")])
            >>> rc.match({"type": "server", "location": "rack1"})  # Returns True
            >>> rc.match({"type": "client", "location": "rack1"})  # Returns False
        """
        for rule in self.rules:
            key, matcher, value = rule
            # Check if key exists in resource_kvs
            if key not in resource_kvs:
                return False
            # Apply the matcher function
            if not matcher(resource_kvs[key], value):
                return False
        return True


class BridgeConfiguration(DataClayObject):
    """Aggregate configuration for the monitoring bridge system.

    This class serves as a central configuration manager, holding multiple
    resource configurations and global settings for the bridge application.
    It provides methods to manage resource configurations dynamically and
    find matching configurations for incoming resources.

    Key Features:
    - Centralized management of resource configurations
    - Dynamic addition/removal of configurations
    - Efficient matching of resources to configurations
    - Configurable dataframe time-to-live settings

    Attributes:
        resource_configurations: Dictionary mapping names to ResourceConfiguration objects
        dataframe_ttl: Time-to-live for dataframes in seconds (default: 60)
    """

    resource_configurations: dict[str, ResourceConfiguration]
    dataframe_ttl: int

    def __init__(self) -> None:
        """Initialize an empty BridgeConfiguration with default settings.

        Creates a new bridge configuration with an empty resource configuration
        dictionary and a default TTL of 60 seconds for dataframes. The TTL
        determines how long dataframes are retained before being considered stale.
        """
        self.resource_configurations = {}
        self.dataframe_ttl = 60

    @activemethod
    def set_res_config(self, rc: ResourceConfiguration) -> None:
        """Add or update a resource configuration.

        Stores the provided resource configuration in the internal dictionary,
        using the configuration's name as the key. If a configuration with the
        same name already exists, it will be replaced with the new one.

        Args:
            rc: ResourceConfiguration object to add or update
        """
        self.resource_configurations[rc.name] = rc

    @activemethod
    def remove_res_config(self, name: str) -> None:
        """Remove a resource configuration by name.

        Deletes the resource configuration with the specified name from the
        internal configuration dictionary.

        Args:
            name: Name of the configuration to remove

        Raises:
            KeyError: If no configuration with the given name exists
        """
        del self.resource_configurations[name]

    @activemethod
    def get_matching_res_configs(
        self, resource_kvs: dict[str, str]
    ) -> List[ResourceConfiguration]:
        """Find all resource configurations that match the given resource attributes.

        Iterates through all stored configurations and returns those whose rules
        match the provided resource key-value pairs. A resource can match multiple
        configurations, and all matching configurations are returned.

        Args:
            resource_kvs: Dictionary of resource attributes to match against

        Returns:
            List of ResourceConfiguration objects that match the resource

        Example:
            >>> bc = BridgeConfiguration()
            >>> rc1 = ResourceConfiguration("high_cpu", [("cpu_usage", operator.gt, 0.8)])
            >>> rc2 = ResourceConfiguration("prod_server", [("env", operator.eq, "production")])
            >>> bc.set_res_config(rc1)
            >>> bc.set_res_config(rc2)
            >>> matches = bc.get_matching_res_configs({"cpu_usage": 0.9, "env": "production"})
            >>> # Returns both rc1 and rc2 as both match the resource
        """
        return [rc for rc in self.resource_configurations.values() if rc.match(resource_kvs)]


class TimeSeriesData(DataClayObject):
    """Class for managing time series data with a sliding window approach.

    This class maintains a single DataFrame with a fixed maximum size,
    implementing a sliding window over time. New data points are appended
    and old data points are removed to maintain the window size.
    """

    dataframe: Optional[pd.DataFrame]
    max_rows: int
    waiters: List[Event]

    def __init__(self, max_rows: int = 300) -> None:
        """Initialize the TimeSeriesData object.

        Args:
            max_rows: Maximum number of rows to keep in the DataFrame. Default is 300,
                      which provides enough history for LSTM training and 5-minute prediction
                      with 3-second interval data collection.
        """
        self.dataframe = None
        self.max_rows = max_rows
        self.waiters = []

    @activemethod
    def add_dataframe(self, df: pd.DataFrame) -> None:
        """Add new data to the unified dataframe, maintaining the sliding window.

        When new data is added, the oldest data points are removed if the total
        size exceeds max_rows.

        Args:
            df: New DataFrame to append
        """
        if self.dataframe is None:
            self.dataframe = df
        else:
            # Append new data
            self.dataframe = pd.concat([self.dataframe, df])

            # Maintain sliding window by removing oldest entries
            if len(self.dataframe) > self.max_rows:
                self.dataframe = self.dataframe.iloc[-self.max_rows :]

        # Notify waiters that new data is available
        for waiter in self.waiters:
            waiter.set()

    @activemethod
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the current unified DataFrame.

        Returns:
            The current DataFrame, or None if no data has been added yet.
        """
        return self.dataframe

    @activemethod
    def wait_for_dataframe(self) -> pd.DataFrame:
        """Wait for new data to be added to the DataFrame.

        This method blocks until new data is added through add_dataframe().

        Returns:
            The updated DataFrame after new data has been added.
        """
        waiter = Event()
        self.waiters.append(waiter)
        waiter.wait()
        self.waiters.remove(waiter)
        return self.dataframe


class Fetcher:
    """Fetcher for retrieving time series data from DataClay.

    This class handles connecting to DataClay, retrieving data through
    the TimeSeriesData object, and processing it into a format suitable
    for LSTM model training.

    Args:
        proxy_host: Host address for the DataClay proxy
        dataset: Dataset name to connect to
    """

    def __init__(self, proxy_host: str = "127.0.0.1", dataset: str = "admin") -> None:
        """Initialize the Fetcher with DataClay connection parameters.

        Args:
            proxy_host: DataClay proxy host address
            dataset: DataClay dataset name
        """
        self.proxy_host = proxy_host
        self.dataset = dataset
        self.client = None
        self.time_series_data = None

        # Initialize connection
        self._connect()

    def _connect(self) -> None:
        """Connect to DataClay and initialize the TimeSeriesData client.

        Raises:
            DataClayException: If there is an error connecting to DataClay
            RuntimeError: If there is an error initializing the TimeSeriesData client
        """
        try:
            self.client = Client(proxy_host=self.proxy_host, dataset=self.dataset)

            if self.client is None:
                raise RuntimeError("Failed to create DataClay client")

            self.client.start()

            # Try to get the existing TimeSeriesData object
            self.time_series_data = TimeSeriesData.get_by_alias("timeseries")

            if self.time_series_data is None:
                raise RuntimeError("Failed to retrieve TimeSeriesData by alias")

        except DataClayException as e:
            raise DataClayException(f"Failed to connect to DataClay: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TimeSeriesData: {e}") from e

    def _disconnect(self) -> None:
        """Disconnect from DataClay."""
        if self.client is not None:
            try:
                self.client.stop()
            except (ConnectionError, RuntimeError, DataClayException) as e:
                print(f"Warning: Error during DataClay disconnect: {e}")
            finally:
                self.client = None
                self.time_series_data = None

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the raw DataFrame into the format needed for model training.

        Converts the raw DataClay metrics into standardized columns with appropriate units:
        - power_microwatts → power_consumption (watts)
        - load_avg_one → cpu_usage
        - memory_bytes → memory_usage (MB)
        - Adds derived power_consumption metric

        Args:
            df: Raw DataFrame from TimeSeriesData

        Returns:
            Processed DataFrame with standardized columns and formats
        """
        if df is None or len(df) == 0:
            return None

        # Create a new dataframe with the same index
        dataset = pd.DataFrame(index=df.index)

        # Convert index to Unix timestamp (seconds since epoch)
        dataset["timestamp"] = dataset.index / 1_000_000_000  # Convert nanoseconds to seconds

        # Find columns by type
        power_col = None
        load_col = None
        mem_total_col = None
        mem_avail_col = None

        for col in df.columns:
            if "power_microwatts" in col:
                power_col = col
            elif "load_avg_one" in col:
                load_col = col
            elif "memory_total_bytes" in col:
                mem_total_col = col
            elif "memory_available_bytes" in col:
                mem_avail_col = col

        # Add power consumption in watts
        if power_col:
            dataset["power_consumption"] = df[power_col] / 1_000_000  # Convert to watts

        # Add CPU usage (load average)
        if load_col:
            dataset["cpu_usage"] = df[load_col]

        # Calculate RAM usage in MB (total - available)
        if mem_total_col and mem_avail_col:
            dataset["memory_usage"] = (df[mem_total_col] - df[mem_avail_col]) / (
                1024 * 1024
            )  # Convert to MB

        # Reset index to have standard integer indexing
        dataset = dataset.reset_index(drop=True)

        return dataset

    def fetch_data(self, timeout: int = 200) -> pd.DataFrame:
        """Fetch data from DataClay and process it for LSTM training.

        Retrieves time series data, converting it to a format suitable for
        the LSTM model with standardized column names and units.

        Args:
            timeout: Timeout in seconds for waiting for data

        Returns:
            Processed DataFrame ready for model training

        Raises:
            TimeoutError: If no data is available within the timeout period
        """
        if self.time_series_data is None:
            raise RuntimeError("TimeSeriesData not initialized")

        # Start timer for timeout
        start_time = time.time()

        # Get current dataframe
        df = self.time_series_data.get_dataframe()

        # If no data is available, poll until timeout
        while df is None:
            # Check if timeout exceeded
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Timed out after {timeout} seconds waiting for data")

            # Wait a bit before trying again
            time.sleep(1)

            if self.time_series_data is None:
                raise RuntimeError("TimeSeriesData became None during fetch operation")

            # Try to get the dataframe again
            df = self.time_series_data.get_dataframe()

        # Process the dataframe
        return self._post_process(df)

    def __del__(self) -> None:
        """Clean up resources when the object is destroyed."""
        self._disconnect()
