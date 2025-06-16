"""Data fetching utilities for ICOS-FL time series prediction.

This module provides the Fetcher class for connecting to DataClay and
retrieving time series data for LSTM model training and evaluation.
It handles establishing connections, retrieving data, and processing
it into a suitable format for the Processor component.
"""

import time
from threading import Event
from typing import List, Optional

import pandas as pd
from dataclay import Client, DataClayObject, activemethod
from dataclay.exceptions import DataClayException

from icos_fl.utils.singleton import Singleton


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


class Fetcher(metaclass=Singleton):
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
