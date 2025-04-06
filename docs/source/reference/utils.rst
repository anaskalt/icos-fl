=====
Utils
=====

This page documents the utility components of ICOS-FL, including the data processing, fetching, and visualization utilities.

Fetcher Module
--------------

.. py:class:: icos_fl.utils.fetcher.TimeSeriesData(max_rows=300)

   Class for managing time series data with a sliding window approach.

   :param max_rows: Maximum number of rows to keep in the DataFrame. Default is 300,
                   which provides enough history for LSTM training and 5-minute prediction
                   with 3-second interval data collection.
   :type max_rows: int, optional

   .. py:method:: add_dataframe(df)

      Add new data to the unified dataframe, maintaining the sliding window.

      When new data is added, the oldest data points are removed if the total
      size exceeds max_rows.

      :param df: New DataFrame to append
      :type df: pd.DataFrame

   .. py:method:: get_dataframe()

      Get the current unified DataFrame.

      :return: The current DataFrame, or None if no data has been added yet.
      :rtype: Optional[pd.DataFrame]

   .. py:method:: wait_for_dataframe()

      Wait for new data to be added to the DataFrame.

      This method blocks until new data is added through add_dataframe().

      :return: The updated DataFrame after new data has been added.
      :rtype: pd.DataFrame

.. py:class:: icos_fl.utils.fetcher.Fetcher(proxy_host="127.0.0.1", dataset="admin")

   Fetcher for retrieving time series data from DataClay.

   This class handles connecting to DataClay, retrieving data through
   the TimeSeriesData object, and processing it into a format suitable
   for LSTM model training.

   :param proxy_host: Host address for the DataClay proxy
   :type proxy_host: str, optional
   :param dataset: Dataset name to connect to
   :type dataset: str, optional

   .. py:method:: fetch_data(timeout=200)

      Fetch data from DataClay and process it for LSTM training.

      Retrieves time series data, converting it to a format suitable for
      the LSTM model with standardized column names and units.

      :param timeout: Timeout in seconds for waiting for data
      :type timeout: int, optional
      :return: Processed DataFrame ready for model training
      :rtype: pd.DataFrame
      :raises TimeoutError: If no data is available within the timeout period

Processor Module
----------------

.. py:class:: icos_fl.utils.processor.TimeSeriesDataset(df, start_index, population, time_step, metric, device)

   Dataset for time series prediction with sliding window approach.

   Creates sequences of consecutive time steps as inputs and
   uses the next value as the prediction target.

   :param df: Input DataFrame containing the time series data
   :type df: pd.DataFrame
   :param start_index: Starting index in the DataFrame to create sequences
   :type start_index: int
   :param population: Total number of samples to include from start_index
   :type population: int
   :param time_step: Number of time steps (sequence length) for LSTM input
   :type time_step: int
   :param metric: Column name in the DataFrame to use as target
   :type metric: str
   :param device: PyTorch device to place tensors on
   :type device: torch.device

   .. py:method:: __getitem__(index)

      Get a sequence and its target.

      :param index: Index of the sequence to retrieve
      :type index: int
      :return: Tuple of (input_sequence, target_value)
      :rtype: Tuple[torch.Tensor, torch.Tensor]

   .. py:method:: __len__()

      Return the number of sequences in the dataset.

      :return: Number of sequences
      :rtype: int

.. py:class:: icos_fl.utils.processor.Processor(time_step, metric, batch_size=64, train_ratio=0.8, device=None)

   Processor for time series data preparation in ICOS-FL.

   This class handles data preprocessing for time series forecasting,
   providing methods for data normalization, sequence creation,
   and DataLoader generation.

   :param time_step: Number of time steps (sequence length) for LSTM input
   :type time_step: int
   :param metric: Default column name in the DataFrame to use as target
   :type metric: str
   :param batch_size: Default batch size for DataLoaders
   :type batch_size: int, optional
   :param train_ratio: Default ratio for train/test split
   :type train_ratio: float, optional
   :param device: PyTorch device to place tensors on
   :type device: Optional[torch.device], optional

   .. py:method:: create_data_loaders(df, time_step=None, metric=None, batch_size=None, train_ratio=None, device=None)

      Create DataLoaders for training and validation.

      This method handles the complete data preparation pipeline:
      1. Normalizes the data
      2. Splits into training and validation sets
      3. Creates appropriate datasets with sliding window sequences
      4. Wraps datasets in DataLoaders

      :param df: DataFrame containing the time series data
      :type df: pd.DataFrame
      :param time_step: Sequence length (uses instance default if None)
      :type time_step: Optional[int], optional
      :param metric: Column name to use as target (uses instance default if None)
      :type metric: Optional[str], optional
      :param batch_size: Batch size for DataLoaders (uses instance default if None)
      :type batch_size: Optional[int], optional
      :param train_ratio: Ratio of data for training (uses instance default if None)
      :type train_ratio: Optional[float], optional
      :param device: PyTorch device (uses instance default if None)
      :type device: Optional[torch.device], optional
      :return: Tuple of (train_dataloader, val_dataloader, train_dataset, val_dataset)
      :rtype: Tuple[DataLoader, DataLoader, TimeSeriesDataset, TimeSeriesDataset]

   .. py:method:: _normalize_data(df)

      Normalize the dataset using standardization (zero mean, unit variance).

      :param df: Input DataFrame containing the time series data
      :type df: pd.DataFrame
      :return: Normalized DataFrame with the same structure
      :rtype: pd.DataFrame

   .. py:method:: _train_test_split(df, train_ratio)

      Split the dataset into training and testing sets.

      :param df: Input DataFrame containing the time series data
      :type df: pd.DataFrame
      :param train_ratio: Ratio for splitting data into train and test sets (0-1)
      :type train_ratio: float
      :return: Tuple containing the number of training and testing samples
      :rtype: Tuple[int, int]

Bridge Configuration
--------------------

.. py:class:: icos_fl.utils.fetcher.ResourceConfiguration(name, rules=None, metric_names=None)

   Hold the configuration for a resource, including the rules to match it.

   The rules will be given in the form of a list of tuples, where each tuple
   contains the key to match, a function to match the value, and the value to
   match.

   :param name: Name of the resource configuration
   :type name: str
   :param rules: List of match rules for the resource, defaults to None
   :type rules: Optional[list[MatchRule]], optional
   :param metric_names: Set of metric names to collect, defaults to None
   :type metric_names: Optional[set[str]], optional

   .. py:method:: add_metric(metric_name)

      Add a metric to collect for this resource.

      :param metric_name: Name of the metric to collect
      :type metric_name: str

   .. py:method:: remove_metric(metric_name)

      Remove a metric from collection.

      :param metric_name: Name of the metric to remove
      :type metric_name: str

   .. py:method:: match(resource_kvs)

      Check if this configuration matches the given resource key-value pairs.

      :param resource_kvs: Dictionary of resource key-value pairs
      :type resource_kvs: dict[str, str]
      :return: True if matches, False otherwise
      :rtype: bool

.. py:class:: icos_fl.utils.fetcher.BridgeConfiguration()

   Aggregate the configuration for the bridge.

   This class holds the configuration for the bridge, including the resource
   configuration objects. It also holds the time-to-live for the dataframes.

   .. py:method:: set_res_config(rc)

      Set a resource configuration.

      :param rc: Resource configuration to set
      :type rc: ResourceConfiguration

   .. py:method:: remove_res_config(name)

      Remove a resource configuration by name.

      :param name: Name of the resource configuration to remove
      :type name: str

   .. py:method:: get_matching_res_configs(resource_kvs)

      Get all resource configurations that match the given resource.

      :param resource_kvs: Dictionary of resource key-value pairs
      :type resource_kvs: dict[str, str]
      :return: List of matching resource configurations
      :rtype: list[ResourceConfiguration]

Utility Classes
---------------

.. py:class:: icos_fl.utils.singleton.Singleton

   A metaclass to make a class a singleton.

   Usage::

       class MySingletonClass(metaclass=Singleton):
           ...

Colors Module
-------------

The ``icos_fl.utils.colors`` module provides color constants and utility functions for terminal output:

.. py:function:: icos_fl.utils.colors.paint(color, text, reset=RST)

   Apply ANSI color codes to text.

   :param color: ANSI color code to apply
   :type color: str
   :param text: Text string to color
   :type text: str
   :param reset: ANSI code to apply after text (default: RST)
   :type reset: str, optional
   :return: Colored text string with reset code appended
   :rtype: str

Logo Module
-----------

The ``icos_fl.utils.logo`` module provides ASCII art and banner functions:

.. py:function:: icos_fl.utils.logo.print_banner(logo, title="", message="", border_color=BCYA, logo_color=BBLU, title_color=BWHT, message_color=BGRN, ver=None, show_version=True)

   Displays a customizable banner with logo and optional text.

   :param logo: The ASCII art logo to display
   :type logo: str
   :param title: Optional title to display above the logo (default: "")
   :type title: str, optional
   :param message: Optional message to display below the logo (default: "")
   :type message: str, optional
   :param border_color: ANSI color code for the border (default: BCYA)
   :type border_color: str, optional
   :param logo_color: ANSI color code for the logo (default: BBLU)
   :type logo_color: str, optional
   :param title_color: ANSI color code for the title (default: BWHT)
   :type title_color: str, optional
   :param message_color: ANSI color code for the message (default: BGRN)
   :type message_color: str, optional
   :param ver: Optional version string (uses icos_fl.version if None)
   :type ver: Optional[str], optional
   :param show_version: Whether to display version information (default: True)
   :type show_version: bool, optional

Example Usage
-------------

.. code-block:: python

   from icos_fl.utils.fetcher import Fetcher
   from icos_fl.utils.processor import Processor
   import torch

   # Connect to DataClay and fetch data
   fetcher = Fetcher(proxy_host="127.0.0.1", dataset="admin")
   df = fetcher.fetch_data(timeout=60)

   # Process data for LSTM training
   processor = Processor(
       time_step=10,
       metric="cpu_usage",
       batch_size=64,
       train_ratio=0.8,
       device=torch.device("cpu")
   )

   # Create DataLoaders
   train_loader, val_loader, _, _ = processor.create_data_loaders(df)
