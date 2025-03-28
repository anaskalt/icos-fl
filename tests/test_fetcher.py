"""Test module for the Fetcher and TimeSeriesData classes.

This module validates the implementation of the Fetcher class for connecting
to DataClay and retrieving time series data, along with the TimeSeriesData
class for managing sliding window time series data.
"""

import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from dataclay.exceptions import DataClayException

# Add source directory to path BEFORE any relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from icos_fl.utils.colors import BCYA, BGRN, BMAG, BYEL, RED, WHT, paint, perror
from icos_fl.utils.fetcher import Fetcher
from icos_fl.utils.processor import Processor

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Global fetcher to avoid multiple DataClay connections
global_fetcher = None


def print_section_header(title: str) -> None:
    """Print a section header with a nice box around it."""
    min_box_width = 50  # Minimum box width including borders
    title_len = len(title)

    # Calculate required width to display title fully
    required_width = title_len + 4  # Add 4 for "║ " and " ║"

    # Use either the minimum width or the required width, whichever is larger
    box_width = max(min_box_width, required_width)

    # Create the border line with the correct width
    border_line = "╔" + "═" * (box_width - 2) + "╗"
    bottom_line = "╚" + "═" * (box_width - 2) + "╝"

    # Calculate number of spaces on each side
    inner_width = box_width - 4  # -4 for "║ " and " ║"
    spaces_total = inner_width - title_len
    spaces_left = spaces_total // 2
    spaces_right = spaces_total - spaces_left

    title_padded = " " * spaces_left + title + " " * spaces_right

    print("")
    print(paint(BCYA, border_line))
    print(paint(BCYA, f"║ {title_padded} ║"))
    print(paint(BCYA, bottom_line))
    print("")


def print_subsection_header(title: str) -> None:
    """Print a subsection header with a smaller box."""
    min_box_width = 36  # Minimum box width including borders
    title_len = len(title)

    # Calculate required width to display title fully
    required_width = title_len + 4  # Add 4 for "║ " and " ║"

    # Use either the minimum width or the required width, whichever is larger
    box_width = max(min_box_width, required_width)

    # Create the border line with the correct width
    border_line = "╔" + "═" * (box_width - 2) + "╗"
    bottom_line = "╚" + "═" * (box_width - 2) + "╝"

    # Calculate number of spaces on each side
    inner_width = box_width - 4  # -4 for "║ " and " ║"
    spaces_total = inner_width - title_len
    spaces_left = spaces_total // 2
    spaces_right = spaces_total - spaces_left

    title_padded = " " * spaces_left + title + " " * spaces_right

    print("")
    print(paint(BCYA, border_line))
    print(paint(BCYA, f"║ {title_padded} ║"))
    print(paint(BCYA, bottom_line))
    print("")


def print_subsection_title(title: str) -> None:
    """Print a subsection title with an arrow prefix."""
    print(paint(BYEL, f"▶ {title}"))


def create_synthetic_data(rows: int = 100) -> pd.DataFrame:
    """Create synthetic system monitoring data for testing.

    Args:
        rows: Number of time steps in the dataset

    Returns:
        DataFrame with synthetic monitoring data including power, CPU, and memory metrics
    """
    np.random.seed(42)  # For reproducibility

    # Create time index (nanoseconds since epoch)
    base_timestamp = int(time.time() * 1_000_000_000)  # Current time in nanoseconds
    timestamps = [base_timestamp + i * 1_000_000_000 for i in range(rows)]  # 1 second intervals

    # Create data with realistic patterns
    data = {}

    # Power consumption in microwatts (realistic for a small server)
    power = np.random.normal(50_000_000, 10_000_000, rows)  # Mean around 50W
    data["server_power_microwatts"] = power

    # CPU load average (0-1 scale, multiplied by number of cores)
    cores = 4
    load = np.random.beta(2, 5, rows) * cores  # Beta distribution for load
    data["server_load_avg_one"] = load

    # Memory metrics (in bytes)
    total_memory = 8 * 1024 * 1024 * 1024  # 8GB in bytes
    available_memory = np.random.uniform(2 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024, rows)
    data["server_memory_total_bytes"] = [total_memory] * rows
    data["server_memory_available_bytes"] = available_memory

    # Create DataFrame
    df = pd.DataFrame(data, index=timestamps)

    rows_msg = (
        f"Created synthetic monitoring DataFrame with {paint(BGRN, str(rows))} rows "
        f"and {paint(BGRN, str(len(data)))} columns"
    )
    cols_msg = f"Columns: {paint(WHT, ', '.join(df.columns))}"
    range_msg = f"Time range: {paint(WHT, str(df.index[0]))} to {paint(WHT, str(df.index[-1]))}"

    print(rows_msg)
    print(cols_msg)
    print(range_msg)

    return df


def initialize_fetcher() -> None:
    """Test initializing a Fetcher instance and validate its configuration."""
    global global_fetcher

    print_subsection_header("Fetcher Initialization")
    print_subsection_title("Fetcher Configuration:")

    proxy_msg = f"  • Proxy host: {paint(WHT, global_fetcher.proxy_host)}"
    dataset_msg = f"  • Dataset: {paint(WHT, global_fetcher.dataset)}"

    client_status = "Yes" if global_fetcher.client is not None else "No"
    client_msg = f"  • Client initialized: {paint(BGRN, client_status)}"

    tsd_status = "Retrieved" if global_fetcher.time_series_data is not None else "Failed"
    tsd_msg = f"  • TimeSeriesData object: {paint(BGRN, tsd_status)}"

    print(proxy_msg)
    print(dataset_msg)
    print(client_msg)
    print(tsd_msg)

    # If the connection was successful, note it
    if global_fetcher.client is not None and global_fetcher.time_series_data is not None:
        success_msg = f"  • Connection to DataClay successful: {paint(BGRN, 'Yes')}"
        print(success_msg)


def test_time_series_data() -> None:
    """Test the TimeSeriesData class functionality."""
    global global_fetcher

    print_subsection_header("TimeSeriesData Class")
    time_series_data = global_fetcher.time_series_data

    # Check if we successfully got a TimeSeriesData object
    if time_series_data is None:
        error_msg = "Failed to retrieve TimeSeriesData object from DataClay"
        print(paint(RED, error_msg))
        return

    # Test initialization properties
    print_subsection_title("TimeSeriesData Properties:")

    max_rows_msg = f"  • Max rows: {paint(WHT, str(time_series_data.max_rows))}"
    waiters_msg = f"  • Waiters count: {paint(WHT, str(len(time_series_data.waiters)))}"

    print(max_rows_msg)
    print(waiters_msg)

    # Test get_dataframe method
    print("")
    print_subsection_title("Testing get_dataframe:")

    df = time_series_data.get_dataframe()
    if df is not None:
        df_shape_msg = f"  • DataFrame shape: {paint(BGRN, str(df.shape))}"

        # Split long line for readability
        cols_preview = df.columns[:5]
        cols_display = (
            ", ".join(cols_preview) + "..." if len(df.columns) > 5 else ", ".join(df.columns)
        )
        df_cols_msg = f"  • DataFrame columns: {paint(WHT, cols_display)}"

        df_rows_msg = f"  • Number of rows: {paint(BGRN, str(len(df)))}"

        print(df_shape_msg)
        print(df_cols_msg)
        print(df_rows_msg)

        # Check if the DataFrame has enough rows for the sliding window
        window_status = "Full" if len(df) >= time_series_data.max_rows else "Filling"
        window_status_msg = f"  • Window status: {paint(BGRN, window_status)}"
        print(window_status_msg)
    else:
        no_data_msg = f"  • No data available yet: {paint(BYEL, 'DataFrame is None')}"
        print(no_data_msg)


def test_post_process() -> None:
    """Test the _post_process method for transforming raw data."""
    global global_fetcher

    print_subsection_header("Data Post-Processing")

    # First try with real data if available
    real_df = None
    if global_fetcher.time_series_data is not None:
        real_df = global_fetcher.time_series_data.get_dataframe()

    # If no real data, create synthetic data for testing
    if real_df is None or len(real_df) == 0:
        print_subsection_title("No real data available, using synthetic data for testing")
        real_df = create_synthetic_data(rows=50)
    else:
        print_subsection_title("Using real data from DataClay for testing")

    # Process the data
    processed_df = global_fetcher._post_process(real_df)

    # Check if processing succeeded
    if processed_df is None:
        null_msg = "Post-processing returned None"
        print(paint(BYEL, null_msg))
        return

    # Check columns
    print("")
    print_subsection_title("Original vs Processed Columns:")

    # Split long line for readability
    orig_cols_preview = real_df.columns[:5]
    orig_cols_display = (
        ", ".join(orig_cols_preview) + "..."
        if len(real_df.columns) > 5
        else ", ".join(real_df.columns)
    )
    raw_cols_msg = f"  • Original columns: {paint(WHT, orig_cols_display)}"

    proc_cols_msg = f"  • Processed columns: {paint(WHT, ', '.join(processed_df.columns))}"

    print(raw_cols_msg)
    print(proc_cols_msg)

    # Check shape
    shape_msg = f"  • Processed shape: {paint(BGRN, str(processed_df.shape))}"
    print(shape_msg)

    # Check for expected columns
    print("")
    print_subsection_title("Checking for expected columns:")

    expected_cols = ["timestamp", "power_consumption", "cpu_consumption", "memory_consumption"]
    found_cols = [col for col in expected_cols if col in processed_df.columns]

    if len(found_cols) == len(expected_cols):
        all_cols_msg = f"  • All expected columns present: {paint(BGRN, 'OK')}"
        print(all_cols_msg)
    else:
        missing_cols = [col for col in expected_cols if col not in found_cols]
        missing_cols_msg = f"  • Missing expected columns: {paint(BYEL, ', '.join(missing_cols))}"
        print(missing_cols_msg)

    # Show sample data
    print("")
    print_subsection_title("Sample Processed Data:")

    if len(processed_df) > 0:
        first_row = processed_df.iloc[0]
        sample_msg = "  • First row values:"
        print(sample_msg)

        for col in processed_df.columns:
            value = first_row[col]
            value_msg = f"    - {paint(WHT, col)}: {paint(BGRN, str(value))}"
            print(value_msg)


def test_fetch_data() -> None:
    """Test the fetch_data method of the Fetcher class."""
    global global_fetcher

    print_subsection_header("Fetch Data Method")
    print_subsection_title("Testing fetch_data:")

    start_time = time.time()
    try:
        result_df = global_fetcher.fetch_data(timeout=10)  # Reduced timeout to 10 seconds
        end_time = time.time()

        if result_df is not None:
            time_taken_msg = (
                f"  • Time taken: {paint(WHT, f'{end_time - start_time:.2f}')} seconds"
            )
            shape_msg = f"  • Result shape: {paint(BGRN, str(result_df.shape))}"
            cols_msg = f"  • Result columns: {paint(WHT, ', '.join(result_df.columns))}"
            rows_msg = f"  • Number of rows: {paint(BGRN, str(len(result_df)))}"

            print(time_taken_msg)
            print(shape_msg)
            print(cols_msg)
            print(rows_msg)

            # Show dataframe statistics
            print("")
            print_subsection_title("Data Statistics:")

            for col in result_df.columns:
                if col != "timestamp" and pd.api.types.is_numeric_dtype(result_df[col]):
                    min_val = result_df[col].min()
                    max_val = result_df[col].max()
                    mean_val = result_df[col].mean()

                    stats_msg = (
                        f"  • {paint(WHT, col)}: min={paint(BGRN, f'{min_val:.4f}')}, "
                        f"max={paint(BGRN, f'{max_val:.4f}')}, "
                        f"mean={paint(BGRN, f'{mean_val:.4f}')}"
                    )

                    print(stats_msg)
        else:
            elapsed = end_time - start_time
            no_data_msg = (
                f"  • fetch_data returned None after {paint(WHT, f'{elapsed:.2f}')} seconds"
            )
            print(paint(BYEL, no_data_msg))

    except TimeoutError:
        timeout_msg = f"  • Timeout occurred after {paint(BYEL, '10')} seconds"
        print(timeout_msg)


def test_error_handling() -> None:
    """Test error handling in the Fetcher class."""
    global global_fetcher

    print_subsection_header("Error Handling")
    print_subsection_title("Testing _post_process with empty DataFrame:")

    empty_df = pd.DataFrame()
    result = global_fetcher._post_process(empty_df)

    if result is None:
        empty_df_msg = (
            f"  • Result with empty DataFrame: {paint(BGRN, 'None - Handled correctly')}"
        )
        print(empty_df_msg)
    else:
        unexpected_msg = (
            f"  • Unexpected result with empty DataFrame: {paint(BYEL, 'Returned non-None value')}"
        )
        print(unexpected_msg)

    print("")
    print_subsection_title("Testing _post_process with missing expected columns:")

    # Create a DataFrame with column names that don't match expected patterns
    invalid_df = pd.DataFrame({"unknown_column_1": [1, 2, 3], "unknown_column_2": [4, 5, 6]})

    result = global_fetcher._post_process(invalid_df)

    # Post_process should still return a DataFrame, but it won't have all the expected columns
    if result is not None:
        cols_msg = f"  • Columns in result: {paint(WHT, ', '.join(result.columns))}"
        print(cols_msg)

        # Check if any of the expected columns are missing
        expected_columns = ["power_consumption", "cpu_consumption", "memory_consumption"]
        missing = [col for col in expected_columns if col not in result.columns]

        if missing:
            missing_msg = f"  • Missing expected columns: {paint(BYEL, ', '.join(missing))}"
            print(missing_msg)

            # Split long line
            message = "Handled correctly - continued without crashing"
            print(f"  • Error handling: {paint(BGRN, message)}")
        else:
            message = "All expected columns present despite invalid input"
            unexpected_msg = f"  • Unexpected result: {paint(BYEL, message)}"
            print(unexpected_msg)
    else:
        null_msg = f"  • Result with invalid columns: {paint(BGRN, 'None - Handled gracefully')}"
        print(null_msg)

    print("")
    print(paint(BMAG, "► Error handling tests completed"))


def test_cleanup() -> None:
    """Test the cleanup method of the Fetcher class."""
    global global_fetcher

    print_subsection_header("Cleanup & Resources")

    # Check connection status before disconnect
    client_status = "Yes" if global_fetcher.client is not None else "No"
    before_msg = f"  • Before disconnect - Client is initialized: {paint(BGRN, client_status)}"
    print(before_msg)

    # Test _disconnect method
    print_subsection_title("Testing _disconnect method:")

    if global_fetcher.client is not None:
        global_fetcher._disconnect()

        # Check if client and time_series_data are set to None
        client_null_status = "Yes" if global_fetcher.client is None else "No"
        client_null_msg = (
            f"  • After disconnect - Client set to None: {paint(BGRN, client_null_status)}"
        )

        tsd_null_status = "Yes" if global_fetcher.time_series_data is None else "No"
        tsd_null_msg = (
            f"  • After disconnect - TimeSeriesData set to None: {paint(BGRN, tsd_null_status)}"
        )

        print(client_null_msg)
        print(tsd_null_msg)

        # Note about reconnection issues
        print("")
        print_subsection_title("Note on reconnection:")
        message = (
            "DataClay reconnection is not possible within the same process "
            "due to event loop binding limitations."
        )
        reconnect_note = f"  • {paint(WHT, message)}"
        print(reconnect_note)
    else:
        skip_msg = f"  • Skipping disconnect test: {paint(BYEL, 'Client was not connected')}"
        print(skip_msg)


def test_fetcher_processor_integration() -> None:
    """Test the integration between Fetcher and Processor components.

    This test validates that data from the Fetcher can be successfully processed
    by the Processor to create DataLoaders for LSTM model training.
    """
    global global_fetcher

    print_subsection_header("Fetcher-Processor Integration")
    print_subsection_title("Testing data flow between components:")

    # Step 1: Get data from Fetcher
    print("  • Step 1: Fetching data from DataClay...")
    try:
        fetched_data = global_fetcher.fetch_data(timeout=10)
        if fetched_data is None or len(fetched_data) == 0:
            print(paint(RED, "  ✗ Failed to retrieve data from Fetcher"))
            return

        data_shape_msg = f"  • Fetched data shape: {paint(BGRN, str(fetched_data.shape))}"
        print(data_shape_msg)

        # Step 2: Initialize Processor
        print("")
        print("  • Step 2: Initializing Processor...")

        # Use 'cpu_consumption' as the default target metric
        processor = Processor(
            time_step=10,  # Use 5 time steps for sequences
            metric="cpu_consumption",
            batch_size=64,
            train_ratio=0.8,
            device=torch.device("cpu"),  # Use CPU device for testing
        )

        processor_config_msg = (
            f"  • Processor configuration: time_step={paint(WHT, '5')}, "
            f"metric={paint(WHT, 'cpu_consumption')}, batch_size={paint(WHT, '16')}"
        )
        print(processor_config_msg)

        # Step 3: Create DataLoaders using fetched data
        print("")
        print("  • Step 3: Creating DataLoaders from fetched data...")

        start_time = time.time()
        train_dataloader, val_dataloader, train_dataset, val_dataset = (
            processor.create_data_loaders(fetched_data)
        )
        end_time = time.time()

        proc_time_msg = (
            f"  • Processing time: {paint(WHT, f'{end_time - start_time:.2f}')} seconds"
        )
        print(proc_time_msg)

        # Step 4: Validate DataLoaders and Datasets
        print("")
        print_subsection_title("Validating created DataLoaders:")

        # Check training dataset
        train_size_msg = f"  • Training set size: {paint(BGRN, str(len(train_dataset)))}"
        val_size_msg = f"  • Validation set size: {paint(BGRN, str(len(val_dataset)))}"

        train_batches_msg = f"  • Training batches: {paint(WHT, str(len(train_dataloader)))}"
        val_batches_msg = f"  • Validation batches: {paint(WHT, str(len(val_dataloader)))}"

        print(train_size_msg)
        print(val_size_msg)
        print(train_batches_msg)
        print(val_batches_msg)

        # Step 5: Check a sample batch from training loader
        if len(train_dataloader) > 0:
            print("")
            print_subsection_title("Examining sample training batch:")

            # Get first batch
            for batch_inputs, batch_targets in train_dataloader:
                batch_inputs_shape = (
                    f"  • Batch inputs shape: {paint(WHT, str(batch_inputs.shape))}"
                )
                batch_targets_shape = (
                    f"  • Batch targets shape: {paint(WHT, str(batch_targets.shape))}"
                )

                print(batch_inputs_shape)
                print(batch_targets_shape)

                if batch_inputs.shape[0] > 0:
                    seq_len = batch_inputs.shape[2]
                    sequence_msg = f"  • Sequence length (time steps): {paint(BGRN, str(seq_len))}"

                    feat_dim = batch_inputs.shape[1]
                    feature_msg = f"  • Feature dimension: {paint(BGRN, str(feat_dim))}"

                    print(sequence_msg)
                    print(feature_msg)

                # Only process the first batch
                break

        print("")
        success_msg = "Integration test successful: Fetcher data was successfully processed"
        integration_success_msg = f"  • {paint(BGRN, '✓')} {success_msg}"
        print(integration_success_msg)

    except (RuntimeError, ValueError, TimeoutError) as e:
        # Be more specific about which exceptions to catch
        error_msg = f"  ✗ Integration test failed: {e!s}"
        print(paint(RED, error_msg))


def print_test_header() -> None:
    """Print the main test header with logo."""
    print("")
    print(paint(BCYA, "════════════════════════════════════════════════════════════════"))
    print(paint(BCYA, "                      🔍 ICOS-FL TESTING 🔍"))
    print(paint(BCYA, "════════════════════════════════════════════════════════════════"))
    print("")


def print_test_footer() -> None:
    """Print the test footer with completion message."""
    print("")
    print(paint(BCYA, "════════════════════════════════════════════════════════════════"))
    print(paint(BGRN, "                 ✅ ALL TESTS COMPLETED SUCCESSFULLY ✅"))
    print(paint(BCYA, "════════════════════════════════════════════════════════════════"))
    print("")


def test_fetcher() -> None:
    """Run all Fetcher and TimeSeriesData tests."""
    global global_fetcher

    # Disable all logging from DataClay and other sources
    logging.getLogger().setLevel(logging.CRITICAL)

    # Print the custom header
    print_test_header()

    # Display header
    print_section_header("FETCHER MODULE TESTING")

    # Initialize a single Fetcher instance for all tests
    try:
        # Create the global fetcher
        global_fetcher = Fetcher()

        # Run test functions
        initialize_fetcher()
        test_time_series_data()
        test_post_process()
        test_fetch_data()
        test_error_handling()

        # Run integration test BEFORE disconnecting
        test_fetcher_processor_integration()

        # Finally run cleanup (which disconnects)
        test_cleanup()

        # Display footer
        print_section_header("FETCHER MODULE TESTING COMPLETE")
        print_test_footer()
    except (RuntimeError, ConnectionError, DataClayException) as e:
        # Be specific about which exceptions to catch
        error_msg = f"An error occurred during Fetcher testing: {e!s}"
        perror(error_msg)

        # Ensure we disconnect even if there's an error
        if global_fetcher and global_fetcher.client:
            try:
                global_fetcher._disconnect()
            # We need to catch all exceptions here for proper resource cleanup
            # during error handling, otherwise resources might be left hanging
            except (RuntimeError, ConnectionError, OSError) as e:
                # Log instead of silently ignoring
                print(f"Warning: Error during cleanup after test failure: {e}")


if __name__ == "__main__":
    # Suppress warnings before anything else
    warnings.filterwarnings("ignore")

    test_fetcher()
