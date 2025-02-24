#!/usr/bin/env python3
"""Test module for DataFetcher with custom mock implementations"""

import sys
import os
import pandas as pd
import numpy as np
from collections import deque
from threading import Event
import threading
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from icos_fl.data.fetcher import DataFetcher, TimeSeriesData, MetricsMapping
from icos_fl.utils.colors import paint, BCYA, FGRY, BWHT, ERR_CLR, INF_CLR

# Custom Mock Classes
class MockDataClayException(Exception):
    """Mock of DataClay's exception"""
    pass

class MockClient:
    """Mock implementation of DataClay Client."""
    def __init__(self, proxy_host: str, dataset: str):
        self.proxy_host = proxy_host
        self.dataset = dataset
        self.connected = False
        
    def start(self):
        self.connected = True

class MockTimeSeriesData:
    """Mock implementation of TimeSeriesData."""
    _instances = {}  # Store instances by alias
    
    def __init__(self):
        self.dataframes = deque(maxlen=5)
        self.waiters = []
        
    def add_dataframe(self, df: pd.DataFrame) -> None:
        self.dataframes.append(df)
        for waiter in self.waiters:
            waiter.set()
            
    def get_last_dataframe(self) -> pd.DataFrame:
        if self.dataframes:
            return self.dataframes[-1]
        return None
        
    def wait_for_dataframe(self, timeout=None) -> pd.DataFrame:
        waiter = Event()
        self.waiters.append(waiter)
        waiter.wait(timeout=timeout)
        self.waiters.remove(waiter)
        return self.get_last_dataframe()
        
    def get_all_dataframes(self) -> list:
        return list(self.dataframes)
        
    def make_persistent(self, alias: str):
        MockTimeSeriesData._instances[alias] = self
        
    @classmethod
    def get_by_alias(cls, alias: str):
        if alias in cls._instances:
            return cls._instances[alias]
        raise MockDataClayException("Not found")

class DataFetcherTester:
    def __init__(self):
        self.total = 0
        self.failures = 0
        
        # Test parameters (from config)
        self.host = "dataclay://metrics-cluster"
        self.dataset = "metrics-dataset"
        self.alias = "metrics-timeseries"
        self.poll_interval = 60
        self.timeout = 300
        self.target_metric = "cpu_usage"
        self.min_samples = 100
        
    def _print_header(self):
        print(paint(BWHT, "\n┌────────────────────────────────────────┐"))
        print(paint(BWHT, "│        Testing DataFetcher Module       │"))
        print(paint(BWHT, "└────────────────────────────────────────┘"))
        
    def _print_footer(self):
        print(paint(BWHT, "\n►► Test Summary:"))
        print(paint(FGRY, f"Total tests: {self.total}"))
        print(paint(INF_CLR, f"Passed: {self.total - self.failures}"))
        print(paint(ERR_CLR if self.failures > 0 else INF_CLR, 
                  f"Failed: {self.failures}"))
    
    def _generate_sample_data(self, rows=10):
        """Generate sample metrics data."""
        return pd.DataFrame({
            'scaph_host_power_microwatts': np.random.uniform(100000, 200000, rows),
            'scaph_process_cpu_usage_percentage': np.random.uniform(0, 100, rows),
            'scaph_host_memory_total_bytes': np.full(rows, 16e9),  # 16GB total
            'scaph_host_memory_available_bytes': np.random.uniform(4e9, 8e9, rows)
        })
    
    def _test_connection_and_initialization(self):
        self.total += 1
        try:
            print(paint(BCYA, "\n● Test 1: Connection & Initialization"))
            
            # Store original imports
            import icos_fl.data.fetcher as fetcher_module
            original_client = fetcher_module.Client
            original_timeseries = fetcher_module.TimeSeriesData
            original_exception = fetcher_module.DataClayException
            
            # Replace with our mocks
            fetcher_module.Client = MockClient
            fetcher_module.TimeSeriesData = MockTimeSeriesData
            fetcher_module.DataClayException = MockDataClayException
            
            try:
                # Create and test fetcher
                fetcher = DataFetcher(
                    host=self.host,
                    dataset=self.dataset,
                    alias=self.alias,
                    target_metric=self.target_metric,
                    poll_interval=self.poll_interval,
                    timeout=self.timeout,
                    min_samples=self.min_samples
                )
                
                # Test connection
                assert fetcher.connect()
                assert fetcher.client.connected
                
                # Test timeseries initialization (new)
                assert fetcher.initialize_timeseries()
                assert isinstance(fetcher.tsd, MockTimeSeriesData)
                
            finally:
                # Restore original classes
                fetcher_module.Client = original_client
                fetcher_module.TimeSeriesData = original_timeseries
                fetcher_module.DataClayException = original_exception
                MockTimeSeriesData._instances.clear()
            
            return paint(INF_CLR, "✓ PASS")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ FAIL: {str(e)}")
    
    def _test_metrics_processing(self):
        self.total += 1
        try:
            print(paint(BCYA, "\n● Test 2: Metrics Processing"))
            
            df = self._generate_sample_data(1)
            
            # Test each metric type
            metrics = ['power_consumption', 'cpu_usage', 'ram_usage']
            for metric in metrics:
                value = MetricsMapping.process_metric(df, metric)
                assert isinstance(value, float)
                
                if metric == 'cpu_usage':
                    assert 0 <= value <= 100
                elif metric == 'ram_usage':
                    assert value > 0 and value < 16e9
            
            # Test invalid metric
            try:
                MetricsMapping.process_metric(df, 'invalid_metric')
                assert False, "Should raise ValueError"
            except ValueError:
                pass
                
            return paint(INF_CLR, "✓ PASS")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ FAIL: {str(e)}")
    
    def _test_data_collection(self):
        self.total += 1
        try:
            print(paint(BCYA, "\n● Test 3: Data Collection"))
            
            # Setup mock environment
            mock_tsd = MockTimeSeriesData()
            fetcher = DataFetcher(
                host=self.host,
                dataset=self.dataset,
                alias=self.alias,
                target_metric=self.target_metric,
                min_samples=5  # Lower for testing
            )
            fetcher.tsd = mock_tsd
            
            # Test initial state
            assert len(fetcher.metrics_buffer) == 0
            
            # Test data fetching
            df = self._generate_sample_data(1)
            mock_tsd.add_dataframe(df)
            
            value = fetcher.fetch_metrics()
            assert isinstance(value, float)
            assert len(fetcher.metrics_buffer) == 1
            
            # Test async data collection
            def add_data():
                time.sleep(0.1)
                mock_tsd.add_dataframe(self._generate_sample_data(1))
                
            thread = threading.Thread(target=add_data)
            thread.start()
            value = fetcher.fetch_metrics()
            thread.join()
            
            assert isinstance(value, float)
            assert len(fetcher.metrics_buffer) == 2
            
            return paint(INF_CLR, "✓ PASS")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ FAIL: {str(e)}")
    
    def _test_data_validation(self):
        self.total += 1
        try:
            print(paint(BCYA, "\n● Test 4: Data Validation"))
            
            # Test with valid data
            df = self._generate_sample_data()
            fetcher = DataFetcher(
                host=self.host,
                dataset=self.dataset,
                alias=self.alias,
                target_metric=self.target_metric
            )
            
            validated = fetcher._validate_data(df)
            assert isinstance(validated, pd.DataFrame)
            assert not validated.isna().any().any()
            
            # Test with missing columns
            bad_df = pd.DataFrame({'wrong_column': [1, 2, 3]})
            try:
                fetcher._validate_data(bad_df)
                assert False, "Should raise ValueError"
            except ValueError:
                pass
            
            return paint(INF_CLR, "✓ PASS")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ FAIL: {str(e)}")
    
    def run(self):
        self._print_header()
        
        print(self._test_connection_and_initialization())
        print(self._test_metrics_processing())
        print(self._test_data_collection())
        print(self._test_data_validation())
        
        self._print_footer()
        sys.exit(1 if self.failures > 0 else 0)

if __name__ == "__main__":
    tester = DataFetcherTester()
    tester.run()