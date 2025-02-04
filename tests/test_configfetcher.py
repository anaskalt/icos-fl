#!/usr/bin/env python3
"""Test module for ConfigFetcher"""

import sys
import os
import tempfile
import yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from icos_fl.utils.configfetcher import ConfigFetcher, ConfigFetchError
from icos_fl.utils.colors import paint, BCYA, FGRY, BWHT, ERR_CLR, INF_CLR
from icos_fl.utils.logger import ICOSLogger

LOG = ICOSLogger("ConfigTest")

SERVER_CONFIG = """
icos:
  federation:
    address: "0.0.0.0"
    port: 8080
    min_clients: 3
    rounds: 100
    timeout: 300
  logging:
    level: "info"
    file: "/var/log/icos-fl/server.log"
"""

CLIENT_CONFIG = """
icos:
  server:
    address: "server"
    port: 8080
  data:
    host: "dataclay://metrics-cluster"
    dataset: "metrics-dataset"
    timeout: 300
"""

TEST_CASES = [
    # (content, should_fail, expected_keys)
    (SERVER_CONFIG, False, ['federation', 'logging']),
    (CLIENT_CONFIG, False, ['server', 'data']),
    ("invalid_yaml: [", True, None),
    ("key_without_icos: value", True, None),
    ("icos:\n  security:\n    ssl: true", False, ['security']),
]

class ConfigFetcherTester:
    def __init__(self):
        self.total = 0
        self.failures = 0
        self.temp_files = []
        
    def _cleanup(self):
        for f in self.temp_files:
            if os.path.exists(f):
                os.remove(f)
                
    def _create_temp_config(self, content):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(content)
            self.temp_files.append(f.name)
            return f.name
            
    def _print_header(self):
        print(paint(BWHT, "\n┌────────────────────────────────────────┐"))
        print(paint(BWHT, "│      Testing ConfigFetcher Module      │"))
        print(paint(BWHT, "└────────────────────────────────────────┘"))
        
    def _print_footer(self):
        print(paint(BWHT, "\n►► Test Summary:"))
        print(paint(FGRY, f"Total cases: {self.total}"))
        print(paint(INF_CLR, f"Passed: {self.total - self.failures}"))
        print(paint(ERR_CLR if self.failures > 0 else INF_CLR, 
                  f"Failed: {self.failures}"))
    
    def _run_case(self, test_num, content, should_fail, expected_keys):
        self.total += 1
        tmp_file = self._create_temp_config(content)
        
        try:
            print(paint(BCYA, f"\n● Test Case {test_num}/{len(TEST_CASES)}"))
            print(paint(FGRY, "Config Content:"))
            print(content.strip())
            
            fetcher = ConfigFetcher(tmp_file)
            config = fetcher.get_configuration()
            
            if should_fail:
                self.failures += 1
                return paint(ERR_CLR, "✘ FAIL: Expected error but succeeded")
                
            missing = [k for k in expected_keys if k not in config]
            if missing:
                self.failures += 1
                return paint(ERR_CLR, f"✘ Missing keys: {missing}")
                
            return paint(INF_CLR, "✓ PASS") + "\n" + paint(FGRY, str(config))
            
        except ConfigFetchError as e:
            if not should_fail:
                self.failures += 1
                return paint(ERR_CLR, f"✘ UNEXPECTED ERROR: {str(e)}")
            return paint(INF_CLR, f"✓ Expected error: {str(e)}")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ CRITICAL ERROR: {str(e)}")
            
    def run(self):
        self._print_header()
        
        for idx, (content, should_fail, keys) in enumerate(TEST_CASES):
            result = self._run_case(idx+1, content, should_fail, keys)
            print(result)
            
        self._print_footer()
        self._cleanup()
        sys.exit(1 if self.failures > 0 else 0)

if __name__ == "__main__":
    tester = ConfigFetcherTester()
    tester.run()