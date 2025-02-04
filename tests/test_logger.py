#!/usr/bin/env python3
"""Test module for ICOSLogger"""

import sys
import os
import time
import random
import threading
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from icos_fl.utils.logger import ICOSLogger
from icos_fl.utils.colors import paint, BCYA, FGRY, BWHT, ERR_CLR, INF_CLR

class LoggerTester:
    def __init__(self):
        self.test_dir = "/tmp/icos_logger_test"
        self.total = 0
        self.failures = 0
        
    def _print_header(self):
        print(paint(BWHT, "\n┌────────────────────────────────────────┐"))
        print(paint(BWHT, "│        Testing ICOSLogger Module       │"))
        print(paint(BWHT, "└────────────────────────────────────────┘"))
        
    def _print_footer(self):
        print(paint(BWHT, "\n►► Test Summary:"))
        print(paint(FGRY, f"Total tests: {self.total}"))
        print(paint(INF_CLR, f"Passed: {self.total - self.failures}"))
        print(paint(ERR_CLR if self.failures > 0 else INF_CLR, 
                  f"Failed: {self.failures}"))
    
    def _stress_worker(self, logger, worker_id):
        for i in range(100):
            logger.debug(f"Worker{worker_id} debug {i}")
            logger.info(f"Worker{worker_id} info {i}")
            time.sleep(random.uniform(0.001, 0.01))
    
    def _test_basic_logging(self):
        self.total += 1
        try:
            print(paint(BCYA, "\n● Test 1: Basic Logging"))
            logger = ICOSLogger("basic_test", 
                              usefile=True, 
                              logdir=self.test_dir,
                              rotmaxbytes=1024)
                              
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
            
            return paint(INF_CLR, "✓ PASS")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ FAIL: {str(e)}")
    
    def _test_rotation(self):
        self.total += 1
        try:
            print(paint(BCYA, "\n● Test 2: File Rotation"))
            logger = ICOSLogger("rotation_test", 
                              usefile=True, 
                              logdir=self.test_dir,
                              rotmaxbytes=1024)
                              
            for _ in range(150):
                logger.info("Rotation test " + "X"*50)
                
            log_files = os.listdir(self.test_dir)
            if len(log_files) < 2:
                self.failures += 1
                return paint(ERR_CLR, "✘ FAIL: Rotation not detected")
                
            return paint(INF_CLR, "✓ PASS")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ FAIL: {str(e)}")
    
    def _test_thread_safety(self):
        self.total += 1
        try:
            print(paint(BCYA, "\n● Test 3: Thread Safety"))
            logger = ICOSLogger("thread_test", 
                              usefile=True, 
                              logdir=self.test_dir)
                              
            threads = []
            for i in range(5):
                t = threading.Thread(target=self._stress_worker, args=(logger, i))
                threads.append(t)
                t.start()
                
            for t in threads:
                t.join()
                
            main_log = os.path.join(self.test_dir, "thread_test.log")
            with open(main_log) as f:
                lines = f.readlines()
                if len(lines) < 500:
                    self.failures += 1
                    return paint(ERR_CLR, f"✘ FAIL: Missing logs ({len(lines)} lines)")
                    
            return paint(INF_CLR, "✓ PASS")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ FAIL: {str(e)}")
    
    def run(self):
        self._print_header()
        
        if os.path.exists(self.test_dir):
            os.system(f"rm -rf {self.test_dir}")
        os.makedirs(self.test_dir)
        
        print(self._test_basic_logging())
        print(self._test_rotation())
        print(self._test_thread_safety())
        
        self._print_footer()
        sys.exit(1 if self.failures > 0 else 0)

if __name__ == "__main__":
    tester = LoggerTester()
    tester.run()