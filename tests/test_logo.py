#!/usr/bin/env python3
"""Test module for ASCII art logos - Visual check"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from icos_fl.utils import logo
from icos_fl.utils.colors import paint, BCYA, FGRY, BWHT, ERR_CLR, INF_CLR

class LogoTester:
    def __init__(self):
        self.total = 0
        self.failures = 0
        
        # Expected logos
        self.expected_logos = [
            'LOGO1',
            'OBJECT1',
            'OBJECT2',
            'OBJECT3',
            'NEURAL_NET',
            'FEDERATED_ARCH',
            'ICOS_FL_LOGO',
            'CLOUD_SYSTEM',
            'ROBOT_ICS'
        ]
        
    def _print_header(self):
        print(paint(BWHT, "\n┌────────────────────────────────────────┐"))
        print(paint(BWHT, "│          Testing Logo Module           │"))
        print(paint(BWHT, "└────────────────────────────────────────┘"))
        
    def _print_footer(self):
        print(paint(BWHT, "\n►► Test Summary:"))
        print(paint(FGRY, f"Total tests: {self.total}"))
        print(paint(INF_CLR, f"Passed: {self.total - self.failures}"))
        print(paint(ERR_CLR if self.failures > 0 else INF_CLR, 
                  f"Failed: {self.failures}"))
    
    def _test_print_logos(self):
        self.total += 1
        try:
            print(paint(BCYA, "\n● Test 1: Logo Visual Check"))
            
            # Check if all expected logos exist and print them
            for logo_name in self.expected_logos:
                if not hasattr(logo, logo_name):
                    raise AttributeError(f"Missing logo: {logo_name}")
                    
                logo_str = getattr(logo, logo_name)
                if not isinstance(logo_str, str):
                    raise TypeError(f"Logo {logo_name} is not a string")
                
                print(paint(BWHT, f"\n[{logo_name}]"))
                print(paint(FGRY, logo_str))
            
            return paint(INF_CLR, "✓ PASS - All logos printed successfully")
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ FAIL: {str(e)}")
    
    def run(self):
        self._print_header()
        print(self._test_print_logos())
        self._print_footer()
        sys.exit(1 if self.failures > 0 else 0)

if __name__ == "__main__":
    tester = LogoTester()
    tester.run()