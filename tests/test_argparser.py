#!/usr/bin/env python3
"""Test module for ICOSArgParser"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from icos_fl.utils.argparser import ICOSArgParser
from icos_fl.utils.colors import paint, BCYA, FGRY, BWHT, ERR_CLR, INF_CLR

TEST_CASES = [
    # (component, args, expected_values)
    ('client', [], {
        'version': False,
        'quiet': False,
        'config': 'config/client.yaml',
        'dry_run': False
    }),
    ('server', ['-v', '-q'], {
        'version': True,
        'quiet': True,
        'config': 'config/server.yaml',
        'dry_run': False
    }),
    ('client', ['-c', 'custom.yaml', '-d'], {
        'version': False,
        'quiet': False,
        'config': 'custom.yaml',
        'dry_run': True
    }),
    ('server', ['--version', '--dry-run'], {
        'version': True,
        'quiet': False,
        'config': 'config/server.yaml',
        'dry_run': True
    })
]

class ArgParserTester:
    def __init__(self):
        self.total = 0
        self.failures = 0
        self.log = []

    def _print_header(self):
        print(paint(BWHT, "\n┌────────────────────────────────────────┐"))
        print(paint(BWHT, "│      Testing ICOSArgParser Module      │"))
        print(paint(BWHT, "└────────────────────────────────────────┘"))

    def _print_footer(self):
        print(paint(BWHT, "\n►► Test Summary:"))
        print(paint(FGRY, f"Total cases: {self.total}"))
        print(paint(INF_CLR, f"Passed: {self.total - self.failures}"))
        print(paint(ERR_CLR if self.failures > 0 else INF_CLR, 
                  f"Failed: {self.failures}"))
        
    def _validate_case(self, component, args, expected):
        self.total += 1
        try:
            parser = ICOSArgParser(f"{component} description", component)
            result = parser.parser.parse_args(args)
            
            actual = {
                'version': result.version,
                'quiet': result.quiet,
                'config': result.config,
                'dry_run': result.dry_run
            }
            
            if actual != expected:
                self.failures += 1
                msg = paint(ERR_CLR, "✘ FAIL") + "\n" 
                msg += paint(FGRY, f"Expected: {expected}\nReceived: {actual}")
                return msg
                
            return paint(INF_CLR, "✓ PASS") + "\n" + paint(FGRY, str(parser))
            
        except Exception as e:
            self.failures += 1
            return paint(ERR_CLR, f"✘ ERROR: {str(e)}")

    def run(self):
        self._print_header()
        
        for idx, (component, args, expected) in enumerate(TEST_CASES):
            print(paint(BCYA, f"\n● Test Case {idx+1}/{len(TEST_CASES)}"))
            print(paint(FGRY, f"Component: {component}"))
            print(paint(FGRY, f"Arguments: {args}"))
            
            result = self._validate_case(component, args, expected)
            print(result)
            
        self._print_footer()
        sys.exit(1 if self.failures > 0 else 0)

if __name__ == "__main__":
    tester = ArgParserTester()
    tester.run()