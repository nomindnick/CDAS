#!/usr/bin/env python
"""
Script to run all integration tests for the CDAS system.

Usage:
    python tests/run_integration_tests.py

Options:
    --verbose   Show verbose output
    --coverage  Generate coverage report
"""

import sys
import subprocess
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run CDAS integration tests")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    return parser.parse_args()


def run_tests(verbose=False, coverage=False):
    """Run the integration tests."""
    # Base command
    cmd = ["pytest", "tests/test_integration/"]
    
    # Add options
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=cdas", "--cov-report=term-missing"])
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    
    return result.returncode


def main():
    """Main entry point."""
    args = parse_args()
    
    # Run the tests
    exit_code = run_tests(verbose=args.verbose, coverage=args.coverage)
    
    # Exit with the same code
    sys.exit(exit_code)


if __name__ == "__main__":
    main()