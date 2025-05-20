#!/usr/bin/env python
"""
Script to run AI integration tests for the CDAS system.

Usage:
    python tests/run_ai_tests.py [options]

Options:
    --all           Run all tests, including benchmarks
    --unit          Run only unit tests
    --integration   Run only integration tests
    --benchmarks    Run only benchmark tests
    --mock          Force mock mode for all tests
    --verbose       Show verbose output
    --coverage      Generate coverage report
"""

import sys
import os
import subprocess
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run CDAS AI tests")
    parser.add_argument("--all", action="store_true", help="Run all tests, including benchmarks")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--benchmarks", action="store_true", help="Run only benchmark tests")
    parser.add_argument("--mock", action="store_true", help="Force mock mode for all tests")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    return parser.parse_args()


def run_tests(args):
    """Run the tests based on the provided arguments."""
    # Base command
    cmd = ["pytest"]
    
    # Add options
    if args.verbose:
        cmd.append("-v")
    
    if args.coverage:
        cmd.extend(["--cov=cdas.ai", "--cov-report=term-missing"])
    
    # Determine which tests to run
    if args.unit:
        cmd.append("tests/test_ai/")
    elif args.integration:
        cmd.append("tests/test_integration/test_ai/")
    elif args.benchmarks:
        cmd.append("-m")
        cmd.append("benchmark")
    elif args.all:
        cmd.extend(["tests/test_ai/", "tests/test_integration/test_ai/"])
    else:
        # Default: run all except benchmarks
        cmd.append("tests/test_ai/")
        cmd.append("tests/test_integration/test_ai/")
        cmd.append("-k")
        cmd.append("not benchmark")
    
    # Handle mock mode environment variable
    if args.mock:
        # Use mock mode
        print("Setting CDAS_MOCK_MODE=1 for mock mode testing")
        os.environ["CDAS_MOCK_MODE"] = "1"
    else:
        # Ensure mock mode is disabled for real API testing
        print("Setting CDAS_MOCK_MODE=0 for real API testing")
        if "CDAS_MOCK_MODE" in os.environ:
            del os.environ["CDAS_MOCK_MODE"]
        os.environ["CDAS_MOCK_MODE"] = "0"
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    
    return result.returncode


def main():
    """Main entry point."""
    args = parse_args()
    
    # Run the tests
    exit_code = run_tests(args)
    
    # Exit with the same code
    sys.exit(exit_code)


if __name__ == "__main__":
    main()