#!/usr/bin/env python
"""
Script to set up testing environment for CDAS.

This script:
1. Creates a virtual environment for testing
2. Installs all dependencies
3. Runs preliminary tests to verify setup

Usage:
    python scripts/setup_testing.py [--verbose]
"""

import os
import sys
import argparse
import subprocess
import platform
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Set up CDAS testing environment")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    return parser.parse_args()


def run_command(cmd, verbose=False, check=True):
    """Run a shell command."""
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=not verbose,
        text=True,
        check=check
    )
    
    return result


def create_venv(venv_path, verbose=False):
    """Create a virtual environment."""
    if venv_path.exists():
        if verbose:
            print(f"Virtual environment already exists at {venv_path}")
        return
    
    print(f"Creating virtual environment at {venv_path}")
    run_command([sys.executable, "-m", "venv", str(venv_path)], verbose=verbose)


def install_dependencies(venv_path, verbose=False):
    """Install dependencies into the virtual environment."""
    # Determine the pip executable path
    if platform.system() == "Windows":
        pip_path = venv_path / "Scripts" / "pip"
    else:
        pip_path = venv_path / "bin" / "pip"
    
    print("Upgrading pip...")
    run_command([str(pip_path), "install", "--upgrade", "pip"], verbose=verbose)
    
    print("Installing development dependencies...")
    run_command([str(pip_path), "install", "-r", "requirements-dev.txt"], verbose=verbose)
    
    print("Installing project in development mode...")
    run_command([str(pip_path), "install", "-e", "."], verbose=verbose)


def run_tests(venv_path, verbose=False):
    """Run a simple test to verify the setup."""
    # Determine the pytest executable path
    if platform.system() == "Windows":
        pytest_path = venv_path / "Scripts" / "pytest"
    else:
        pytest_path = venv_path / "bin" / "pytest"
    
    print("Running a simple test to verify setup...")
    test_cmd = [str(pytest_path), "tests/test_db", "-v"]
    result = run_command(test_cmd, verbose=verbose, check=False)
    
    if result.returncode != 0:
        print("WARNING: Initial test failed. Check the output for errors.")
    else:
        print("Initial test passed!")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get project root directory
    root_dir = Path(__file__).parent.parent.absolute()
    os.chdir(root_dir)
    
    # Path to virtual environment
    venv_path = root_dir / "venv-test"
    
    # Create virtual environment
    create_venv(venv_path, verbose=args.verbose)
    
    # Install dependencies
    install_dependencies(venv_path, verbose=args.verbose)
    
    # Run tests
    run_tests(venv_path, verbose=args.verbose)
    
    print("\nSetup complete! You can now run tests with:")
    if platform.system() == "Windows":
        print(f"  {venv_path}\\Scripts\\pytest")
    else:
        print(f"  {venv_path}/bin/pytest")
    
    print("\nOr use the Makefile commands:")
    print("  make test")
    print("  make coverage")


if __name__ == "__main__":
    main()