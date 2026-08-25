"""Verify the software environment for the EEF1A Hetnet project."""

import sys
from importlib.metadata import PackageNotFoundError, version

import hetmatpy
import hetnetpy
import networkx
import numpy
import pandas
import scipy
import yaml


PACKAGES = [
    "hetnetpy",
    "hetmatpy",
    "numpy",
    "scipy",
    "pandas",
    "networkx",
    "PyYAML",
    "pytest",
    "ipykernel",
]


def installed_version(package_name: str) -> str:
    """Return the installed version of a Python package."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


print("=" * 60)
print("EEF1A HETNET ENVIRONMENT CHECK")
print("=" * 60)

print(f"\nPython executable: {sys.executable}")
print(f"Python version: {sys.version}")

print("\nInstalled packages:")

for package in PACKAGES:
    print(f"  {package}: {installed_version(package)}")

print("\nImport checks completed successfully.")
print("=" * 60)
