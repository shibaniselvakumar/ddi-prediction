#!/usr/bin/env python
"""
This script tries to install dependencies in a compatible way
"""
import subprocess
import sys

packages = [
    "numpy",
    "matplotlib",
]

for pkg in packages:
    print(f"\n=== Installing {pkg} ===")
    result = subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--pre"], 
                          capture_output=False)
    if result.returncode != 0:
        print(f"Warning: Failed to install {pkg}")

# Try importing and provide feedback
print("\n=== Checking installed packages ===")
try:
    import numpy
    print(f"✓ numpy: {numpy.__version__}")
except:
    print("✗ numpy: NOT installed")

try:
    import matplotlib
    print(f"✓ matplotlib: {matplotlib.__version__}")
except:
    print("✗ matplotlib: NOT installed")

try:
    import keras
    print(f"✓ keras: {keras.__version__}")
except:
    print("✗ keras: NOT installed")

try:
    import sklearn
    print(f"✓ scikit-learn: {sklearn.__version__}")
except:
    print("✗ scikit-learn: NOT installed")
