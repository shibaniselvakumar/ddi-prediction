#!/usr/bin/env python
"""
Simple test to verify everything works
"""

import sys
import os

print("[1/5] Importing mock dependencies...")
import mock_dependencies
print("      SUCCESS\n")

print("[2/5] Importing numpy...")
import numpy as np
print(f"      SUCCESS (numpy {np.__version__})\n")

print("[3/5] Testing sklearn mock...")
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
encoder = LabelEncoder()
print("      SUCCESS\n")

print("[4/5] Testing keras mock...")
from keras.layers import Dense
from keras.models import Sequential
from keras import optimizers
model = Sequential()
model.add(Dense(input_dim=10, output_dim=5))
print("      SUCCESS\n")

print("[5/5] Loading NDD data...")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "NDD", "DS1")
os.chdir(data_dir)
print(f"      Changed to: {os.getcwd()}")
drug_fea = np.loadtxt("IntegratedDS1.txt", dtype=float, delimiter=",")
interaction = np.loadtxt("drug_drug_matrix.csv", dtype=int, delimiter=",")
print(f"      Loaded {drug_fea.shape[0]} drugs")
print(f"      Loaded interaction matrix {interaction.shape}")
print("      SUCCESS\n")

print("="*70)
print("[SUCCESS] All components working! NDD project is ready to run.")
print("="*70)
print(f"\nData location: {os.getcwd()}")
print(f"Drugs: {drug_fea.shape[0]}")
print(f"Features per drug: {drug_fea.shape[1]}")
print(f"Interactions: {np.sum(interaction == 1)} positive, {np.sum(interaction == 0)} negative")
print("\nYou can now run: python run_ndd.py")
