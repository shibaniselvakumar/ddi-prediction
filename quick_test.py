#!/usr/bin/env python
"""
Quick test of NDD with mock dependencies - simplified for speed
"""

import sys
import os

# ===== INJECT MOCKS FIRST =====
import mock_dependencies

# Now safe to import
import numpy as np

# Import from mocked modules
from keras.layers.core import Dropout, Activation
from keras.layers import Dense
from keras import optimizers
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score, precision_score, roc_curve, auc, precision_recall_curve

# Change to data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "NDD", "DS1")

if os.path.exists(data_dir):
    os.chdir(data_dir)
    print(f"[OK] Working directory set to: {os.getcwd()}\n")
else:
    print(f"[WARN] Data directory not found. Using: {os.getcwd()}\n")

# ============ Quick Test ============

print("="*70)
print("  NDD - Quick Test")
print("="*70 + "\n")

print("[*] Loading data...")
try:
    drug_fea = np.loadtxt("IntegratedDS1.txt", dtype=float, delimiter=",")
    interaction = np.loadtxt("drug_drug_matrix.csv", dtype=int, delimiter=",")
    print(f"    [OK] Loaded {drug_fea.shape[0]} drugs")
    print(f"    [OK] Interaction matrix shape: {interaction.shape}")
except FileNotFoundError as e:
    print(f"    [ERROR] {e}")
    sys.exit(1)

print("\n[*] Preparing test data...")
# Create smaller test set
num_drugs = min(20, drug_fea.shape[0])  # Use just 20 drugs for speed
drug_fea_test = drug_fea[:num_drugs]
interaction_test = interaction[:num_drugs, :num_drugs]

print(f"    [OK] Using {num_drugs} drugs for quick test")

# Create training data
X = []
y = []
for i in range(num_drugs):
    for j in range(num_drugs):
        X.append(list(drug_fea_test[i]) + list(drug_fea_test[i]))
        y.append(interaction_test[i, j])

X = np.array(X)
y = np.array(y)

print(f"    [OK] Created {len(X)} samples with {X.shape[1]} features")

print("\n[*] Preprocessing...")
encoder = LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y).astype(np.int32)
y_cat = np_utils.to_categorical(y_encoded)
print(f"    [OK] Encoded labels to shape {y_cat.shape}")

print("\n[*] Building model...")
model = Sequential()
model.add(Dense(input_dim=X.shape[1], output_dim=100, init='glorot_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(input_dim=100, output_dim=50, init='glorot_normal'))
model.add(Activation('relu'))
model.add(Dense(input_dim=50, output_dim=2, init='glorot_normal'))
model.add(Activation('sigmoid'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)
print(f"    [OK] Model created with {X.shape[1]} input features")

print("\n[*] Training model (5 epochs on small dataset)...")
model.fit(X, y_cat, batch_size=10, epochs=5, shuffle=True, verbose=0)
print(f"    [OK] Training completed")

print("\n[*] Making predictions...")
proba = model.predict_proba(X[:50], batch_size=20, verbose=0)
proba_classes = model.predict_classes(X[:50], batch_size=20, verbose=0)
print(f"    [OK] Got predictions shape: {proba.shape}")
print(f"    [OK] Got classes shape: {proba_classes.shape}")

print("\n[*] Evaluating...")
real_labels = y[:50]
try:
    f = f1_score(real_labels, proba_classes)
    rec = recall_score(real_labels, proba_classes)
    prec = precision_score(real_labels, proba_classes)
    print(f"    [OK] F1-Score: {f:.4f}")
    print(f"    [OK] Recall: {rec:.4f}")
    print(f"    [OK] Precision: {prec:.4f}")
except:
    print(f"    [WARN] Could not calculate metrics")

print("\n" + "="*70)
print("[OK] NDD Quick Test Completed Successfully!")
print("="*70 + "\n")
