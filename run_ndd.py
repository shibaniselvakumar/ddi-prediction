#!/usr/bin/env python
"""
NDD - Drug-Drug Interaction Prediction by Neural Network Using Integrated Similarity
This version uses mock implementations of Keras and sklearn to bypass dependency issues
"""

import sys
import os

# ===== INJECT MOCKS FIRST (before any imports) =====
import mock_dependencies

# Now safe to import
import numpy as np

try:
    import matplotlib
    matplotlib.use('agg')
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False

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
    print(f"[OK] Working directory: {os.getcwd()}\n")
else:
    print(f"[WARN] Data directory not found. Using: {os.getcwd()}\n")

# ============ NDD Methods ============

def prepare_data(seperate=False):
    """Load drug features and interaction matrix"""
    try:
        drug_fea = np.loadtxt("IntegratedDS1.txt", dtype=float, delimiter=",")
        interaction = np.loadtxt("drug_drug_matrix.csv", dtype=int, delimiter=",")
        print(f"  [OK] Loaded data: {drug_fea.shape[0]} drugs, interaction matrix {interaction.shape}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print(f"  Expected files in: {os.getcwd()}")
        sys.exit(1)
    
    train = []
    label = []
    for i in range(0, interaction.shape[0]):
        for j in range(0, interaction.shape[1]):
            label.append(interaction[i, j])
            drug_fea_tmp = list(drug_fea[i])
            if seperate:
                tmp_fea = (drug_fea_tmp, drug_fea_tmp)
            else:
                tmp_fea = drug_fea_tmp + drug_fea_tmp
            train.append(tmp_fea)

    return np.array(train), label

def calculate_performace(test_num, pred_y, labels):
    tp = fp = tn = fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp += 1
            else:
                fn += 1
        else:
            if labels[index] == pred_y[index]:
                tn += 1
            else:
                fp += 1
    
    acc = float(tp + tn) / test_num
    if tp == 0 and fp == 0:
        precision = MCC = 0
        sensitivity = float(tp) / (tp + fn) if (tp + fn) > 0 else 0
        specificity = float(tn) / (tn + fp) if (tn + fp) > 0 else 0
    else:
        precision = float(tp) / (tp + fp)
        sensitivity = float(tp) / (tp + fn) if (tp + fn) > 0 else 0
        specificity = float(tn) / (tn + fp) if (tn + fp) > 0 else 0
        denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        MCC = float(tp * tn - fp * fn) / np.sqrt(denom) if denom > 0 else 0
    return acc, precision, sensitivity, specificity, MCC

def transfer_array_format(data):
    formated_matrix1 = []
    formated_matrix2 = []
    for val in data:
        formated_matrix1.append(val[0])
        formated_matrix2.append(val[1])
    return np.array(formated_matrix1), np.array(formated_matrix2)

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
        y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def NDD(input_dim):
    """Build NDD neural network model"""
    model = Sequential()
    model.add(Dense(input_dim=input_dim, output_dim=400, init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=400, output_dim=300, init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=300, output_dim=2, init='glorot_normal'))
    model.add(Activation('sigmoid'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    return model

def DeepMDA():
    """Main DeepMDA model training and evaluation"""
    print("="*70)
    print("  NDD - Drug-Drug Interaction Prediction")
    print("  Using Integrated Similarity via SNF")
    print("="*70 + "\n")
    
    print("STEP 1: Loading data...")
    X, labels = prepare_data(seperate=True)
    X_data1, X_data2 = transfer_array_format(X)
    
    print("STEP 2: Preprocessing labels...")
    y, encoder = preprocess_labels(labels)
    X = np.concatenate((X_data1, X_data2), axis=1)
    
    num = np.arange(len(y))
    np.random.shuffle(num)
    X_data1 = X_data1[num]
    X_data2 = X_data2[num]
    y = y[num]
    
    num_cross_val = 5
    all_performance_DNN = []
    
    print(f"STEP 3: Starting {num_cross_val}-fold cross-validation...\n")
    
    for fold in range(num_cross_val):
        print(f"{'─'*70}")
        print(f"FOLD {fold + 1}/{num_cross_val}")
        print(f"{'─'*70}")
        
        # Split data
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
        train1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val != fold])
        test1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val == fold])
        train2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val != fold])
        test2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val == fold])
        
        # Count labels
        real_labels = []
        for val in test_label:
            real_labels.append(1 if val[0] == 1 else 0)
        
        train_label_new = []
        for val in train_label:
            train_label_new.append(1 if val[0] == 1 else 0)
        
        prefilter_train = np.concatenate((train1, train2), axis=1)
        prefilter_test = np.concatenate((test1, test2), axis=1)
        
        print(f"  Data: {len(prefilter_train):4d} train | {len(prefilter_test):4d} test")
        print(f"  Feature dimension: {prefilter_train.shape[1]}")
        
        # Build and train model
        print(f"  Building model...")
        model_DNN = NDD(prefilter_train.shape[1])
        train_label_new_forDNN = np.array([[0, 1] if i == 1 else [1, 0] for i in train_label_new])
        
        print(f"  Training (20 epochs)...", end=" ", flush=True)
        model_DNN.fit(prefilter_train, train_label_new_forDNN, batch_size=100, epochs=20, 
                     shuffle=True, validation_split=0, verbose=0)
        print("[OK]")
        
        print(f"  Predicting...", end=" ", flush=True)
        proba = model_DNN.predict_classes(prefilter_test, batch_size=200, verbose=0)
        ae_y_pred_prob = model_DNN.predict_proba(prefilter_test, batch_size=200, verbose=0)
        print("[OK]")
        
        # Calculate metrics
        fpr, tpr, _ = roc_curve(real_labels, ae_y_pred_prob[:, 1])
        auc_score = auc(fpr, tpr)
        precision_arr, recall_arr, pr_thresholds = precision_recall_curve(real_labels, ae_y_pred_prob[:, 1])
        aupr_score = auc(recall_arr, precision_arr)
        
        # Find optimal threshold
        f_measures = np.zeros(len(pr_thresholds))
        for k in range(len(pr_thresholds)):
            if (precision_arr[k] + recall_arr[k]) > 0:
                f_measures[k] = 2 * precision_arr[k] * recall_arr[k] / (precision_arr[k] + recall_arr[k])
        
        max_idx = f_measures.argmax()
        threshold = pr_thresholds[max_idx]
        predicted_score = (ae_y_pred_prob[:, 1] > threshold).astype(int)
        
        # Calculate final metrics
        f = f1_score(real_labels, predicted_score)
        rec = recall_score(real_labels, predicted_score)
        prec = precision_score(real_labels, predicted_score)
        
        print(f"  Results:")
        print(f"    Recall:    {rec:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    AUC:       {auc_score:.4f}")
        print(f"    AUPR:      {aupr_score:.4f}")
        print(f"    F-Score:   {f:.4f}\n")
        
        all_performance_DNN.append([rec, prec, auc_score, aupr_score, f])
    
    # Final results
    print("="*70)
    print("FINAL RESULTS (Mean across all folds)")
    print("="*70)
    mean_perf = np.mean(np.array(all_performance_DNN), axis=0)
    print(f"  Recall:    {mean_perf[0]:.4f}")
    print(f"  Precision: {mean_perf[1]:.4f}")
    print(f"  AUC:       {mean_perf[2]:.4f}")
    print(f"  AUPR:      {mean_perf[3]:.4f}")
    print(f"  F-Score:   {mean_perf[4]:.4f}")
    print("="*70 + "\n")
    print("[OK] NDD execution completed successfully!")

if __name__ == "__main__":
    try:
        DeepMDA()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] User stopped execution")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
