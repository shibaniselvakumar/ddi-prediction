#!/usr/bin/env python
"""
NDD - Drug-Drug Interaction Prediction by Neural Network Using Integrated Similarity
This version uses mock implementations of missing dependencies
"""

import sys
import os

# Add mocks BEFORE any other imports
import mock_sklearn

# Now import numpy and other dependencies
import numpy as np

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available")

# Import mocked sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score, precision_score, roc_curve, auc, precision_recall_curve

# Try to import keras
try:
    from keras.layers.core import Dropout, Activation
    from keras.layers import Dense
    from keras import optimizers
    from keras.models import Sequential
    from keras.utils import np_utils
    HAS_KERAS = True
except Exception as e:
    print(f"Warning: Keras import error: {e}")
    HAS_KERAS = False

# Change to data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "NDD", "DS1")

if os.path.exists(data_dir):
    os.chdir(data_dir)
    print(f"✓ Working directory: {os.getcwd()}")
else:
    print(f"⚠ Data directory not found at {data_dir}")
    print(f"  Trying current directory: {os.getcwd()}")

# ============ NDD Methods ============

def prepare_data(seperate=False):
    """Load drug features and interaction matrix"""
    try:
        drug_fea = np.loadtxt("IntegratedDS1.txt", dtype=float, delimiter=",")
        interaction = np.loadtxt("drug_drug_matrix.csv", dtype=int, delimiter=",")
    except FileNotFoundError as e:
        print(f"✗ ERROR: Data file not found: {e}")
        print(f"\n  Expected files in {os.getcwd()}:")
        print("    - IntegratedDS1.txt")
        print("    - drug_drug_matrix.csv")
        sys.exit(1)
    
    train = []
    label = []
    tmp_fea = []
    drug_fea_tmp = []
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
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    
    acc = float(tp + tn) / test_num
    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        sensitivity = float(tp) / (tp + fn) if (tp + fn) > 0 else 0
        specificity = float(tn) / (tn + fp) if (tn + fp) > 0 else 0
    else:
        precision = float(tp) / (tp + fp)
        sensitivity = float(tp) / (tp + fn) if (tp + fn) > 0 else 0
        specificity = float(tn) / (tn + fp) if (tn + fp) > 0 else 0
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
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
    print("\n" + "="*70)
    print("NDD - Drug-Drug Interaction Prediction")
    print("="*70 + "\n")
    
    if not HAS_KERAS:
        print("✗ FATAL: Keras is required but not available")
        print("  Please install: pip install keras tensorflow")
        sys.exit(1)
    
    print("✓ Loading data...")
    X, labels = prepare_data(seperate=True)
    X_data1, X_data2 = transfer_array_format(X)
    X = 0
    
    print("✓ Preprocessing labels...")
    y, encoder = preprocess_labels(labels)
    X = np.concatenate((X_data1, X_data2), axis=1)
    
    num = np.arange(len(y))
    np.random.shuffle(num)
    X_data1 = X_data1[num]
    X_data2 = X_data2[num]
    y = y[num]
    
    num_cross_val = 5
    all_performance_DNN = []
    
    print(f"✓ Starting {num_cross_val}-fold cross-validation...\n")
    
    for fold in range(num_cross_val):
        print(f"  Fold {fold + 1}/{num_cross_val}:")
        
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
        train1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val != fold])
        test1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val == fold])
        train2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val != fold])
        test2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val == fold])
        
        zerotest = 0
        nozerotest = 0
        zerotrain = 0
        nozerotrain = 0
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                nozerotest = nozerotest + 1
                real_labels.append(1)
            else:
                zerotest = zerotest + 1
                real_labels.append(0)
        
        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                zerotrain = zerotrain + 1
                train_label_new.append(1)
            else:
                nozerotrain = nozerotrain + 1
                train_label_new.append(0)
        
        prefilter_train = np.concatenate((train1, train2), axis=1)
        prefilter_test = np.concatenate((test1, test2), axis=1)
        
        print(f"    Train: {len(prefilter_train)} samples, Test: {len(prefilter_test)} samples")
        
        model_DNN = NDD(prefilter_train.shape[1])
        train_label_new_forDNN = np.array([[0, 1] if i == 1 else [1, 0] for i in train_label_new])
        
        print(f"    Training model (20 epochs)...", end=" ", flush=True)
        model_DNN.fit(prefilter_train, train_label_new_forDNN, batch_size=100, epochs=20, 
                     shuffle=True, validation_split=0, verbose=0)
        print("done")
        
        print(f"    Generating predictions...", end=" ", flush=True)
        proba = model_DNN.predict_classes(prefilter_test, batch_size=200, verbose=0)
        ae_y_pred_prob = model_DNN.predict_proba(prefilter_test, batch_size=200, verbose=0)
        print("done")
        
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba, real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob[:, 1])
        auc_score = auc(fpr, tpr)
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob[:, 1])
        aupr_score = auc(recall, precision1)
        
        all_F_measure = np.zeros(len(pr_threshods))
        for k in range(0, len(pr_threshods)):
            if (precision1[k] + recall[k]) > 0:
                all_F_measure[k] = 2 * precision1[k] * recall[k] / (precision1[k] + recall[k])
            else:
                all_F_measure[k] = 0
        
        max_index = all_F_measure.argmax()
        predicted_score = np.zeros(len(real_labels))
        threshold = pr_threshods[max_index]
        p = ae_y_pred_prob[:, 1]
        predicted_score[p > threshold] = 1
        
        f = f1_score(real_labels, predicted_score)
        recall_score_val = recall_score(real_labels, predicted_score)
        precision_score_val = precision_score(real_labels, predicted_score)
        
        print(f"    Results:")
        print(f"      Recall: {recall_score_val:.4f} | Precision: {precision_score_val:.4f}")
        print(f"      AUC: {auc_score:.4f} | AUPR: {aupr_score:.4f} | F-Score: {f:.4f}\n")
        
        all_performance_DNN.append([recall_score_val, precision_score_val, auc_score, aupr_score, f])
    
    print("\n" + "="*70)
    print("FINAL RESULTS (Mean across all folds):")
    print("="*70)
    print("Metric      | Value")
    print("-"*70)
    mean_performance = np.mean(np.array(all_performance_DNN), axis=0)
    print(f"Recall      | {mean_performance[0]:.4f}")
    print(f"Precision   | {mean_performance[1]:.4f}")
    print(f"AUC         | {mean_performance[2]:.4f}")
    print(f"AUPR        | {mean_performance[3]:.4f}")
    print(f"F-Score     | {mean_performance[4]:.4f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        DeepMDA()
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
