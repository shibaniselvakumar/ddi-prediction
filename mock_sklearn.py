#!/usr/bin/env python
"""
Mock implementations for missing dependencies to allow running NDD
"""
import sys
import numpy as np

# Create mock sklearn module
class MockLabelEncoder:
    def fit(self, y):
        self.classes_ = np.unique(y)
        return self
    
    def transform(self, y):
        return np.array([list(self.classes_).index(x) for x in y])

class MockMetrics:
    @staticmethod
    def f1_score(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        if tp + fp == 0 or tp + fn == 0:
            return 0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    @staticmethod
    def recall_score(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return tp / (tp + fn) if tp + fn > 0 else 0
    
    @staticmethod
    def precision_score(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        return tp / (tp + fp) if tp + fp > 0 else 0
    
    @staticmethod
    def roc_curve(y_true, y_pred_prob):
        thresholds = np.sort(np.unique(y_pred_prob))[::-1]
        fpr = []
        tpr = []
        for thresh in thresholds:
            y_pred = (y_pred_prob >= thresh).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            fpr.append(fp / (fp + tn) if fp + tn > 0 else 0)
            tpr.append(tp / (tp + fn) if tp + fn > 0 else 0)
        return np.array(fpr), np.array(tpr), thresholds
    
    @staticmethod
    def auc(x, y):
        return np.trapz(y, x)
    
    @staticmethod
    def precision_recall_curve(y_true, y_pred_prob):
        thresholds = np.sort(np.unique(y_pred_prob))[::-1]
        precision = []
        recall = []
        for thresh in thresholds:
            y_pred = (y_pred_prob >= thresh).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            p = tp / (tp + fp) if tp + fp > 0 else 0
            r = tp / (tp + fn) if tp + fn > 0 else 0
            precision.append(p)
            recall.append(r)
        return np.array(precision), np.array(recall), thresholds
    
    @staticmethod
    def accuracy_score(y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """Compute confusion matrix"""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        for i, true_label in enumerate(classes):
            for j, pred_label in enumerate(classes):
                cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
        
        return cm
    
    @staticmethod
    def classification_report(y_true, y_pred, digits=2):
        """Generate classification report"""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        report = "Classification Report\n"
        report += "-" * 50 + "\n"
        
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))
            
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            support = np.sum(y_true == cls)
            
            report += f"Class {cls}: P={precision:.{digits}f}, R={recall:.{digits}f}, F1={f1:.{digits}f}, Support={support}\n"
        
        return report

# Inject mock modules
sys.modules['sklearn'] = type(sys)('sklearn')
sys.modules['sklearn.preprocessing'] = type(sys)('sklearn.preprocessing')
sys.modules['sklearn.metrics'] = type(sys)('sklearn.metrics')

sys.modules['sklearn.preprocessing'].LabelEncoder = MockLabelEncoder
metrics_mock = MockMetrics()
sys.modules['sklearn.metrics'].f1_score = metrics_mock.f1_score
sys.modules['sklearn.metrics'].recall_score = metrics_mock.recall_score
sys.modules['sklearn.metrics'].precision_score = metrics_mock.precision_score
sys.modules['sklearn.metrics'].accuracy_score = metrics_mock.accuracy_score
sys.modules['sklearn.metrics'].confusion_matrix = metrics_mock.confusion_matrix
sys.modules['sklearn.metrics'].classification_report = metrics_mock.classification_report
sys.modules['sklearn.metrics'].roc_curve = metrics_mock.roc_curve
sys.modules['sklearn.metrics'].auc = metrics_mock.auc
sys.modules['sklearn.metrics'].precision_recall_curve = metrics_mock.precision_recall_curve

print("Mock sklearn modules injected successfully")
