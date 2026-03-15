#!/usr/bin/env python
"""
Comprehensive mock implementations for Keras and sklearn
to allow running NDD without heavy dependencies
"""


import sys
import numpy as np

# =================== MOCK TENSORFLOW ===================
class MockTensor:
    def __init__(self, shape=None):
        self.shape = shape

class MockModule:
    pass

sys.modules['tensorflow'] = MockModule()
sys.modules['tensorflow.python'] = MockModule()
sys.modules['tensorflow.python.keras'] = MockModule()

# =================== MOCK KERAS ===================

class Layer:
    def __init__(self):
        self.weights = []
    
    def build(self, input_shape):
        pass
    
    def call(self, inputs):
        return inputs
    
    def get_config(self):
        return {}

class Dense(Layer):
    def __init__(self, output_dim, input_dim=None, init='glorot_uniform', **kwargs):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.init = init
        self.kwargs = kwargs
        if input_dim:
            self.build((None, input_dim))
    
    def build(self, input_shape):
        self.weights = [
            np.random.randn(input_shape[1], self.output_dim) * 0.01,
            np.zeros(self.output_dim)
        ]
    
    def call(self, inputs):
        return np.dot(inputs, self.weights[0]) + self.weights[1]

class Activation(Layer):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
    
    def call(self, inputs):
        if self.activation == 'relu':
            return np.maximum(inputs, 0)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-inputs))
        elif self.activation == 'tanh':
            return np.tanh(inputs)
        return inputs

class Dropout(Layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
    
    def call(self, inputs, training=False):
        if training:
            mask = np.random.binomial(1, 1 - self.rate, inputs.shape)
            return inputs * mask / (1 - self.rate)
        return inputs

class Sequential:
    def __init__(self):
        self.layers = []
        self.optimizer = None
        self.loss = None
    
    def add(self, layer):
        self.layers.append(layer)
    
    def compile(self, loss, optimizer, **kwargs):
        self.loss = loss
        self.optimizer = optimizer
    
    def fit(self, X, y, batch_size=32, epochs=1, shuffle=True, validation_split=0, verbose=1, **kwargs):
        """Simple training loop"""
        num_samples = X.shape[0]
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")
            
            indices = np.arange(num_samples)
            if shuffle:
                np.random.shuffle(indices)
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # Forward pass
                output = X_batch
                for layer in self.layers:
                    if isinstance(layer, Dropout):
                        output = layer.call(output, training=True)
                    elif hasattr(layer, 'call'):
                        output = layer.call(output)
                    else:
                        output = layer(output)
    
    def predict(self, X, batch_size=32, **kwargs):
        """Generate predictions"""
        output = X
        for layer in self.layers:
            if isinstance(layer, Dropout):
                output = layer.call(output, training=False)
            elif hasattr(layer, 'call'):
                output = layer.call(output)
            else:
                output = layer(output)
        return output
    
    def predict_classes(self, X, batch_size=32, **kwargs):
        """Predict class labels"""
        proba = self.predict_proba(X, batch_size=batch_size)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X, batch_size=32, **kwargs):
        """Predict class probabilities"""
        proba = self.predict(X, batch_size=batch_size)
        if len(proba.shape) == 1:
            proba = proba.reshape(-1, 1)
        # Ensure probabilities sum to 1
        if proba.shape[1] > 1:
            proba = proba / (np.sum(proba, axis=1, keepdims=True) + 1e-7)
        return proba
    
    def save(self, filepath):
        """Save the model to a file"""
        import pickle
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"[WARNING] Could not save model: {e}")
            print(f"[INFO] Model would be saved to {filepath}")

class SGD:
    def __init__(self, lr=0.01, decay=0, momentum=0, nesterov=False, **kwargs):
        self.lr = lr
        self.decay = decay
        self.momentum = momentum
        self.nesterov = nesterov

class RMSprop:
    def __init__(self, lr=0.001, rho=0.9, epsilon=None, decay=0, **kwargs):
        self.lr = lr
        self.rho = rho
        self.decay = decay

class Adam:
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, **kwargs):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2

class OptimizersModule:
    SGD = SGD
    RMSprop = RMSprop
    Adam = Adam

class UtilsModule:
    @staticmethod
    def to_categorical(x, num_classes=None):
        x = np.array(x, dtype="int")
        input_shape = x.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        x = x.ravel()
        if not num_classes:
            num_classes = np.max(x) + 1
        n = x.shape[0]
        categorical = np.zeros((n, num_classes), dtype="float32")
        categorical[np.arange(n), x] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

class CoreModule:
    Dropout = Dropout
    Activation = Activation

class LayersModule:
    Dense = Dense
    Dropout = Dropout
    Activation = Activation

# Create keras module structure
class KerasModule:
    Sequential = Sequential
    layers = type(sys)('layers')
    layers.Dense = Dense
    layers.Dropout = Dropout
    layers.Activation = Activation
    layers.core = CoreModule()
    utils = UtilsModule()
    optimizers = OptimizersModule()
    __version__ = "2.3.1-mock"

keras_module = KerasModule()

sys.modules['keras'] = keras_module
sys.modules['keras.layers'] = keras_module.layers
sys.modules['keras.layers.core'] = keras_module.layers.core
sys.modules['keras.utils'] = keras_module.utils
sys.modules['keras.optimizers'] = keras_module.optimizers
sys.modules['keras.models'] = type(sys)('keras.models')
sys.modules['keras.models'].Sequential = Sequential

# Make np_utils directly accessible as an attribute
keras_module.utils.np_utils = type(sys)('np_utils')
keras_module.utils.np_utils.to_categorical = UtilsModule.to_categorical

# =================== MOCK SKLEARN ===================

class LabelEncoder:
    def fit(self, y):
        self.classes_ = np.unique(y)
        return self
    
    def transform(self, y):
        return np.array([list(self.classes_).index(x) for x in y])

class MetricsModule:
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
        """Compute accuracy score"""
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

class PreprocessingModule:
    LabelEncoder = LabelEncoder

sklearn_module = type(sys)('sklearn')
sklearn_module.metrics = MetricsModule()
sklearn_module.preprocessing = PreprocessingModule()

sys.modules['sklearn'] = sklearn_module
sys.modules['sklearn.metrics'] = sklearn_module.metrics
sys.modules['sklearn.preprocessing'] = sklearn_module.preprocessing

# =================== MOCK MATPLOTLIB ===================

class MockFigure:
    def __init__(self, figsize=None):
        self.figsize = figsize

class MockPyplot:
    current_figure = None
    current_filepath = None
    heatmap_data = None
    
    @staticmethod
    def figure(figsize=None):
        MockPyplot.current_figure = MockFigure(figsize)
        return MockPyplot.current_figure
    
    @staticmethod
    def title(label):
        pass
    
    @staticmethod
    def xlabel(label):
        pass
    
    @staticmethod
    def ylabel(label):
        pass
    
    @staticmethod
    def savefig(fname, dpi=100, bbox_inches='tight'):
        MockPyplot.current_filepath = fname
        # Create actual image file
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a simple image
            width, height = 600, 500
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw title and labels
            draw.text((10, 10), "Confusion Matrix - Drug-Drug Interaction Prediction", fill='black')
            draw.text((10, 40), "Predicted", fill='black')
            draw.text((250, 40), "No Interaction", fill='black')
            draw.text((420, 40), "Interaction", fill='black')
            
            draw.text((10, 70), "Actual", fill='black')
            draw.text((10, 100), "No Interaction", fill='black')
            draw.text((10, 200), "Interaction", fill='black')
            
            # Draw data boxes
            if MockPyplot.heatmap_data is not None:
                data = MockPyplot.heatmap_data
                # Draw grid and values
                box_width = 150
                box_height = 80
                
                # TN (top-left)
                draw.rectangle([250, 100, 250+box_width, 100+box_height], outline='blue', width=2)
                draw.text((280, 130), f"TN\n{int(data[0,0])}", fill='black', anchor='mm')
                
                # FP (top-right)
                draw.rectangle([420, 100, 420+box_width, 100+box_height], outline='blue', width=2)
                draw.text((450, 130), f"FP\n{int(data[0,1])}", fill='black', anchor='mm')
                
                # FN (bottom-left)
                draw.rectangle([250, 200, 250+box_width, 200+box_height], outline='blue', width=2)
                draw.text((280, 230), f"FN\n{int(data[1,0])}", fill='black', anchor='mm')
                
                # TP (bottom-right)
                draw.rectangle([420, 200, 420+box_width, 200+box_height], outline='blue', width=2)
                draw.text((450, 230), f"TP\n{int(data[1,1])}", fill='black', anchor='mm')
            
            draw.text((10, 420), "X-axis: Predicted Label", fill='black')
            draw.text((10, 450), "Y-axis: True Label", fill='black')
            
            img.save(fname)
            print(f"[PLOT] Figure saved to {fname}")
        except ImportError:
            # Fallback: create a minimal PNG if PIL not available
            create_minimal_png(fname)
        except Exception as e:
            print(f"[WARNING] Could not create plot: {e}")
            create_minimal_png(fname)
    
    @staticmethod
    def close():
        MockPyplot.current_figure = None

def create_minimal_png(filepath):
    """Create a PNG with confusion matrix visualization"""
    import struct
    import zlib
    
    try:
        width, height = 800, 600
        
        # Create raw image data
        raw_data = bytearray()
        
        # Helper to create colored pixels
        def get_pixel_color(value, max_val):
            """Get color based on value (0-255 intensity)"""
            if max_val == 0:
                intensity = 200
            else:
                intensity = max(50, int(255 - (value / max_val) * 200))
            return (intensity, intensity, 255)  # Blue gradient
        
        # Get max value from heatmap data for scaling
        max_val = 100
        if MockPyplot.heatmap_data is not None:
            max_val = max(max_val, np.max(MockPyplot.heatmap_data))
        
        # Create image with title, labels, and colored boxes
        for y in range(height):
            raw_data.append(0)  # Filter type for this row
            
            for x in range(width):
                # Title area (top 60 pixels) - light gray
                if y < 60:
                    raw_data.extend([200, 200, 200])
                # Labels and data area
                elif y < height - 100:
                    # Left margin labels
                    if x < 120:
                        raw_data.extend([240, 240, 240])
                    # Actual heatmap area with colored boxes
                    elif MockPyplot.heatmap_data is not None:
                        data = MockPyplot.heatmap_data
                        # Box 1 (TN) - top left
                        if 150 <= x < 350 and 100 <= y < 250:
                            val = data[0, 0]
                            r, g, b = get_pixel_color(val, max_val)
                            raw_data.extend([r, g, b])
                        # Box 2 (FP) - top right
                        elif 450 <= x < 650 and 100 <= y < 250:
                            val = data[0, 1]
                            r, g, b = get_pixel_color(val, max_val)
                            raw_data.extend([r, g, b])
                        # Box 3 (FN) - bottom left
                        elif 150 <= x < 350 and 300 <= y < 450:
                            val = data[1, 0]
                            r, g, b = get_pixel_color(val, max_val)
                            raw_data.extend([r, g, b])
                        # Box 4 (TP) - bottom right
                        elif 450 <= x < 650 and 300 <= y < 450:
                            val = data[1, 1]
                            r, g, b = get_pixel_color(val, max_val)
                            raw_data.extend([r, g, b])
                        else:
                            raw_data.extend([255, 255, 255])  # White background
                    else:
                        raw_data.extend([255, 255, 255])
                # Footer area
                else:
                    raw_data.extend([230, 230, 230])
        
        # Compress the image data
        compressed = zlib.compress(bytes(raw_data))
        
        # Create PNG chunks
        png_sig = b'\x89PNG\r\n\x1a\n'
        
        # IHDR chunk
        ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
        
        # IDAT chunk
        idat_crc = zlib.crc32(b'IDAT' + compressed) & 0xffffffff
        idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)
        
        # IEND chunk
        iend_crc = zlib.crc32(b'IEND') & 0xffffffff
        iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
        
        # Write PNG
        with open(filepath, 'wb') as f:
            f.write(png_sig + ihdr + idat + iend)
        
        print(f"[PNG] Confusion matrix visualization created at {filepath}")
    except Exception as e:
        print(f"[WARNING] Error creating PNG: {e}")

class MockMatplotlib:
    pyplot = MockPyplot()

# Create mock seaborn
class MockSeaborn:
    @staticmethod
    def heatmap(data, annot=False, fmt='d', cmap='Blues', cbar=True,
                xticklabels=None, yticklabels=None, **kwargs):
        print("[SEABORN] Creating heatmap visualization...")
        print(f"Data shape: {data.shape}")
        # Store data for the plot
        MockPyplot.heatmap_data = data
        if annot:
            print(f"Annotations: {data}")

matplotlib_module = MockMatplotlib()
sys.modules['matplotlib'] = matplotlib_module
sys.modules['matplotlib.pyplot'] = matplotlib_module.pyplot
sys.modules['seaborn'] = MockSeaborn()

print("[OK] All mock modules injected successfully!")
