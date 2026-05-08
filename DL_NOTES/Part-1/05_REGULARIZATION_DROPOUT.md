# Deep Learning: Regularization and Dropout

## Overview
**Regularization** techniques prevent neural networks from overfitting.
- Overfitting: Model memorizes training data instead of learning generalizable patterns
- Regularization: Forces model to learn simpler, more robust patterns
- Critical for good generalization on unseen data
- Multiple techniques can be combined

---

## The Overfitting Problem

### What is Overfitting?
```
Training Loss: 0.01  (Very low)
Validation Loss: 0.5  (Very high)

Problem: Model fits noise in training data
Solution: Regularization
```

### Common Symptoms
- Training accuracy high, validation accuracy low
- Loss curve plateaus but keeps decreasing on training set
- Large weight values

---

## L1 and L2 Regularization

### L2 Regularization (Ridge)
Add penalty for large weights to loss function.

**Formula:**
```
Total_Loss = Data_Loss + λ × Σ(w²)

where λ is regularization strength (typically 0.0001-0.001)
```

**Intuition:**
- Penalizes large weights
- Encourages smaller weight values
- Spreads influence across all features

**Code Example:**
```python
from tensorflow import keras
from tensorflow.keras import regularizers

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', 
                      kernel_regularizer=regularizers.l2(0.001),
                      input_shape=(784,)),
    
    keras.layers.Dense(64, activation='relu',
                      kernel_regularizer=regularizers.l2(0.001)),
    
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X_train, y_train, epochs=20, validation_split=0.2)
```

**Effect on Loss:**
```
Without L2:   Loss = |y_pred - y_true|²
With L2:      Loss = |y_pred - y_true|² + 0.001 × Σ(w²)
```

### L1 Regularization (Lasso)
Add penalty for absolute weight values.

**Formula:**
```
Total_Loss = Data_Loss + λ × Σ(|w|)
```

**Difference from L2:**
- L1 pushes weights to exactly zero
- L2 shrinks weights but rarely to zero
- L1 performs feature selection

**Code Example:**
```python
from tensorflow.keras import regularizers

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu',
                      kernel_regularizer=regularizers.l1(0.001),
                      input_shape=(784,)),
    
    keras.layers.Dense(10, activation='softmax')
])
```

### L1 + L2 (Elastic Net)
Combination of both L1 and L2 regularization.

**Code Example:**
```python
from tensorflow.keras import regularizers

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu',
                      kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001),
                      input_shape=(784,)),
    
    keras.layers.Dense(10, activation='softmax')
])
```

---

## Dropout

### Overview
Randomly deactivate neurons during training to prevent co-adaptation.

**Concept:**
```
During Training: Randomly set 50% of neurons to 0
During Testing: Use all neurons (scaled appropriately)

Result: Ensemble of thinned networks
```

### How It Works

**With 50% Dropout:**
```
Original network: x → [neuron1, neuron2, neuron3] → output
During training:  x → [neuron1, X, neuron3] → output (neuron2 dropped)
Next iteration:   x → [X, neuron2, X] → output (different neurons dropped)
During testing:   x → [neuron1×0.5, neuron2×0.5, neuron3×0.5] → output
```

### Code Example

```python
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.5),              # Drop 50% of neurons
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),              # Drop 50% of neurons
    
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X_train, y_train, epochs=20, validation_split=0.2)
```

### Dropout Rates
```
Typical values: 0.2 - 0.5
- Too low (0.1): Little effect
- Too high (0.8): Information loss
- 0.5: Often a good starting point
```

### Interpretation as Ensemble
Dropout is equivalent to training multiple thinned networks and averaging them:
```
Ensemble of ~2^n different thinned networks
where n = number of hidden neurons

Single forward pass = approximate ensemble prediction
```

---

## Batch Normalization

### Overview
Normalize layer inputs to zero mean and unit variance.

**Formula:**
```
Normalized = (x - batch_mean) / √(batch_variance + ε)
Scaled = γ × Normalized + β

where γ and β are learnable parameters
```

### Benefits
1. **Regularization effect**: Reduces internal covariate shift
2. **Allows higher learning rates**: More stable training
3. **Reduces sensitivity to initialization**
4. **Acts as regularizer**: Similar to dropout
5. **Helps with vanishing gradient**: Maintains gradient flow

### Code Example

```python
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, input_shape=(784,)),
    layers.BatchNormalization(),      # Normalize inputs
    layers.Activation('relu'),
    
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X_train, y_train, epochs=20)
```

### Training vs Testing
```
Training: Use batch statistics (mean/var)
Testing: Use running average of batch statistics
```

---

## Early Stopping

### Overview
Stop training when validation loss stops improving.

**Concept:**
```
Epoch 1: Train Loss = 0.5, Val Loss = 0.48
Epoch 2: Train Loss = 0.3, Val Loss = 0.45 ✓ (improving)
Epoch 3: Train Loss = 0.2, Val Loss = 0.50 ✗ (validation worsened)
Epoch 4: Train Loss = 0.1, Val Loss = 0.52 ✗

Stop at Epoch 2 (best validation loss)
```

### Code Example

```python
from tensorflow.keras.callbacks import EarlyStopping

callback = EarlyStopping(
    monitor='val_loss',        # Metric to monitor
    patience=5,                # Wait 5 epochs for improvement
    restore_best_weights=True  # Restore weights from best epoch
)

model.fit(
    X_train, y_train,
    epochs=100,                # Could train for many epochs
    validation_split=0.2,
    callbacks=[callback]       # Stop early if no improvement
)
```

### Parameters
```python
EarlyStopping(
    monitor='val_loss',        # 'val_loss', 'val_accuracy', etc
    patience=5,                # Stop if no improvement for N epochs
    min_delta=0.001,           # Minimum change to count as improvement
    restore_best_weights=True  # Restore best weights
)
```

---

## Data Augmentation

### Overview
Create new training samples by transforming existing ones.

**Image Augmentation Example:**
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,         # Random rotations
    width_shift_range=0.2,     # Random horizontal shifts
    height_shift_range=0.2,    # Random vertical shifts
    horizontal_flip=True,      # Random horizontal flips
    zoom_range=0.2             # Random zoom
)

# Use augmented data for training
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          epochs=20)
```

**Text Augmentation Example:**
```
Original: "I love this movie"
Augmentation:
- Paraphrase: "I really enjoyed this film"
- Add noise: "I lvoe this movie"
- Synonym replacement: "I adore this film"
```

**Benefits:**
- Increases effective training data size
- Improves generalization
- Reduces overfitting

---

## Weight Decay

### Overview
Penalize large weights (similar to L2 but in optimizer).

**Code Example:**
```python
from tensorflow.keras.optimizers import AdamW

optimizer = AdamW(learning_rate=0.001, weight_decay=0.0001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy')
```

---

## Combining Regularization Techniques

### Example: Multi-layer Defense
```python
model = keras.Sequential([
    layers.Dense(512, input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    layers.Dense(10, activation='softmax')
])

# Optimizer with weight decay
optimizer = keras.optimizers.AdamW(weight_decay=0.0001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy')

# Early stopping callback
callback = EarlyStopping(patience=10, restore_best_weights=True)

model.fit(X_train, y_train, epochs=100, 
         validation_split=0.2, callbacks=[callback])
```

---

## Choosing Regularization Strategies

| Problem | Strategy |
|---------|----------|
| Model overfitting | Use Dropout + Early Stopping |
| Very large weights | Add L2 regularization |
| Want sparse model | Use L1 regularization |
| Training unstable | Use Batch Normalization |
| Need more data | Data Augmentation |
| Need quick training | Batch Normalization |

---

## Best Practices
1. **Start with L2 regularization**: λ = 0.0001 to 0.001
2. **Add Dropout (0.2-0.5)**: After dense layers
3. **Use Batch Normalization**: Before activation functions
4. **Use Early Stopping**: Monitor validation loss
5. **Data Augmentation**: Always do for images/time-series
6. **Monitor train-val gap**: Large gap indicates overfitting
7. **Ensemble approach**: Combine multiple regularization techniques

---

## References
- Srivastava et al. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014)
- Ioffe & Szegedy "Batch Normalization: Accelerating Deep Network Training" (2015)
- Goodfellow et al. "Deep Learning" (MIT Press)

---
