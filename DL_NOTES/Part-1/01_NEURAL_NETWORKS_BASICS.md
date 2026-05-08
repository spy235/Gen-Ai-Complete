# Deep Learning: Neural Networks Basics

## Overview
**Neural Networks** are computational models inspired by biological neural networks in animal brains.
- Composed of interconnected nodes (neurons) organized in layers
- Learn patterns from data through training
- Foundation of modern deep learning
- Can approximate any continuous function (Universal Approximation Theorem)

---

## Key Concepts

### 1. Neuron (Perceptron)
Basic building block of neural networks.

**Structure:**
```
Inputs (x₁, x₂, ..., xₙ)
         ↓ (weights w₁, w₂, ..., wₙ)
    Linear Sum: z = Σ(wᵢ × xᵢ) + b
         ↓
  Activation Function: f(z)
         ↓
    Output: y
```

**Mathematical Formula:**
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
y = f(z)
```

Where:
- `w` = weights (learnable parameters)
- `b` = bias (learnable parameter)
- `f` = activation function
- `x` = input features

### 2. Network Architecture
Layers of neurons stacked together.

**Typical Structure:**
```
Input Layer → Hidden Layers → Output Layer
(n features)  (m neurons each)  (k classes/outputs)
```

### 3. Parameters
Weights and biases that the network learns.

```
Total Parameters = (input_features × hidden_neurons + hidden_neurons) 
                   + (hidden_neurons × output_neurons + output_neurons)
```

### 4. Forward Propagation
Computing output from input through all layers.

```python
# Single neuron forward pass
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
y = activation(z)

# Multi-layer network
layer1 = activation(X @ W1 + b1)  # Hidden layer
layer2 = activation(layer1 @ W2 + b2)  # Hidden layer
output = activation(layer2 @ W3 + b3)  # Output layer
```

---

## Neural Network Implementation

### Using TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),      # Hidden layer 1
    layers.Dense(64, activation='relu'),                            # Hidden layer 2
    layers.Dense(32, activation='relu'),                            # Hidden layer 3
    layers.Dense(10, activation='softmax')                          # Output layer
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Predict
predictions = model.predict(X_test)
```

### Layer Parameters
```python
Dense(
    units=128,              # Number of neurons in layer
    activation='relu',      # Activation function
    input_shape=(784,),     # Input shape for first layer
    kernel_initializer='glorot_uniform',  # Weight initialization
    bias_initializer='zeros'              # Bias initialization
)
```

---

## Types of Layers

### 1. Dense (Fully Connected)
Every neuron connects to every neuron in next layer.

```python
layers.Dense(128, activation='relu')
# Each neuron receives input from all previous neurons
```

### 2. Activation Layers
Apply non-linearity to layer outputs.

```python
layers.Activation('relu')
layers.Activation('sigmoid')
layers.Activation('softmax')
```

### 3. Dropout Layer
Randomly deactivates neurons during training to prevent overfitting.

```python
layers.Dropout(0.5)  # Drop 50% of neurons
```

### 4. Batch Normalization
Normalizes layer inputs to improve training.

```python
layers.BatchNormalization()
```

---

## How Neural Networks Learn

### 1. Loss Function
Measures difference between predicted and actual output.

**Common Loss Functions:**
```
Classification: Categorical Cross-Entropy
                Binary Cross-Entropy
                
Regression:     Mean Squared Error (MSE)
                Mean Absolute Error (MAE)
```

### 2. Optimization
Process of updating weights to minimize loss.

**Algorithm:** Gradient Descent
```
1. Forward pass: Calculate output
2. Calculate loss: Measure prediction error
3. Backward pass (Backpropagation): Calculate gradients
4. Update weights: w = w - learning_rate × gradient
5. Repeat until convergence
```

### 3. Training Loop
```python
for epoch in range(num_epochs):
    for batch in training_data:
        # Forward pass
        predictions = model(batch_X)
        
        # Calculate loss
        loss = loss_function(predictions, batch_y)
        
        # Backward pass (automatic with TensorFlow)
        gradients = compute_gradients(loss, model.weights)
        
        # Update weights
        update_weights(model.weights, gradients)
```

---

## Network Architectures

### 1. Shallow Network
One hidden layer.

```
Input (10) → Hidden (5) → Output (2)
```

### 2. Deep Network
Multiple hidden layers.

```
Input (10) → Hidden1 (128) → Hidden2 (64) → Hidden3 (32) → Output (2)
```

### 3. Wide Network
Many neurons in each layer.

```
Input (10) → Hidden1 (1000) → Hidden2 (500) → Output (2)
```

### Code Example
```python
# Shallow Network
shallow_model = keras.Sequential([
    layers.Dense(5, activation='relu', input_shape=(10,)),
    layers.Dense(2, activation='softmax')
])

# Deep Network
deep_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(10,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Wide Network
wide_model = keras.Sequential([
    layers.Dense(1000, activation='relu', input_shape=(10,)),
    layers.Dense(500, activation='relu'),
    layers.Dense(2, activation='softmax')
])
```

---

## Training Process

### Data Preparation
```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
```

### Model Training
```python
# Compile
model.compile(
    optimizer='adam',                    # Optimizer algorithm
    loss='categorical_crossentropy',     # Loss function
    metrics=['accuracy']                 # Metrics to monitor
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=20,                          # Number of passes through data
    batch_size=32,                      # Samples per batch
    validation_split=0.2,               # 20% for validation
    verbose=1                           # Print progress
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

### Predictions
```python
# Single prediction
single_prediction = model.predict(X_test[0:1])
print(f"Predicted class: {np.argmax(single_prediction)}")

# Batch predictions
batch_predictions = model.predict(X_test)
predicted_classes = np.argmax(batch_predictions, axis=1)
```

---

## Advantages of Neural Networks
- Can learn complex non-linear relationships
- Automatic feature extraction
- Scales well with more data
- Works with high-dimensional data
- State-of-the-art for many tasks

## Disadvantages of Neural Networks
- Requires large amounts of data
- Computationally expensive
- Difficult to interpret (black box)
- Prone to overfitting if not careful
- Hyperparameter tuning is complex

---

## Common Issues and Solutions

### Overfitting
Network memorizes training data instead of learning patterns.

**Solutions:**
```python
# 1. Dropout
layers.Dropout(0.5)

# 2. Early Stopping
callbacks = [keras.callbacks.EarlyStopping(patience=5)]
model.fit(..., callbacks=callbacks)

# 3. Regularization (L1/L2)
layers.Dense(128, activation='relu', 
             kernel_regularizer=keras.regularizers.l2(0.001))

# 4. More training data
# 5. Reduce model complexity
```

### Underfitting
Network is too simple to learn patterns.

**Solutions:**
```python
# 1. Increase model complexity
layers.Dense(512, activation='relu')

# 2. Train longer
epochs = 100

# 3. Better features

# 4. Reduce regularization
```

### Vanishing/Exploding Gradients
Gradients become too small or too large during backpropagation.

**Solutions:**
```python
# 1. Better initialization
kernel_initializer='he_normal'

# 2. Batch Normalization
layers.BatchNormalization()

# 3. Gradient clipping
optimizer = keras.optimizers.Adam(clipvalue=1.0)

# 4. Use skip connections (ResNet)
```

---

## Use Cases
- Image classification
- Natural language processing
- Speech recognition
- Time series prediction
- Sentiment analysis
- Recommendation systems
- Game playing (AlphaGo)

---

## Best Practices
1. **Normalize input data**: Use StandardScaler or MinMaxScaler
2. **Start simple**: Add complexity gradually
3. **Use validation set**: Monitor generalization
4. **Early stopping**: Prevent overfitting
5. **Batch normalization**: Stabilize training
6. **Proper initialization**: Use He initialization for ReLU
7. **Monitor metrics**: Track accuracy, loss, etc.

---
