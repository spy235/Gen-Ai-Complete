# Deep Learning: Backpropagation Algorithm

## Overview
**Backpropagation** is the algorithm used to train neural networks by calculating gradients of the loss function.
- Core algorithm for learning in deep networks
- Calculates how much each weight contributes to the error
- Enables efficient gradient computation via chain rule
- Foundation of all modern deep learning

---

## Key Concepts

### 1. Loss Function
Measures the difference between predicted and actual output.

**Example:**
```
Prediction: 0.9
Actual: 1.0
Loss: (0.9 - 1.0)² = 0.01
```

**Common Loss Functions:**
```python
# Classification
tf.keras.losses.CategoricalCrossentropy()
tf.keras.losses.BinaryCrossentropy()

# Regression
tf.keras.losses.MeanSquaredError()
tf.keras.losses.MeanAbsoluteError()
```

### 2. Gradient
Direction and magnitude of steepest descent.

**Mathematical Definition:**
```
∇L = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ]
```

**Interpretation:**
- Large positive gradient: Weight causes large error increase
- Large negative gradient: Weight causes large error decrease
- Small gradient: Weight has little impact

### 3. Chain Rule
Fundamental calculus rule used in backpropagation.

**Formula:**
```
dz/dx = (dz/dy) × (dy/dx)
```

**Example:**
```
z = (2x + 1)²
dz/dx = 2(2x + 1) × 2 = 4(2x + 1)
```

---

## How Backpropagation Works

### Step 1: Forward Pass
Compute output from input through all layers.

```
Input: x
Hidden1 = σ(W₁ × x + b₁)
Hidden2 = σ(W₂ × Hidden1 + b₂)
Output = W₃ × Hidden2 + b₃
Loss = L(Output, Target)
```

### Step 2: Calculate Output Layer Gradient
Start with gradient of loss with respect to output.

```
∂L/∂output = -2 × (target - output)  [for MSE loss]
```

### Step 3: Propagate Gradient Backwards
Use chain rule to calculate gradient for each layer going backwards.

```
Layer 3 gradient:
∂L/∂W₃ = ∂L/∂output × ∂output/∂W₃

Layer 2 gradient:
∂L/∂W₂ = ∂L/∂output × ∂output/∂Hidden2 × ∂Hidden2/∂W₂

Layer 1 gradient:
∂L/∂W₁ = ∂L/∂output × ∂output/∂Hidden2 × ∂Hidden2/∂Hidden1 × ∂Hidden1/∂W₁
```

### Step 4: Update Weights
Use gradients to update weights in direction that reduces loss.

```
W_new = W_old - learning_rate × ∂L/∂W
```

---

## Mathematical Example

### Simple Network
```
Input (x) → Neuron1 (W₁, b₁) → ReLU → Neuron2 (W₂, b₂) → Output
```

### Forward Pass
```
z₁ = W₁ × x + b₁
a₁ = ReLU(z₁) = max(0, z₁)
z₂ = W₂ × a₁ + b₂
L = (z₂ - y)²  [MSE Loss]
```

### Backward Pass
```
∂L/∂z₂ = 2(z₂ - y)

∂L/∂W₂ = ∂L/∂z₂ × ∂z₂/∂W₂ = 2(z₂ - y) × a₁

∂L/∂a₁ = ∂L/∂z₂ × ∂z₂/∂a₁ = 2(z₂ - y) × W₂

∂L/∂z₁ = ∂L/∂a₁ × ∂a₁/∂z₁ = ∂L/∂a₁ × (1 if z₁ > 0 else 0)

∂L/∂W₁ = ∂L/∂z₁ × ∂z₁/∂W₁ = ∂L/∂z₁ × x
```

### Weight Update
```
W₁_new = W₁ - α × ∂L/∂W₁
W₂_new = W₂ - α × ∂L/∂W₂

where α = learning_rate
```

---

## Backpropagation in Practice

### Automatic Differentiation with TensorFlow

```python
import tensorflow as tf

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile (automatically handles backpropagation)
model.compile(optimizer='adam', loss='mse')

# Training (backpropagation happens automatically)
model.fit(X_train, y_train, epochs=10)
```

### Manual Backpropagation

```python
import tensorflow as tf

# Data
X = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y = tf.constant([[1.0], [0.0]])

# Model
W1 = tf.Variable(tf.random.normal((2, 4)))
b1 = tf.Variable(tf.zeros((4,)))
W2 = tf.Variable(tf.random.normal((4, 1)))
b2 = tf.Variable(tf.zeros((1,)))

# Training step
learning_rate = 0.01

for epoch in range(100):
    with tf.GradientTape() as tape:
        # Forward pass
        z1 = tf.matmul(X, W1) + b1
        a1 = tf.nn.relu(z1)
        logits = tf.matmul(a1, W2) + b2
        
        # Loss
        loss = tf.reduce_mean(tf.square(logits - y))
    
    # Backward pass (compute gradients)
    gradients = tape.gradient(loss, [W1, b1, W2, b2])
    
    # Update weights
    for var, grad in zip([W1, b1, W2, b2], gradients):
        var.assign_sub(learning_rate * grad)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")
```

---

## Vanishing and Exploding Gradients

### Vanishing Gradient Problem
Gradients become extremely small as they propagate backwards.

```
∂L/∂W₁ = ∂L/∂output × ∂output/∂h₁ × ∂h₁/∂h₂ × ... × ∂L/∂W₁
       = 0.5 × 0.4 × 0.3 × 0.2 × 0.1 = 0.0012

Result: Very small gradient update, slow learning
```

**Causes:**
- Chain of multiplications with values < 1
- Sigmoid/Tanh activation functions (gradients max ~0.25)

**Solutions:**
```python
# 1. Use ReLU instead of Sigmoid/Tanh
layers.Dense(128, activation='relu')

# 2. Batch Normalization
layers.BatchNormalization()

# 3. Skip connections
# Output = input + f(input)

# 4. Layer Normalization
layers.LayerNormalization()
```

### Exploding Gradient Problem
Gradients become extremely large.

```
∂L/∂W₁ = 2.5 × 2.0 × 3.0 × 2.5 = 37.5

Result: Large weight updates, unstable training
```

**Causes:**
- Chain of multiplications with values > 1
- Poorly initialized weights

**Solutions:**
```python
# 1. Gradient Clipping
optimizer = tf.keras.optimizers.Adam(clipvalue=1.0)

# 2. Better weight initialization
layers.Dense(128, kernel_initializer='he_normal')

# 3. Batch Normalization
layers.BatchNormalization()

# 4. Reduce learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
```

---

## Learning Dynamics

### Learning Rate Impact

**Too High Learning Rate:**
```
Loss may diverge or oscillate
Weights jump over optimal values
```

**Too Low Learning Rate:**
```
Very slow convergence
May get stuck in local minima
```

**Just Right:**
```
Smooth convergence to minimum
Efficient training
```

### Batch Size Effect

**Small Batch Size:**
```
- Noisier gradients
- Better generalization
- Slower per update
```

**Large Batch Size:**
```
- More stable gradients
- May overfit
- Faster per update
```

---

## Implementation Best Practices

### Model Compilation
```python
model.compile(
    optimizer='adam',           # Handles gradient updates
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Training with Monitoring
```python
history = model.fit(
    X_train, y_train,
    batch_size=32,             # Smaller batch → noisier gradients
    epochs=50,
    validation_data=(X_val, y_val),
    verbose=1
)

# Plot loss curve
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train', 'Validation'])
plt.show()
```

### Gradient Monitoring
```python
# Get gradients manually
with tf.GradientTape() as tape:
    logits = model(X)
    loss = loss_fn(y, logits)

gradients = tape.gradient(loss, model.trainable_weights)

# Check for NaN or Inf
for grad in gradients:
    if tf.reduce_any(tf.math.is_nan(grad)):
        print("Warning: NaN gradient detected!")
    if tf.reduce_any(tf.math.is_inf(grad)):
        print("Warning: Inf gradient detected!")
```

---

## Key Takeaways
1. **Backpropagation** computes gradients via chain rule
2. **Forward pass** computes loss
3. **Backward pass** computes gradients
4. **Weight update** uses gradients to minimize loss
5. **Vanishing/exploding gradients** are common issues
6. **Modern techniques** (batch norm, ReLU, skip connections) help
7. **Learning rate** is critical for convergence

---

## References
- Rumelhart et al. "Learning Representations by Back-Propagating Errors" (1986)
- Goodfellow et al. "Deep Learning" (MIT Press)
- TensorFlow Backpropagation Documentation

---
