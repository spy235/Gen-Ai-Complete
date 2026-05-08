# Deep Learning: Activation Functions

## Overview
**Activation Functions** introduce non-linearity into neural networks.
- Without them, deep networks become just linear transformations
- Enable networks to learn complex patterns
- Applied to neuron outputs in each layer
- Critical for model expressiveness

---

## Why Activation Functions Matter

### Problem Without Activation
```
y = W₃ × (W₂ × (W₁ × x + b₁) + b₂) + b₃
  = W₃ × W₂ × W₁ × x + (bias terms)
  = W_combined × x + b_combined
  
Result: Just a linear function (no matter how deep!)
```

### With Activation
```
y = f(W₃ × f(W₂ × f(W₁ × x + b₁) + b₂) + b₃)

Result: Can approximate non-linear functions
```

---

## Common Activation Functions

### 1. ReLU (Rectified Linear Unit)
Most popular activation function in modern deep learning.

**Formula:**
```
f(x) = max(0, x)
```

**Characteristics:**
- Returns 0 for negative inputs
- Returns input for positive inputs
- Simple and fast to compute
- Computationally efficient

**Code Example:**
```python
import tensorflow as tf
from tensorflow.keras import layers

# In model
layers.Dense(128, activation='relu')

# Standalone
output = tf.nn.relu(input)

# Custom implementation
def relu(x):
    return tf.maximum(0.0, x)
```

**Visual:**
```
     |     /
     |    /
-----|---/----
    /|
   / |
```

**Advantages:**
- Faster convergence
- Reduces vanishing gradient problem
- Biologically plausible
- Sparse activation (many zeros)

**Disadvantages:**
- Dead ReLU problem (neurons always output 0)

---

### 2. Leaky ReLU
Variation of ReLU that addresses the dead ReLU problem.

**Formula:**
```
f(x) = x if x > 0
f(x) = α × x if x ≤ 0   (α is small constant, typically 0.01)
```

**Code Example:**
```python
# Using keras
layers.LeakyReLU(alpha=0.01)

# Standalone
output = tf.nn.leaky_relu(input, alpha=0.01)

# Custom implementation
def leaky_relu(x, alpha=0.01):
    return tf.maximum(alpha * x, x)
```

**Advantages:**
- Addresses dead ReLU problem
- Allows small negative gradients
- Better than ReLU in some cases

**Disadvantages:**
- Slightly more computationally expensive
- Hyperparameter α to tune

---

### 3. Sigmoid
Classic activation function, used mainly in output layers for binary classification.

**Formula:**
```
f(x) = 1 / (1 + e^(-x))
```

**Range:** 0 to 1

**Code Example:**
```python
# Using keras
layers.Dense(1, activation='sigmoid')  # Binary classification

# Standalone
output = tf.nn.sigmoid(input)
```

**Visual:**
```
1.0 |     -----
    |   /
0.5 | /
    |/
0.0 |
```

**Advantages:**
- Output between 0 and 1 (interpretable as probability)
- Smooth gradient
- Well-understood mathematically

**Disadvantages:**
- Vanishing gradient problem (gradients near 0)
- Not zero-centered output
- Computationally expensive
- Slower convergence

**Use Cases:**
- Binary classification output layer
- Rare in hidden layers (ReLU preferred)

---

### 4. Tanh (Hyperbolic Tangent)
Similar to sigmoid but with output range -1 to 1.

**Formula:**
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

**Range:** -1 to 1

**Code Example:**
```python
layers.Dense(128, activation='tanh')

# Standalone
output = tf.nn.tanh(input)
```

**Visual:**
```
1.0 |      ----
    |    /
0.0 | --
    |    \
-1.0|     ----
```

**Advantages:**
- Zero-centered output (better than sigmoid)
- Stronger gradients than sigmoid
- Symmetric around origin

**Disadvantages:**
- Still suffers from vanishing gradient
- Computationally expensive
- Slower convergence than ReLU

**Use Cases:**
- Hidden layers (better than sigmoid)
- RNNs and LSTMs
- When symmetric output is needed

---

### 5. Softmax
Used in multi-class classification output layer.

**Formula:**
```
f(xᵢ) = e^(xᵢ) / Σ(e^(xⱼ)) for all j
```

**Characteristics:**
- Outputs sum to 1 (probability distribution)
- Used only in output layer
- Applicable for multi-class (≥ 3 classes)

**Code Example:**
```python
# Multi-class classification
layers.Dense(10, activation='softmax')  # 10 classes

# Standalone
output = tf.nn.softmax(logits)

# Custom
def softmax(x):
    exp_x = tf.exp(x - tf.reduce_max(x))  # Numerical stability
    return exp_x / tf.reduce_sum(exp_x)
```

**Example:**
```
Input: [2.0, 1.0, 0.1]
Output: [0.7, 0.2, 0.1]  (probabilities)
```

**Advantages:**
- Produces valid probability distribution
- Perfect for multi-class classification
- Differentiable

**Disadvantages:**
- Only for output layer
- Computationally expensive

---

### 6. ELU (Exponential Linear Unit)
Smooth activation that can output negative values.

**Formula:**
```
f(x) = x if x > 0
f(x) = α × (e^x - 1) if x ≤ 0   (α typically 1.0)
```

**Code Example:**
```python
layers.ELU(alpha=1.0)
```

**Advantages:**
- Smooth for negative values
- Reduces bias shift
- Faster convergence than ReLU

---

### 7. SELU (Scaled ELU)
Self-normalizing activation function.

**Formula:**
```
f(x) = λ × x if x > 0
f(x) = λ × α × (e^x - 1) if x ≤ 0
where λ ≈ 1.0507 and α ≈ 1.6733
```

**Code Example:**
```python
layers.Dense(128, activation='selu')
```

**Advantages:**
- Self-normalizing properties
- Works well with deep networks
- Reduced vanishing gradient

**Disadvantage:**
- Requires specific weight initialization (lecun_normal)

---

### 8. Swish
Modern activation function discovered by Google.

**Formula:**
```
f(x) = x × sigmoid(x)
```

**Code Example:**
```python
# Option 1
layers.Dense(128, activation='swish')

# Option 2: Custom
def swish(x):
    return x * tf.nn.sigmoid(x)

layers.Lambda(swish)
```

**Advantages:**
- Self-gated mechanism
- Often outperforms ReLU
- Smooth and non-monotonic

---

### 9. Mish
Another modern activation function.

**Formula:**
```
f(x) = x × tanh(softplus(x))
```

**Code Example:**
```python
# Custom implementation
def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))

layers.Lambda(mish)
```

---

## Comparison Table

| Function | Range | Hidden/Output | Gradient Issues | Computation |
|----------|-------|---------------|-----------------|-------------|
| ReLU | 0 to ∞ | Hidden | Dead neurons | Fast |
| Leaky ReLU | -∞ to ∞ | Hidden | Better | Fast |
| Sigmoid | 0 to 1 | Output | Vanishing | Slow |
| Tanh | -1 to 1 | Hidden | Vanishing | Slow |
| Softmax | 0 to 1 | Output | None | Medium |
| ELU | -∞ to ∞ | Hidden | Better | Medium |
| SELU | -∞ to ∞ | Hidden | Better | Medium |
| Swish | -∞ to ∞ | Hidden | Good | Medium |

---

## Choosing Activation Functions

### For Hidden Layers
1. **Try ReLU first**: Fast, effective, standard
2. **If issues**: Try Leaky ReLU or ELU
3. **Deep networks**: Consider SELU or Swish

### For Output Layers
1. **Binary classification**: Sigmoid
2. **Multi-class**: Softmax
3. **Regression**: Linear (no activation)

---

## Implementation Example

```python
from tensorflow import keras
from tensorflow.keras import layers

# Different activations
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),     # ReLU
    layers.Dense(64, activation='relu'),                          # ReLU
    layers.Dense(32, activation='relu'),                          # ReLU
    layers.Dense(10, activation='softmax')                        # Softmax
])

# Alternative with Leaky ReLU
model = keras.Sequential([
    layers.Dense(128, input_shape=(784,)),
    layers.LeakyReLU(alpha=0.01),
    
    layers.Dense(64),
    layers.LeakyReLU(alpha=0.01),
    
    layers.Dense(32),
    layers.LeakyReLU(alpha=0.01),
    
    layers.Dense(10, activation='softmax')
])

# With multiple activation types
model = keras.Sequential([
    layers.Dense(128, input_shape=(784,)),
    layers.Activation('relu'),
    
    layers.Dense(64),
    layers.Activation('tanh'),
    
    layers.Dense(10, activation='softmax')
])
```

---

## Best Practices
1. **Use ReLU by default** for hidden layers
2. **Use Sigmoid** for binary classification output
3. **Use Softmax** for multi-class output
4. **Use Linear** for regression output
5. **Monitor gradient flow** to detect vanishing/exploding gradients
6. **Experiment with different activations** for your specific problem
7. **Consider computational cost** for real-time applications

---
