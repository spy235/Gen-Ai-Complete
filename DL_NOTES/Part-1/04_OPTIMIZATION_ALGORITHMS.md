# Deep Learning: Optimization Algorithms

## Overview
**Optimization Algorithms** are methods to update neural network weights to minimize the loss function.
- Foundation of neural network training
- Different algorithms have different convergence properties
- Choice of optimizer significantly impacts training
- Modern algorithms (Adam, RMSprop) are often superior to basic SGD

---

## Gradient Descent Basics

### Standard Gradient Descent
Update weights in direction of negative gradient.

**Formula:**
```
w_new = w_old - learning_rate × ∂L/∂w
```

**Algorithm:**
```
1. Calculate gradient: ∂L/∂w
2. Update weight: w = w - α × ∂L/∂w
3. Repeat until convergence
```

**Code Example:**
```python
learning_rate = 0.01

for epoch in range(100):
    # Calculate gradient (via backpropagation)
    gradient = compute_gradient(model, X, y)
    
    # Update weights
    model.weights -= learning_rate * gradient
```

**Characteristics:**
- Deterministic (same gradient each iteration)
- Requires computing gradient on entire dataset
- Can be slow for large datasets
- Often gets stuck in local minima

---

## Stochastic Gradient Descent (SGD)

### Overview
Update weights using gradient from single sample instead of entire dataset.

**Formula:**
```
w_new = w_old - learning_rate × ∂L_i/∂w

where L_i is loss for single sample i
```

**Algorithm:**
```
1. Shuffle training data
2. For each sample:
   a. Calculate gradient on that sample
   b. Update weight immediately
3. Repeat until convergence
```

**Code Example:**
```python
from tensorflow.keras.optimizers import SGD

optimizer = SGD(learning_rate=0.01)

model.compile(optimizer=optimizer, loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=1)
```

**Characteristics:**
- Updates more frequently → faster convergence
- Noisy gradients (due to single sample) → helps escape local minima
- Can oscillate around minimum
- Less stable than batch GD

**Visual:**
```
Batch GD:    Smooth path to minimum
SGD:         Noisy, wandering path to minimum
```

---

## Mini-Batch Gradient Descent

### Overview
Update weights using gradient from small batch of samples (best of both worlds).

**Formula:**
```
w_new = w_old - learning_rate × (1/n_batch) × Σ(∂L_i/∂w)

where n_batch is batch size (typically 32-256)
```

**Code Example:**
```python
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

**Characteristics:**
- Balance between batch GD and SGD
- More stable than SGD, faster than batch GD
- Most commonly used
- Good generalization

**Typical Batch Sizes:**
```
Small datasets: 32-64
Medium datasets: 128-256
Large datasets: 256-512
```

---

## Momentum

### Overview
Accumulate gradients over time to build momentum.

**Formula:**
```
v_t = β × v_{t-1} + (1-β) × ∂L/∂w
w_new = w_old - learning_rate × v_t

where β is momentum (typically 0.9)
```

**Intuition:**
- Gradients like velocity accelerating down a hill
- Momentum accumulates like mass moving downward
- Helps overcome local minima

**Code Example:**
```python
from tensorflow.keras.optimizers import SGD

optimizer = SGD(learning_rate=0.01, momentum=0.9)

model.compile(optimizer=optimizer, loss='mse')
```

**Visual:**
```
Without Momentum:    Oscillates heavily around minimum
With Momentum:       Smoother convergence, speeds up
```

**Advantage:**
- Faster convergence
- Reduces oscillations
- Better handling of ravines

---

## Nesterov Accelerated Gradient (NAG)

### Overview
Look ahead before applying momentum.

**Formula:**
```
v_t = β × v_{t-1} + ∂L/∂(w - α × v_{t-1})
w_new = w_old - learning_rate × v_t
```

**Intuition:**
- First apply momentum, then calculate gradient
- More accurate gradient due to look-ahead

**Code Example:**
```python
from tensorflow.keras.optimizers import SGD

optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(optimizer=optimizer, loss='mse')
```

---

## AdaGrad (Adaptive Gradient)

### Overview
Adapt learning rate for each parameter based on historical gradients.

**Formula:**
```
g_t = ∂L/∂w
G_t = G_{t-1} + g_t²
w_new = w_old - (learning_rate / √(G_t + ε)) × g_t

where ε is small constant (typically 1e-7)
```

**Intuition:**
- Parameters with large gradients get smaller learning rate
- Parameters with small gradients get larger learning rate
- Different learning rate for each parameter

**Code Example:**
```python
from tensorflow.keras.optimizers import Adagrad

optimizer = Adagrad(learning_rate=0.01)

model.compile(optimizer=optimizer, loss='mse')
```

**Advantages:**
- Good for sparse data
- Less tuning needed
- Different rates per parameter

**Disadvantages:**
- Learning rate decreases monotonically
- Can become too small for large gradient accumulation

---

## RMSprop (Root Mean Square Propagation)

### Overview
Adaptive learning rate using exponential moving average of squared gradients.

**Formula:**
```
g_t = ∂L/∂w
E[g²]_t = β × E[g²]_{t-1} + (1-β) × g_t²
w_new = w_old - (learning_rate / √(E[g²]_t + ε)) × g_t

where β is decay rate (typically 0.9)
```

**Code Example:**
```python
from tensorflow.keras.optimizers import RMSprop

optimizer = RMSprop(learning_rate=0.001, decay=0.9)

model.compile(optimizer=optimizer, loss='mse')
```

**Advantages:**
- Maintains adaptive learning rates
- Learning rate doesn't decay to zero
- Works well in practice
- Good for RNNs

---

## Adam (Adaptive Moment Estimation)

### Overview
Combines momentum and RMSprop. Most popular optimizer in modern deep learning.

**Formula:**
```
g_t = ∂L/∂w
m_t = β₁ × m_{t-1} + (1-β₁) × g_t          (first moment estimate)
v_t = β₂ × v_{t-1} + (1-β₂) × g_t²         (second moment estimate)
m̂_t = m_t / (1-β₁^t)                       (bias corrected)
v̂_t = v_t / (1-β₂^t)                       (bias corrected)
w_new = w_old - learning_rate × m̂_t / √(v̂_t + ε)

where β₁=0.9, β₂=0.999, ε=1e-7 (default)
```

**Code Example:**
```python
from tensorflow.keras.optimizers import Adam

# Default
optimizer = Adam()

# Custom
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer=optimizer, loss='mse')
model.fit(X_train, y_train, epochs=50)
```

**Characteristics:**
- Combines best of momentum and RMSprop
- Adaptive learning rates per parameter
- Bias correction for early iterations
- Fast convergence

**Advantages:**
- Works well in most cases
- Requires minimal tuning
- Great for deep networks
- Convergence is usually fast and stable

**Disadvantages:**
- May not generalize as well as SGD in some cases
- Can overfit with smaller datasets

---

## AdamW (Adam with Decoupled Weight Decay)

### Overview
Adam with proper L2 regularization (weight decay).

**Difference from Adam:**
- Decouples weight decay from gradient-based update
- Better regularization effect

**Code Example:**
```python
from tensorflow.keras.optimizers import AdamW

optimizer = AdamW(learning_rate=0.001, weight_decay=0.0001)

model.compile(optimizer=optimizer, loss='mse')
```

---

## Learning Rate Scheduling

### Overview
Adjust learning rate during training.

**Common Strategies:**

**1. Step Decay**
```python
def lr_scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.5
    return lr

callback = keras.callbacks.LearningRateScheduler(lr_scheduler)
model.fit(X_train, y_train, callbacks=[callback], epochs=100)
```

**2. Exponential Decay**
```python
initial_lr = 0.1
decay_rate = 0.1
decay_steps = 1000

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=decay_steps,
    decay_rate=decay_rate
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

**3. Cosine Annealing**
```python
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.1,
    decay_steps=1000
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

---

## Optimizer Comparison

| Optimizer | Convergence | Memory | CPU | Tuning | Use Case |
|-----------|-------------|--------|-----|--------|----------|
| SGD | Slow | Low | Low | High | When compute limited |
| Momentum | Medium | Low | Low | Medium | Stable training |
| AdaGrad | Medium | Medium | Medium | Low | Sparse data |
| RMSprop | Fast | Medium | Medium | Low | RNNs |
| Adam | Very Fast | High | Medium | Low | General purpose |
| AdamW | Very Fast | High | Medium | Low | With regularization |

---

## Choosing an Optimizer

### Start with Adam
```python
optimizer = keras.optimizers.Adam()
```

### If overfitting is a problem
```python
# Use SGD with momentum (better generalization)
optimizer = keras.optimizers.SGD(momentum=0.9)

# Or use AdamW with weight decay
optimizer = keras.optimizers.AdamW(weight_decay=0.0001)
```

### For sparse data
```python
optimizer = keras.optimizers.Adagrad()
```

### For RNNs
```python
optimizer = keras.optimizers.RMSprop()
```

---

## Best Practices

1. **Start with Adam**: Good default choice
2. **Use learning rate scheduling**: Decay learning rate over time
3. **Monitor loss curves**: Watch for divergence or slow convergence
4. **Try different optimizers**: Empirically test for your problem
5. **Adjust hyperparameters carefully**: Small changes can matter
6. **Use warm restarts**: Reset learning rate periodically
7. **Gradient clipping**: Prevent exploding gradients

---

## References
- Kingma & Ba "Adam: A Method for Stochastic Optimization" (2014)
- Tieleman & Hinton "RMSprop" (2012)
- Rumelhart et al. "Learning with Momentum" (1986)

---
