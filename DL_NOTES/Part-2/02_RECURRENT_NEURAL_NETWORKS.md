# Deep Learning: Recurrent Neural Networks (RNNs)

## Overview
**Recurrent Neural Networks (RNNs)** process sequential data by maintaining internal state/memory.
- Designed for time-series, text, speech, and other sequences
- Can process variable-length inputs
- Share weights across time steps
- Capture temporal dependencies
- Foundation for NLP and speech tasks

---

## Sequential Data

### Examples
```
Text: "I love pizza" (sequence of words)
Time-series: Stock prices over time
Speech: Audio frames over time
Video: Frame sequence
DNA: Sequence of nucleotides
```

### Why RNNs?
Regular neural networks assume independent samples. RNNs leverage temporal/sequential relationships.

---

## How RNNs Work

### Recurrent Structure
Pass hidden state from one time step to next.

```
Time Step t:
input[t] → [RNN Cell] → output[t]
           ↑        ↓
      hidden[t-1] → hidden[t]

hidden[t] = f(input[t], hidden[t-1])
output[t] = g(hidden[t])
```

### Mathematical Formula
```
h_t = tanh(W_h × h_{t-1} + W_x × x_t + b_h)
y_t = W_y × h_t + b_y

where:
h_t = hidden state at time t
x_t = input at time t
W_h, W_x, W_y = weight matrices
```

### Unrolled Through Time
```
t=0           t=1           t=2           t=3
x₀ → [RNN] → x₁ → [RNN] → x₂ → [RNN] → x₃ → [RNN]
     ↓            ↓            ↓            ↓
     y₀           y₁           y₂           y₃
```

---

## RNN Implementation

### Using Keras

```python
from tensorflow.keras import layers

# SimpleRNN
model = keras.Sequential([
    layers.SimpleRNN(64, activation='relu', input_shape=(timesteps, input_features)),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Input Shape
```python
# (batch_size, timesteps, features)
# Example: 32 samples, 100 time steps, 50 features
input_shape = (32, 100, 50)

layers.SimpleRNN(64, input_shape=(100, 50))
```

---

## Sequence-to-Sequence Tasks

### Many-to-One (Classification)
```
Sequence: [word1, word2, word3, word4]
Output:   Sentiment label

RNN:
x₁ → h₁ → 
x₂ → h₂ → 
x₃ → h₃ → Dense → Sentiment
x₄ → h₄ ↓
```

**Code:**
```python
model = keras.Sequential([
    layers.SimpleRNN(64, activation='relu', input_shape=(4, embed_dim)),
    layers.Dense(10, activation='softmax')  # Only output at end
])
```

### Many-to-Many (Tagging)
```
Sequence: [word1, word2, word3, word4]
Output:   [tag1, tag2, tag3, tag4]

RNN:
x₁ → h₁ → Dense → y₁
x₂ → h₂ → Dense → y₂
x₃ → h₃ → Dense → y₃
x₄ → h₄ → Dense → y₄
```

**Code:**
```python
model = keras.Sequential([
    layers.SimpleRNN(64, activation='relu', return_sequences=True, input_shape=(4, embed_dim)),
    layers.Dense(num_tags, activation='softmax')  # Output at each step
])
```

### Sequence-to-Sequence (Translation)
```
Input:  [word1, word2, word3] → Encoder
                                   ↓
                              Context vector
                                   ↓
Output: [mot1, mot2, mot3, mot4] ← Decoder
```

**Code:**
```python
# Encoder
encoder_input = layers.Input(shape=(None, embed_dim))
encoder_rnn = layers.SimpleRNN(64, return_state=True)
_, encoder_state = encoder_rnn(encoder_input)

# Decoder
decoder_input = layers.Input(shape=(None, embed_dim))
decoder_rnn = layers.SimpleRNN(64, return_sequences=True)
decoder_output = decoder_rnn(decoder_input, initial_state=encoder_state)
decoder_dense = layers.Dense(vocab_size, activation='softmax')
decoder_output = decoder_dense(decoder_output)

model = keras.Model([encoder_input, decoder_input], decoder_output)
```

---

## Vanishing Gradient Problem

### The Problem
Gradients become exponentially small through time steps.

```
∂L/∂h₀ = ∂L/∂h_T × ∂h_T/∂h_{T-1} × ... × ∂h₁/∂h₀
       = ∏(W^T × diag(tanh'(h_t)))

If W has small eigenvalues: exponentially small
If W has large eigenvalues: exponentially large
```

### Consequence
- Early time steps don't affect loss
- Can't learn long-term dependencies
- Network ignores distant context

### Solution: LSTM/GRU
Use gating mechanisms to control information flow.

---

## LSTM (Long Short-Term Memory)

### Overview
Specialized RNN cell with gates to control information flow.

```
Cell State (memory)
        ↑    ↓
    [Gate] [Gate] [Gate]
        ↑    ↑    ↑
    [x_t] [h_{t-1}]
```

### Key Components

**1. Forget Gate**
Decide what to forget from cell state.
```
f_t = σ(W_f × [h_{t-1}, x_t] + b_f)
```

**2. Input Gate**
Decide what new information to add.
```
i_t = σ(W_i × [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_c × [h_{t-1}, x_t] + b_c)
```

**3. Cell State Update**
Update memory with forget and input gates.
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
```

**4. Output Gate**
Decide what to output.
```
o_t = σ(W_o × [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

### LSTM Implementation

```python
from tensorflow.keras import layers

model = keras.Sequential([
    layers.LSTM(64, activation='relu', return_sequences=False, input_shape=(timesteps, features)),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### With Multiple LSTM Layers

```python
model = keras.Sequential([
    layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(timesteps, features)),
    layers.LSTM(32, activation='relu', return_sequences=False),
    layers.Dense(10, activation='softmax')
])
```

---

## GRU (Gated Recurrent Unit)

### Overview
Simplified LSTM with fewer parameters.

### Components
Similar to LSTM but with 2 gates instead of 3:

**1. Reset Gate**
```
r_t = σ(W_r × [h_{t-1}, x_t])
```

**2. Update Gate**
```
u_t = σ(W_u × [h_{t-1}, x_t])
```

**3. Candidate Hidden State**
```
h̃_t = tanh(W × [r_t ⊙ h_{t-1}, x_t])
```

**4. New Hidden State**
```
h_t = (1 - u_t) ⊙ h̃_t + u_t ⊙ h_{t-1}
```

### Implementation

```python
model = keras.Sequential([
    layers.GRU(64, activation='relu', return_sequences=True, input_shape=(timesteps, features)),
    layers.GRU(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

---

## Bidirectional RNNs

### Overview
Process sequence forwards and backwards.

```
Forward:  →  →  →  →
Sequence: [x₁, x₂, x₃, x₄]
Backward: ←  ←  ←  ←

Concat hidden states
```

### Implementation

```python
model = keras.Sequential([
    layers.Bidirectional(layers.LSTM(64, return_sequences=True), 
                        input_shape=(timesteps, features)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(10, activation='softmax')
])
```

### Output Shape
```
Regular RNN output: (batch, timesteps, 64)
Bidirectional output: (batch, timesteps, 128)  # 64×2
```

---

## Text Processing with RNNs

### Tokenization and Embedding

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = ["I love pizza", "I love coding", "Pizza is delicious"]
labels = [1, 1, 0]

# Tokenize
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
sequences = pad_sequences(sequences, maxlen=50)

# Build model
model = keras.Sequential([
    layers.Embedding(1000, 64, input_length=50),
    layers.LSTM(32),
    layers.Dense(1, activation='sigmoid')
])
```

---

## Use Cases

### Natural Language Processing
- Sentiment analysis
- Machine translation
- Text generation
- Named entity recognition
- Question answering

### Time Series Prediction
- Stock price forecasting
- Weather prediction
- Energy consumption forecasting

### Speech Recognition
- Audio to text conversion

### Video Understanding
- Action recognition
- Video captioning

---

## RNN vs LSTM vs GRU

| Aspect | RNN | LSTM | GRU |
|--------|-----|------|-----|
| **Vanishing Gradient** | Severe | Handled | Handled |
| **Parameters** | Few | Many | Medium |
| **Training Speed** | Fast | Slow | Medium |
| **Long Dependencies** | Poor | Excellent | Excellent |
| **Default Choice** | No | Yes | Yes (lighter) |

---

## Best Practices

1. **Use LSTM/GRU by default**: Better than SimpleRNN
2. **Use Bidirectional**: If order not critical
3. **Proper padding**: Pad sequences to same length
4. **Embedding layer**: Convert integers to vectors
5. **Layer normalization**: Help with training
6. **Dropout between layers**: Prevent overfitting
7. **Use gradient clipping**: Prevent exploding gradients

---

## References
- Hochreiter & Schmidhuber "LSTM" (1997)
- Cho et al. "GRU" (2014)
- Goodfellow et al. "Deep Learning" (MIT Press)

---
