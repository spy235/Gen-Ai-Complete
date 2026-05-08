# Deep Learning: Attention Mechanism and Transformers

## Overview
**Attention Mechanism** allows models to focus on relevant parts of input.
**Transformers** are models based entirely on attention, revolutionizing NLP and other domains.
- Enables parallel processing (faster than RNNs)
- Captures long-range dependencies
- Foundation of BERT, GPT, Claude, etc.
- State-of-the-art for most NLP tasks

---

## The Problem with RNNs

### Sequential Processing
```
RNN: x₁ → h₁ → x₂ → h₂ → x₃ → h₃ → x₄ → h₄

Problem: Must process sequentially (no parallelization)
Cannot compute h₂ until h₁ is computed
Slow for long sequences
```

### Limited Context
```
RNN: x₁ → h₁ → x₂ → h₂ → ... → x₁₀₀ → output

Problem: Information from x₁ heavily compressed
Difficult to maintain dependencies over 100 steps
```

---

## Attention Mechanism

### Core Idea
Pay different amounts of attention to different parts of input.

```
Query: What am I looking for?
Key: What information is available?
Value: What information should I use?

Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

### Simple Example

**Task:** Translate "I love pizza"

**Attention to English words:**
```
When translating "I":     Attend to "I" (100%)
When translating "aime":  Attend to "love" (100%)
When translating "pizza": Attend to "pizza" (100%)
```

**Attention to German words while translating:**
```
Output: "Ich liebe Pizza"
  ↑        ↑    ↑
Input:   "I"  "love" "pizza"
```

---

## Scaled Dot-Product Attention

### Mathematical Formula

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V

where:
Q = Query matrix (batch, seq_len, d_k)
K = Key matrix (batch, seq_len, d_k)
V = Value matrix (batch, seq_len, d_v)
d_k = dimension of key
```

### Step-by-Step

**1. Compute Attention Scores**
```
Scores = Q × K^T / √d_k

Result: (batch, seq_len_q, seq_len_k)
Score[i,j] = similarity between query i and key j
```

**2. Apply Softmax**
```
Weights = softmax(Scores)

Result: Probability distribution (sum to 1)
```

**3. Multiply by Values**
```
Output = Weights × V

Result: Weighted combination of values
```

### Example

```
Queries:     [[1, 0], [0, 1], [1, 1]]
Keys:        [[1, 0], [0, 1], [1, 1]]
Values:      [['a'], ['b'], ['c']]

Scores = Q × K^T = [[1, 0, 1],
                     [0, 1, 1],
                     [1, 1, 2]]

Softmax(Scores) ≈ [[0.5, 0.2, 0.3],
                    [0.2, 0.5, 0.3],
                    [0.2, 0.2, 0.6]]

Output = Weights × Values ≈ [['0.5a+0.2b+0.3c'],
                              ['0.2a+0.5b+0.3c'],
                              ['0.2a+0.2b+0.6c']]
```

---

## Multi-Head Attention

### Overview
Use multiple attention heads in parallel.

```
Input → Head 1 → Output 1
     ↓
     → Head 2 → Output 2
     ↓
     → Head 3 → Output 3
     ↓
     Concatenate → Linear → Final Output
```

### Benefits
- Different heads learn different attention patterns
- Richer representation
- More expressive than single head

### Mathematical Formula

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., head_h) × W^O

where:
head_i = Attention(Q × W_i^Q, K × W_i^K, V × W_i^V)
```

### Implementation

```python
from tensorflow import keras
from tensorflow.keras import layers

multi_head_attention = layers.MultiHeadAttention(
    num_heads=8,           # 8 attention heads
    key_dim=64,            # Dimension per head
    dropout=0.1
)

# Usage
output = multi_head_attention(
    query=x,
    value=x,
    key=x
)
```

---

## Transformer Architecture

### Overview
Stack of encoder and decoder layers using multi-head attention.

```
Input
  ↓
[Encoder Layer × 6] (with Multi-Head Attention + Feed-Forward)
  ↓
[Decoder Layer × 6] (with Masked Attention + Cross-Attention + Feed-Forward)
  ↓
Output
```

### Encoder Block

```
Input
  ↓
Multi-Head Attention
  ↓
Add & Normalize (Residual connection + Layer Norm)
  ↓
Feed-Forward Network (Dense layers)
  ↓
Add & Normalize
  ↓
Output
```

**Code:**
```python
def encoder_block(inputs, num_heads=8):
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(num_heads=num_heads)(inputs, inputs)
    attention_output = layers.Add()([inputs, attention_output])  # Residual
    attention_output = layers.LayerNormalization()(attention_output)
    
    # Feed-forward
    ffn_output = layers.Dense(512, activation='relu')(attention_output)
    ffn_output = layers.Dense(inputs.shape[-1])(ffn_output)
    ffn_output = layers.Add()([attention_output, ffn_output])  # Residual
    ffn_output = layers.LayerNormalization()(ffn_output)
    
    return ffn_output
```

### Decoder Block

Similar to encoder but with:
1. **Masked Multi-Head Attention**: Can't see future tokens
2. **Cross-Attention**: Attend to encoder output
3. **Feed-Forward**

---

## Complete Transformer Implementation

### Simple Transformer for Translation

```python
from tensorflow import keras
from tensorflow.keras import layers

def create_transformer(
    vocab_size, max_seq_len, embed_dim=512, num_heads=8, 
    num_layers=6, ffn_dim=2048
):
    # Encoder
    encoder_inputs = layers.Input(shape=(max_seq_len,))
    encoder_embedding = layers.Embedding(vocab_size, embed_dim)(encoder_inputs)
    encoder_embedding += layers.Embedding(max_seq_len, embed_dim)(
        tf.range(max_seq_len)
    )  # Positional encoding
    
    encoder_output = encoder_embedding
    for _ in range(num_layers):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(num_heads, embed_dim // num_heads)(
            encoder_output, encoder_output
        )
        attn_output = layers.Add()([encoder_output, attn_output])
        attn_output = layers.LayerNormalization()(attn_output)
        
        # Feed forward
        ffn_output = layers.Dense(ffn_dim, activation='relu')(attn_output)
        ffn_output = layers.Dense(embed_dim)(ffn_output)
        encoder_output = layers.Add()([attn_output, ffn_output])
        encoder_output = layers.LayerNormalization()(encoder_output)
    
    # Decoder (similar structure)
    decoder_inputs = layers.Input(shape=(max_seq_len,))
    decoder_embedding = layers.Embedding(vocab_size, embed_dim)(decoder_inputs)
    decoder_embedding += layers.Embedding(max_seq_len, embed_dim)(
        tf.range(max_seq_len)
    )
    
    decoder_output = decoder_embedding
    for _ in range(num_layers):
        # Self-attention
        attn_output = layers.MultiHeadAttention(num_heads, embed_dim // num_heads)(
            decoder_output, decoder_output
        )
        attn_output = layers.Add()([decoder_output, attn_output])
        attn_output = layers.LayerNormalization()(attn_output)
        
        # Cross-attention
        cross_attn = layers.MultiHeadAttention(num_heads, embed_dim // num_heads)(
            attn_output, encoder_output
        )
        cross_attn = layers.Add()([attn_output, cross_attn])
        cross_attn = layers.LayerNormalization()(cross_attn)
        
        # Feed forward
        ffn_output = layers.Dense(ffn_dim, activation='relu')(cross_attn)
        ffn_output = layers.Dense(embed_dim)(ffn_output)
        decoder_output = layers.Add()([cross_attn, ffn_output])
        decoder_output = layers.LayerNormalization()(decoder_output)
    
    # Output
    outputs = layers.Dense(vocab_size, activation='softmax')(decoder_output)
    
    model = keras.Model([encoder_inputs, decoder_inputs], outputs)
    return model

# Create and compile
model = create_transformer(vocab_size=10000, max_seq_len=100)
model.compile(optimizer='adam', loss='categorical_crossentropy')
```

---

## Pre-trained Transformer Models

### Using Hugging Face Transformers

```python
# Not with pure TensorFlow, but conceptually:
# from transformers import AutoTokenizer, AutoModel

# BERT (Bidirectional Encoder)
# - Pre-trained on masked language modeling
# - Great for understanding (classification, NER)

# GPT (Generative Pre-trained Transformer)
# - Pre-trained on next token prediction
# - Great for generation

# T5 (Text-to-Text Transfer Transformer)
# - Unified framework for multiple tasks
```

### Key Pre-trained Models
- **BERT**: Understanding tasks
- **GPT**: Generation tasks
- **T5**: Text-to-text
- **RoBERTa**: Improved BERT
- **ALBERT**: Lighter BERT
- **ELECTRA**: Discriminator-based
- **DistilBERT**: Faster, smaller BERT

---

## Positional Encoding

### Problem
Attention mechanism has no notion of position.

```
"dog bites man" and "man bites dog" would be treated the same
Positions matter!
```

### Solution: Positional Encoding
Add position-dependent patterns to embeddings.

**Formula:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

where:
pos = position in sequence
i = dimension
d = embedding dimension
```

**Implementation:**
```python
def positional_encoding(max_seq_len, embed_dim):
    import numpy as np
    
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(embed_dim)[np.newaxis, :]
    
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / embed_dim)
    
    pe = np.zeros((max_seq_len, embed_dim))
    pe[:, 0::2] = np.sin(pos * angle_rates[:, 0::2])
    pe[:, 1::2] = np.cos(pos * angle_rates[:, 1::2])
    
    return tf.constant(pe, dtype=tf.float32)

# Usage
pe = positional_encoding(max_seq_len=100, embed_dim=512)
```

---

## Advantages of Transformers

1. **Parallelization**: Process all tokens simultaneously
2. **Long-range dependencies**: Attention has direct connection
3. **Pre-training**: Works well with massive unlabeled data
4. **Transfer learning**: Fine-tune on downstream tasks
5. **Scalability**: Works well with large datasets
6. **Interpretability**: Attention weights show what model attends to

---

## Use Cases

### NLP
- Machine translation
- Text summarization
- Question answering
- Named entity recognition
- Sentiment analysis
- Text generation

### Vision
- Image classification
- Object detection
- Image captioning

### Multimodal
- Vision + Language tasks
- Video understanding

---

## Best Practices

1. **Use pre-trained models**: BERT, GPT, etc.
2. **Fine-tune on your data**: Transfer learning
3. **Proper tokenization**: Use standard tokenizers
4. **Layer normalization**: Helps training
5. **Gradient clipping**: Prevent instability
6. **Monitor attention weights**: Interpretability

---

## References
- Vaswani et al. "Attention is All You Need" (2017)
- Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
- Radford et al. "Language Models are Unsupervised Multitask Learners" (2019)

---
