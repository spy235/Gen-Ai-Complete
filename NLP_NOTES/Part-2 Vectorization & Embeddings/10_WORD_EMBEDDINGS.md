# NLP Word Embeddings Guide

## Overview
**Word Embeddings** are dense numerical representations of words that capture semantic meaning.
- Converts words into vectors in a continuous space
- Captures word relationships and similarities
- Dense vectors (unlike sparse BoW/TF-IDF)
- Enables semantic understanding between words
- Foundation for modern NLP (transformers, neural networks)

---

## Problem with Previous Methods

### Bag of Words & TF-IDF Limitations
```
Representation: [0, 0, 1, 0, 0, 1, ...]  ← Sparse, no meaning
Similarity: "king" and "queen" = completely different
Distance: "king" and "car" = same distance as "king" and "queen"
Context: No information about surrounding words
```

### Word Embeddings Solution
```
Representation: [0.25, -0.89, 0.12, 0.45, ...]  ← Dense, meaningful
Similarity: "king" and "queen" = very similar
Distance: "king" and "queen" = close, "king" and "car" = far
Context: Captures word relationships through training
```

---

## Key Concepts

### 1. Vector Space
Words are represented as points in a multi-dimensional space.
- Words with similar meanings are close together
- Dimensionality typically 100-300 for practical applications
- Each dimension captures semantic features (gender, tense, etc.)

### 2. Semantics
Word embeddings capture meaning relationships:
```
king - man + woman ≈ queen
Paris - France + Germany ≈ Berlin
happiness + beautiful ≈ (concept near joy)
```

### 3. Vector Operations
Mathematical operations on embeddings reveal linguistic relationships:
```
analogies: king - man + woman = queen
similarity: cosine_similarity(king, queen) ≈ 0.92
distance: euclidean_distance(king, queen) < distance(king, car)
```

---

## Word2Vec - Word Embedding Algorithm

### Overview
Word2Vec is an algorithm developed by Google (Mikolov et al., 2013) that learns word embeddings from text.
- Two architectures: Skip-gram and CBOW
- Unsupervised learning
- Fast and efficient
- Pre-trained models available

### Skip-gram Model
Predicts context words from a target word.

```
Input: "I love pizza"
Target word: "love"
Context words: ["I", "pizza"]

Model learns: love → predicts surrounding words
```

### CBOW (Continuous Bag of Words) Model
Predicts target word from context words.

```
Input: "I love pizza"
Context words: ["I", "pizza"]
Target word: "love"

Model learns: [I, pizza] → predicts love
```

---

## Using Pre-trained Word2Vec Models

### Installation
```bash
pip install gensim
```

### Loading Google News Word2Vec Model

```python
import gensim.downloader as api

# Load pre-trained Word2Vec model (Google News)
# Trained on 100 billion words from Google News dataset
# Vocabulary: 3 million words
# Dimensions: 300

wv = api.load('word2vec-google-news-300')

# Check if a word exists
print('king' in wv)  # True
```

### Getting Word Vectors

```python
# Get vector for a word
vec_king = wv['king']
print(vec_king)
# Output: [-0.03857422  0.08496094 -0.11450195 ... ] (300 dimensions)

# Check vector shape
print(vec_king.shape)  # (300,)

# Get vector for another word
vec_cricket = wv['cricket']
```

---

## Word Similarity Operations

### 1. Most Similar Words

```python
# Find words most similar to 'cricket'
similar_words = wv.most_similar('cricket')
print(similar_words)
# Output: [('cricket', 0.95), ('baseball', 0.82), ('sports', 0.78), ...]

# Find 10 most similar words
similar_words = wv.most_similar('happy', topn=10)

# Find words similar to 'happy' but exclude certain words
similar_words = wv.most_similar('happy', negative=['sad'])
```

### 2. Word Similarity Score

```python
# Cosine similarity between two words
similarity = wv.similarity("hockey", "sports")
print(similarity)  # Output: 0.85 (0-1 range, 1=identical)

# Similarity matrix (compare multiple word pairs)
sim_king_queen = wv.similarity("king", "queen")      # 0.92
sim_king_car = wv.similarity("king", "car")          # 0.15
sim_happy_sad = wv.similarity("happy", "sad")        # 0.25
```

### 3. Analogies and Arithmetic

```python
# Classic analogy: king - man + woman = queen
vec_analogy = wv['king'] - wv['man'] + wv['woman']

# Find most similar word to this vector
result = wv.most_similar([vec_analogy])
print(result)
# Output: [('queen', 0.89), ('princess', 0.82), ...]

# More analogies
# Paris - France + Germany = Berlin
vec = wv['Paris'] - wv['France'] + wv['Germany']
print(wv.most_similar([vec]))

# happy - sad + angry = ?
vec = wv['happy'] - wv['sad'] + wv['angry']
print(wv.most_similar([vec]))
```

---

## Complete Example: Word2Vec Analysis

### Step 1: Load Model
```python
import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors

# Load pre-trained model
wv = api.load('word2vec-google-news-300')

print(f"Vocabulary size: {len(wv)}")  # ~3 million words
```

### Step 2: Explore Relationships
```python
# Find similar words
print("Words similar to 'cricket':")
print(wv.most_similar('cricket', topn=5))
# Output: [('cricket', 0.95), ('baseball', 0.82), ('sports', 0.78), ...]

print("\nWords similar to 'happy':")
print(wv.most_similar('happy', topn=5))
```

### Step 3: Semantic Relationships
```python
# Test semantic relationships
print("Similarity scores:")
print(f"cricket & sports: {wv.similarity('cricket', 'sports'):.3f}")  # High
print(f"cricket & elephant: {wv.similarity('cricket', 'elephant'):.3f}")  # Low
print(f"king & queen: {wv.similarity('king', 'queen'):.3f}")  # High
print(f"king & car: {wv.similarity('king', 'car'):.3f}")  # Low
```

### Step 4: Solve Analogies
```python
# Solve word analogies
def solve_analogy(w1, w2, w3):
    """
    Solves: w1 is to w2 as w3 is to ?
    Example: king is to queen as man is to ?
    """
    vec = wv[w3] - wv[w1] + wv[w2]
    result = wv.most_similar([vec], topn=1)
    return result[0][0]

print(f"king is to queen as man is to: {solve_analogy('king', 'queen', 'man')}")
# Output: woman

print(f"Paris is to France as Berlin is to: {solve_analogy('Paris', 'France', 'Berlin')}")
# Output: Germany
```

---

## Training Custom Word2Vec Models

### From Corpus

```python
from gensim.models import Word2Vec
from nltk import sent_tokenize, word_tokenize

# Example corpus
corpus = [
    "I love pizza and coding",
    "I love coding more than pizza",
    "Pizza is delicious and I love it",
    "Coding is fun and challenging"
]

# Tokenize into sentences and words
sentences = [word_tokenize(sent.lower()) for sent in corpus]

# Train Word2Vec model
model = Word2Vec(
    sentences=sentences,
    vector_size=100,        # Dimension of word vectors
    window=5,               # Context window size
    min_count=1,            # Minimum word frequency
    workers=4,              # Number of threads
    sg=0                    # 0=CBOW, 1=Skip-gram
)

# Use trained model
print(model.wv['pizza'])  # Get vector
print(model.wv.most_similar('love', topn=5))  # Similar words
```

### Parameters Explained
```python
Word2Vec(
    sentences,              # List of tokenized sentences
    vector_size=100,        # Embedding dimension (100-300)
    window=5,               # Context window (words around target)
    min_count=1,            # Ignore words appearing < min_count times
    workers=4,              # Parallel processing threads
    sg=0,                   # 0=CBOW, 1=Skip-gram
    epochs=5                # Training iterations
)
```

---

## Advantages of Word Embeddings
- **Dense representation**: Captures relationships in continuous space
- **Semantic meaning**: Similar words are close together
- **Captures patterns**: Analogy relationships preserved
- **Low dimensionality**: 100-300 dimensions vs millions for BoW
- **Pre-trained models**: Can use without training
- **Transfer learning**: Works across different NLP tasks

## Disadvantages of Word Embeddings
- **Black box**: Difficult to interpret individual dimensions
- **Fixed for each word**: No handling of polysemy (word sense ambiguity)
  - "bank" (financial) vs "bank" (river) = same vector
- **Out-of-vocabulary words**: Unknown words not handled
- **Training data bias**: Reflects biases in training data
- **Context-independent**: Each word has single vector (solved by contextual embeddings)

---

## Word Embeddings vs Previous Methods

| Aspect | BoW | TF-IDF | Word2Vec |
|--------|-----|--------|----------|
| **Representation** | Sparse | Sparse | Dense |
| **Dimensions** | 10,000+ | 10,000+ | 100-300 |
| **Semantic** | No | No | Yes |
| **Context** | No | No | Yes |
| **Similarity** | Limited | Limited | Accurate |
| **Training** | None | On corpus | Pre-trained or train |
| **Analogies** | No | No | Yes |
| **Use Case** | Classification | Ranking | Deep learning |

---

## Beyond Word2Vec

### GloVe (Global Vectors)
- Combines global matrix factorization with local context windows
- Often performs better than Word2Vec

### FastText
- Extension of Word2Vec
- Handles out-of-vocabulary words using subwords
- Better for morphologically rich languages

### Contextual Embeddings (BERT, ELMo)
- Different vector for same word in different contexts
- Captures polysemy (multiple meanings)
- State-of-the-art for most NLP tasks

```python
# Example: "bank" has different meanings
# Word2Vec: bank = [0.25, -0.89, 0.12, ...]  (single vector)
# BERT: bank (financial) = different from bank (river) based on context
```

---

## Use Cases
- **Text classification**: Use embeddings as features
- **Semantic similarity**: Find similar documents
- **Recommendation systems**: Recommend similar products/content
- **Machine translation**: Capture language semantics
- **Question answering**: Match questions to answers semantically
- **Sentiment analysis**: Embeddings preserve emotional content
- **Named entity recognition**: Learn entity type patterns

---

## Best Practices
1. **Use pre-trained models**: Save training time and resources
2. **Combine with classifiers**: Embeddings are features, not final output
3. **Normalize vectors**: Many models use unit length vectors
4. **Choose right dimension**: 100 for simple, 300 for complex
5. **Consider domain**: Use domain-specific models when available
6. **Update for new data**: Retrain if domain significantly different

---

## References
- Mikolov et al. "Efficient Estimation of Word Representations in Vector Space" (2013)
- Word2Vec: https://code.google.com/archive/p/word2vec/
- Gensim: https://radimrehurek.com/gensim/
- Google News Word2Vec vectors

---
