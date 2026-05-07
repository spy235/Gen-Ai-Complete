# NLP N-Grams Guide

## Overview
**N-Grams** are contiguous sequences of N words from a text document.
- Captures word order information (unlike BoW)
- Helps identify common word patterns
- Improves context understanding
- Used in language modeling and text prediction

---

## Key Concepts

### What are N-Grams?
N-Grams represent different sequence lengths:

**Example Text:** "I love pizza and coding"

- **Unigrams (1-gram)**: Individual words
  - Tokens: [I], [love], [pizza], [and], [coding]

- **Bigrams (2-gram)**: Pairs of consecutive words
  - Tokens: [I love], [love pizza], [pizza and], [and coding]

- **Trigrams (3-gram)**: Triplets of consecutive words
  - Tokens: [I love pizza], [love pizza and], [pizza and coding]

- **4-gram**: Four consecutive words
  - Tokens: [I love pizza and], [love pizza and coding]

---

## Why N-Grams Matter

### Problem with Unigrams (BoW)
```
Text 1: "dog bites man"
Text 2: "man bites dog"

BoW representation: SAME (word order lost)
```

### Solution with Bigrams
```
Text 1: [dog bites], [bites man]
Text 2: [man bites], [bites dog]

Bigrams: DIFFERENT (word order captured)
```

---

## N-Gram Types

### 1. Character N-Grams
N-grams at character level (rare in NLP, more for spell checking).

```python
Text: "python"
Character bigrams: ['py', 'yt', 'th', 'ho', 'on']
```

### 2. Word N-Grams
N-grams at word level (most common in NLP).

```python
Text: "machine learning is fun"
Word unigrams: ['machine', 'learning', 'is', 'fun']
Word bigrams: ['machine learning', 'learning is', 'is fun']
Word trigrams: ['machine learning is', 'learning is fun']
```

---

## Implementation with sklearn

### Unigrams (1-gram)

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love pizza and coding",
    "I love coding more than pizza",
    "Pizza is delicious and I love it"
]

# Unigrams only
cv = CountVectorizer(max_features=1000, binary=True, ngram_range=(1,1))
X = cv.fit_transform(corpus).toarray()

print(f"Vocabulary: {cv.vocabulary_}")
# Output: {'and': 0, 'coding': 1, 'delicious': 2, 'i': 3, 'it': 4, ...}
```

### Bigrams (2-gram)

```python
# Bigrams only
cv = CountVectorizer(max_features=1000, binary=True, ngram_range=(2,2))
X = cv.fit_transform(corpus).toarray()

vocab = cv.vocabulary_
print(vocab)
# Output: {'and coding': 0, 'and i': 1, 'coding more': 2, 'i love': 3, ...}
```

### Trigrams (3-gram)

```python
# Trigrams only
cv = CountVectorizer(max_features=1000, binary=True, ngram_range=(3,3))
X = cv.fit_transform(corpus).toarray()

vocab = cv.vocabulary_
print(vocab)
# Output: {'and i love': 0, 'i love coding': 1, 'love pizza and': 2, ...}
```

### Mixed N-Grams (Unigrams + Bigrams)

```python
# Combine unigrams and bigrams
cv = CountVectorizer(max_features=1000, binary=True, ngram_range=(1,2))
X = cv.fit_transform(corpus).toarray()

vocab = cv.vocabulary_
# Output includes both single words AND bigrams
# {'and': 0, 'and coding': 1, 'and i': 2, 'coding': 3, 
#  'coding more': 4, 'delicious': 5, ...}
```

---

## N-Gram Parameters

```python
CountVectorizer(
    ngram_range=(min_n, max_n),    # (1,1)=unigrams, (2,2)=bigrams, 
                                    # (1,2)=uni+bigrams, etc
    max_features=1000,              # Keep top features
    binary=True,                    # True=presence, False=count
    lowercase=True,                 # Convert to lowercase
    stop_words='english',           # Remove stopwords
    min_df=1,                       # Min document frequency
    max_df=1.0                      # Max document frequency
)
```

### ngram_range Explained
```python
ngram_range=(1,1)  → Unigrams only
ngram_range=(2,2)  → Bigrams only
ngram_range=(3,3)  → Trigrams only
ngram_range=(1,2)  → Unigrams + Bigrams
ngram_range=(1,3)  → Unigrams + Bigrams + Trigrams
ngram_range=(2,3)  → Bigrams + Trigrams
```

---

## Complete Example: N-Grams with Text Preprocessing

### Step 1: Clean Text
```python
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')

# Load data
messages = pd.read_csv('spamcollection/spam.csv', 
                       sep='\t', 
                       names=["label", "message"])

# Create corpus
corpus = []
ps = PorterStemmer()

for i in range(len(messages)):
    # Remove special characters
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    
    # Convert to lowercase
    review = review.lower()
    
    # Split into words
    review = review.split()
    
    # Stem and remove stopwords
    review = [ps.stem(word) for word in review 
              if word not in stopwords.words('english')]
    
    # Join back
    review = ' '.join(review)
    corpus.append(review)
```

### Step 2: Extract N-Grams

```python
from sklearn.feature_extraction.text import CountVectorizer

# Bigrams
cv_bigram = CountVectorizer(max_features=1000, binary=True, ngram_range=(2,2))
X_bigram = cv_bigram.fit_transform(corpus).toarray()

print(f"Bigram vocabulary size: {len(cv_bigram.vocabulary_)}")
print(f"Bigrams: {list(cv_bigram.vocabulary_.keys())[:10]}")

# Trigrams
cv_trigram = CountVectorizer(max_features=1000, binary=True, ngram_range=(3,3))
X_trigram = cv_trigram.fit_transform(corpus).toarray()

print(f"Trigram vocabulary size: {len(cv_trigram.vocabulary_)}")
print(f"Trigrams: {list(cv_trigram.vocabulary_.keys())[:10]}")
```

---

## N-Grams with TF-IDF

### Bigram TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF with bigrams
tfidf = TfidfVectorizer(max_features=100, ngram_range=(2,2))
X = tfidf.fit_transform(corpus).toarray()

print(f"Top bigrams: {tfidf.get_feature_names_out()}")

# Access vocabulary
vocab = tfidf.vocabulary_
print(vocab)
```

### Trigram TF-IDF

```python
# TF-IDF with trigrams
tfidf = TfidfVectorizer(max_features=100, ngram_range=(3,3))
X = tfidf.fit_transform(corpus).toarray()

print(f"Top trigrams: {tfidf.get_feature_names_out()}")
```

### Mixed TF-IDF

```python
# TF-IDF with unigrams and bigrams
tfidf = TfidfVectorizer(max_features=100, ngram_range=(1,2))
X = tfidf.fit_transform(corpus).toarray()

# Contains both single words and bigrams
features = tfidf.get_feature_names_out()
print(f"Total features: {len(features)}")
print(f"Sample features: {features[:20]}")
```

---

## Output Example

### Unigram Matrix
```
            and  coding  delicious  I  love  pizza
Document1   1    1       0         1  1     1
Document2   0    1       0         1  1     1
Document3   1    0       1         1  1     1
```

### Bigram Matrix
```
            and i  coding more  i love  love pizza  pizza is
Document1   1      0            1       1           0
Document2   0      1            1       1           0
Document3   1      0            1       1           1
```

### Trigram Matrix
```
            and i love  i love pizza  love pizza and  pizza is delicious
Document1   1           1             0               0
Document2   0           1             0               0
Document3   0           0             1               1
```

---

## Advantages of N-Grams
- **Captures context**: Word order matters
- **Language patterns**: Identifies common phrases
- **Better semantics**: "New York" vs "York New" captured
- **Collocations**: Identifies words that frequently occur together
- **Improved predictions**: Better for language modeling

## Disadvantages of N-Grams
- **Sparse vectors**: Increases dimensionality significantly
- **Data requirement**: Needs more training data
- **Computational cost**: More features = slower processing
- **Curse of dimensionality**: Bigrams/trigrams quickly increase features
- **Over-specific**: May capture noise instead of patterns

---

## Comparison: Unigrams vs Bigrams vs Trigrams

| Aspect | Unigrams | Bigrams | Trigrams |
|--------|----------|---------|----------|
| **Context** | None | Limited | More |
| **Features** | Few | More | Many |
| **Sparsity** | Least sparse | Sparser | Most sparse |
| **Interpretability** | High | Medium | Low |
| **Computational** | Fast | Slower | Slowest |
| **Data needed** | Less | More | Even more |
| **Effectiveness** | Good | Better | Can overfit |

---

## Use Cases

### Spam Detection
```
Bigram: "click here" → Often appears in spam
Trigram: "congratulations you won" → Common spam phrase
```

### Autocomplete / Predictive Text
```
User types: "I love"
Bigrams suggest: "I love pizza", "I love coding"
```

### Machine Translation
```
N-grams help capture language-specific phrase patterns
"New York" (bigram) → "Nueva York" (not word-by-word)
```

### Plagiarism Detection
```
Trigrams capture writing style patterns
Similar trigram sequences → Likely plagiarism
```

### Sentiment Analysis
```
Bigram: "not good" → Negative despite "good"
Unigram: "good" → Would be positive (loses negation)
```

---

## Best Practices
1. **Start with unigrams**: Baseline comparison
2. **Add bigrams cautiously**: Can increase dimensionality
3. **Limit trigrams**: Usually overkill unless needed
4. **Use with preprocessing**: Stemming/lemmatization helps
5. **Monitor sparsity**: Avoid overly sparse matrices
6. **Combine with other features**: Don't rely on N-grams alone

---
