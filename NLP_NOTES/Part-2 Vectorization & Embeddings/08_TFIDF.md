# NLP TF-IDF Guide

## Overview
**TF-IDF** (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects how important a word is to a document in a collection of documents.
- Weights words based on their importance
- Solves the limitation of Bag of Words
- Considers both word frequency and document rarity
- Produces more meaningful representations than simple counts

---

## Problem with Bag of Words
BoW treats all words equally:
- Common words (the, a, is) get high counts but low importance
- Rare but meaningful words get low counts but high importance
- TF-IDF solves this by weighting words

---

## Components of TF-IDF

### 1. Term Frequency (TF)
Measures how frequently a word appears in a document.

**Formula:**
```
TF(word, document) = (count of word in document) / (total words in document)
```

**Example:**
```
Document: "I love pizza. Pizza is delicious and I love pizza"
Total words: 9

TF(pizza) = 3/9 = 0.33
TF(I) = 2/9 = 0.22
TF(love) = 2/9 = 0.22
TF(the) = 0/9 = 0.0
```

### 2. Inverse Document Frequency (IDF)
Measures how rare or common a word is across all documents.

**Formula:**
```
IDF(word) = log(total documents / documents containing word)
```

**Example (5 documents total):**
```
IDF(pizza) = log(5/2) = 0.916    (appears in 2 documents - rare)
IDF(the) = log(5/5) = 0.0        (appears in all documents - common)
IDF(coding) = log(5/1) = 1.609   (appears in 1 document - very rare)
```

### 3. TF-IDF Score
Final importance score combining both metrics.

**Formula:**
```
TF-IDF(word, document) = TF(word, document) × IDF(word)
```

**Example:**
```
TF-IDF(pizza) = 0.33 × 0.916 = 0.302
TF-IDF(the) = 0.22 × 0.0 = 0.0       (common word, low importance)
TF-IDF(coding) = 0.11 × 1.609 = 0.177
```

---

## TF-IDF Implementation

### Using sklearn TfidfVectorizer

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Sample documents
corpus = [
    "I love pizza and coding",
    "I love coding more than pizza",
    "Pizza is delicious and I love it"
]

# Create TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=100)

# Fit and transform
X = tfidf.fit_transform(corpus).toarray()

# Display results with proper formatting
np.set_printoptions(edgeitems=30, linewidth=100000, 
                    formatter=dict(float=lambda x:"%.3g" % x))
print(X)

print(f"Shape: {X.shape}")  # (3, vocabulary_size)
print(f"Feature names: {tfidf.get_feature_names_out()}")
print(f"Vocabulary: {tfidf.vocabulary_}")
```

### Output Example
```
Document 1: [0.333  0.333  0.333  0.0    0.0    0.333  0.0    ...]
Document 2: [0.0    0.333  0.333  0.333  0.0    0.333  0.333  ...]
Document 3: [0.333  0.0    0.333  0.0    0.333  0.333  0.0    ...]

Feature names: ['and', 'coding', 'delicious', 'i', 'it', 'love', 'more', 'pizza', 'than']
```

---

## Parameters

```python
TfidfVectorizer(
    max_features=100,           # Keep top 100 most important words
    ngram_range=(1,1),          # (1,1) = unigrams, (2,2) = bigrams, etc
    lowercase=True,             # Convert to lowercase
    stop_words='english',       # Remove English stopwords
    min_df=1,                   # Minimum document frequency
    max_df=1.0,                 # Maximum document frequency
    sublinear_tf=False,         # Apply sublinear tf scaling (log normalization)
    norm='l2'                   # Apply L2 norm normalization
)
```

---

## Complete Preprocessing + TF-IDF Example

### Step 1: Clean Text
```python
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')

# Load data
messages = pd.read_csv('spamcollection/spam.csv', 
                       sep='\t', 
                       names=["label", "message"])

# Create corpus with lemmatization
corpus = []
lemmatizer = WordNetLemmatizer()

for i in range(len(messages)):
    # Remove special characters
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    
    # Convert to lowercase
    review = review.lower()
    
    # Split into words
    review = review.split()
    
    # Lemmatize and remove stopwords
    review = [lemmatizer.lemmatize(word, pos='v') 
              for word in review 
              if word not in stopwords.words('english')]
    
    # Join back
    review = ' '.join(review)
    corpus.append(review)
```

### Step 2: Apply TF-IDF
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectors
tfidf = TfidfVectorizer(max_features=100)
X = tfidf.fit_transform(corpus).toarray()

print(f"TF-IDF matrix shape: {X.shape}")
print(f"Feature names: {tfidf.get_feature_names_out()[:10]}")  # First 10 words
```

---

## TF-IDF with N-Grams

### Bigrams (2-word combinations)
```python
# Extract 2-word phrases
tfidf = TfidfVectorizer(max_features=100, ngram_range=(2,2))
X = tfidf.fit_transform(corpus).toarray()

# View bigram vocabulary
vocab = tfidf.vocabulary_
print(vocab)
# Output: {'love pizza': 0, 'pizza is': 1, 'is delicious': 2, ...}
```

### Example Bigram Output
```
Features: ['and i', 'delicious and', 'i love', 'love pizza', 'pizza is', ...]
Document 1: [0.408, 0.0, 0.408, 0.408, 0.0, ...]
Document 2: [0.0, 0.0, 0.408, 0.408, 0.0, ...]
Document 3: [0.408, 0.408, 0.408, 0.0, 0.408, ...]
```

---

## Advantages of TF-IDF
- **Weighs importance**: Rare words get higher weights
- **Ignores common words**: Stop words naturally get low scores
- **Better than BoW**: Produces more meaningful representations
- **Interpretable**: Can see which words are important
- **Fast**: Computationally efficient

## Disadvantages of TF-IDF
- **Ignores word order**: "dog bites man" vs "man bites dog" treated same
- **No semantics**: Similar words treated as different
- **Sparse vectors**: High dimensionality with many zeros
- **No context**: Doesn't capture surrounding context
- **Document length bias**: Longer documents may be favored

---

## Comparison: BoW vs TF-IDF

| Aspect | Bag of Words | TF-IDF |
|--------|-------------|--------|
| **Weights** | Word count | TF × IDF score |
| **Common words** | High value | Low value (natural) |
| **Rare words** | Low value | High value (if important) |
| **Interpretation** | Frequency | Importance |
| **Sparsity** | Sparse | More sparse |
| **Performance** | Good for simple tasks | Better for complex tasks |

---

## Use Cases
- **Document retrieval**: Search engines use TF-IDF ranking
- **Spam detection**: Identify important spam indicators
- **Text classification**: Feature extraction for ML models
- **Information retrieval**: Rank documents by relevance
- **Recommendation systems**: Content-based recommendations
- **Plagiarism detection**: Compare document similarity

### Spam Detection Example
```python
# Common spam words get highlighted by TF-IDF
# "Congratulations won lottery" → high TF-IDF score if rare across corpus
# "the and is" → low TF-IDF score (common across all emails)
```

---

## Accessing TF-IDF Values

### Get Important Words in Document
```python
# Get feature names
feature_names = tfidf.get_feature_names_out()

# Get TF-IDF scores for first document
document_idx = 0
tfidf_scores = X[document_idx]

# Get top 5 important words
top_indices = np.argsort(tfidf_scores)[-5:][::-1]
top_words = [feature_names[i] for i in top_indices]
print(f"Top words: {top_words}")
```

---
