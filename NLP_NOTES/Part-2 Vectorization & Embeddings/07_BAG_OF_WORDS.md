# NLP Bag of Words Guide

## Overview
**Bag of Words (BoW)** converts text documents into numerical vectors for machine learning models.
- Represents text as a collection of words (ignoring word order)
- Counts the frequency of each word in the document
- Creates a vocabulary of all unique words across documents
- Essential technique for converting text to numerical format

## Text to Vectors - Why Necessary
Machine learning algorithms require numerical input, not text. Bag of Words bridges this gap:
- **Input**: Raw text documents
- **Output**: Numerical vectors (matrices)
- **Purpose**: Enable ML models to process text data

---

## Key Concepts

### 1. Corpus
A collection of text documents used for analysis.
```
Example: [
    "I love pizza",
    "I love coding",
    "Pizza is delicious"
]
```

### 2. Vocabulary
A dictionary of all unique words in the corpus.
```
Vocabulary: {pizza, love, I, coding, is, delicious}
Size: 6 words
```

### 3. Document Vector
A numerical representation where each position represents word count.
```
Document: "I love pizza"
Vector: [1, 1, 1, 0, 0, 0]
        (I, love, pizza, coding, is, delicious)
```

---

## 1. One-Hot Encoding

### Purpose
Converts categorical variables (words) into binary vectors.
- Each unique word gets its own column
- Value is 1 if word is present, 0 if absent
- **Problem**: Ignores word frequency

### Code Example
```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Sample text
documents = [
    "I love pizza",
    "I love coding",
    "Pizza is delicious"
]

# One-hot encoding
# Output shape: (3, 6) - 3 documents, 6 unique words
# Each row is a one-hot encoded document
```

### Characteristics
- Binary representation only
- Does not capture word frequency
- Sparse vectors (mostly zeros)
- Useful for presence/absence detection

---

## 2. Bag of Words (CountVectorizer)

### Purpose
Converts text documents to a matrix where each value is the count/frequency of words.
- Captures word frequency
- Creates a sparse matrix representation
- Default method for BoW implementation

### Code Example
```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
corpus = [
    "I love pizza and coding",
    "I love coding more than pizza",
    "Pizza is delicious and I love it"
]

# Create Count Vectorizer
cv = CountVectorizer(max_features=100, binary=False)

# Fit and transform corpus
X = cv.fit_transform(corpus).toarray()

# X shape: (3, vocabulary_size)
# Each cell contains word count in that document
```

### Output Example
```
            and  coding  delicious  I  it  love  more  pizza  than
Document1   1    1       0         1  0   1     0     1      0
Document2   0    1       0         1  0   1     1     1      1
Document3   1    0       1         1  1   1     0     1      0
```

### Parameters
```python
CountVectorizer(
    max_features=100,           # Keep top 100 most frequent words
    binary=False,               # False = count, True = presence/absence
    ngram_range=(1,1),          # (1,1) = unigrams only
    stop_words='english',       # Remove common English words
    lowercase=True,             # Convert to lowercase
    min_df=1,                   # Minimum document frequency
    max_df=1.0                  # Maximum document frequency
)
```

---

## 3. Binary Bag of Words

### Purpose
Similar to BoW but uses binary values (0 or 1) instead of counts.
- Only indicates presence or absence
- Useful when word frequency doesn't matter
- Faster computation

### Code Example
```python
from sklearn.feature_extraction.text import CountVectorizer

# Binary Bag of Words
cv = CountVectorizer(max_features=100, binary=True)
X = cv.fit_transform(corpus).toarray()

# Output has only 0s and 1s
```

---

## Complete Preprocessing Pipeline Example

### Step 1: Load and Clean Data
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

### Step 2: Create Bag of Words
```python
from sklearn.feature_extraction.text import CountVectorizer

# Create vocabulary and transform
cv = CountVectorizer(max_features=100, binary=True)
X = cv.fit_transform(corpus).toarray()

print(f"Shape: {X.shape}")  # (num_documents, 100)
print(f"Vocabulary size: {len(cv.vocabulary_)}")
```

---

## Accessing Vocabulary

### View Word-Index Mapping
```python
# Get vocabulary dictionary
vocab = cv.vocabulary_
print(vocab)
# Output: {'word1': 0, 'word2': 1, 'word3': 2, ...}

# Get feature names (words)
feature_names = cv.get_feature_names_out()
print(feature_names)
# Output: ['word1', 'word2', 'word3', ...]

# Get count of specific word in document
word_index = vocab['pizza']
count_in_doc1 = X[0, word_index]
```

---

## Advantages of Bag of Words
- Simple and easy to understand
- Fast to compute
- Works well with many ML algorithms
- Captures word frequency information
- Interpretable results

## Disadvantages of Bag of Words
- **Ignores word order**: "dog bites man" vs "man bites dog" treated same
- **Ignores semantics**: Similar words treated differently
- **High dimensionality**: Vocabulary can be huge
- **Sparse vectors**: Most values are zero
- **No context**: Doesn't capture meaning

## Better Alternatives
- **TF-IDF**: Weighted word frequencies
- **Word Embeddings**: Dense vector representations
- **Word2Vec**: Semantic word relationships

---

## Use Cases
- **Spam detection**: Identify spam emails using word patterns
- **Document classification**: Categorize documents by content
- **Sentiment analysis**: Determine positive/negative sentiment
- **Information retrieval**: Search and ranking systems
- **Text similarity**: Compare document similarity

---
