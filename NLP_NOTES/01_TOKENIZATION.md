# NLP Tokenization Guide

## Overview
Tokenization is the process of breaking text into smaller pieces called **tokens**.
- **Tokens** can be words, sentences, or subwords
- Usually the first step in text processing

## NLP Data Structure Hierarchy
```
Corpus           = Entire library (collection of text)
Documents        = Individual books within the corpus
Sentences        = Paragraphs broken into sentences
Words            = Individual words in a sentence
Vocabulary       = Dictionary of unique words used in the corpus
```

## Setup Requirements
```python
import nltk
# Download required resources
nltk.download('punkt_tab')  # For sentence tokenization
```

---

## 1. Sentence Tokenization

### Purpose
Break a paragraph or document into individual sentences.

### Code Example
```python
from nltk import sent_tokenize

corpus = ''' A corpus is a large collection of text (or speech) used for analysis.
Think of it as a dataset of language.
It can include books, articles, emails, tweets, etc.
Used to train or evaluate language models.
Example: A dataset of 10,000 news articles is a corpus
'''

# Tokenize into sentences
document = sent_tokenize(corpus)
print(document)
# Output: ['A corpus is a large collection...', 'Think of it as...', ...]

# Iterate through sentences
for sentence in document:
    print(sentence)
```

---

## 2. Word Tokenization

### Purpose
Break text (paragraph or sentence) into individual words.

### Code Example
```python
from nltk.tokenize import word_tokenize

# Tokenize entire corpus into words
words = word_tokenize(corpus)
print(words)
# Output: ['A', 'corpus', 'is', 'a', 'large', 'collection', ...]

# Tokenize each sentence into words
for sentence in document:
    print(word_tokenize(sentence))
```

---

## 3. Wordpunct Tokenization

### Purpose
Similar to word tokenization but separates punctuation as distinct tokens.

### Code Example
```python
from nltk.tokenize import wordpunct_tokenize

result = wordpunct_tokenize(corpus)
print(result)
# Output: ['A', 'corpus', 'is', 'a', 'large', 'collection', 'of', 'text', '(', 'or', 'speech', ')', ...]
```

---

## 4. TreeBank Word Tokenization

### Purpose
Advanced tokenization that handles punctuation and special cases like contractions.
Includes punctuation as separate tokens from words.

### Code Example
```python
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
result = tokenizer.tokenize(corpus)
print(result)
# Output: ['A', 'corpus', 'is', 'a', 'large', 'collection', 'of', 'text', '(', 'or', 'speech', ')', ...]
```

---

## Key Differences

| Method | Handles Punctuation | Use Case |
|--------|-------------------|----------|
| `word_tokenize()` | Separates most | General purpose |
| `wordpunct_tokenize()` | Separates all | Punctuation-aware |
| `TreebankWordTokenizer()` | Separates with special rules | Linguistic analysis |
| `sent_tokenize()` | N/A (for sentences) | Sentence-level splitting |

---

## Common Use Cases
- **Text classification**: Tokenize into words/sentences for feature extraction
- **Language models**: Prepare text for training by tokenizing
- **Text analysis**: Break down documents for analysis
- **NLP pipelines**: First step in most NLP processing chains
