# NLP Lemmatization Guide

## Overview
**Lemmatization** is the process of converting words to their base dictionary form (lemma).
- Similar to stemming but produces valid dictionary words
- Returns a root word that has actual meaning
- More accurate than stemming but computationally heavier

## Key Difference from Stemming
```
Stemming:      eating, eats, eaten → eat (may not be valid)
Lemmatization: eating, eats, eaten → eat (always valid word)
```

---

## Setup Requirements
```python
import nltk
nltk.download('wordnet')  # Required for WordNetLemmatizer
```

---

## WordNet Lemmatizer

### Purpose
Uses WordNet corpus to find the root form of words.
Best for Q&A systems, chatbots, and text summarization.

### Code Example
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Without POS tag (defaults to noun)
lemmatizer.lemmatize("going")  # Output: 'going' (wrong - needs verb tag)

# With POS (Part-of-Speech) tag - CORRECT
lemmatizer.lemmatize("going", pos='v')  # Output: 'go'
```

---

## Part-of-Speech (POS) Tags

POS tags specify the word type for accurate lemmatization:

| Tag | POS | Example |
|-----|-----|---------|
| `'n'` | Noun | cat, book, dog |
| `'v'` | Verb | run, eat, go |
| `'a'` | Adjective | beautiful, big, small |
| `'r'` | Adverb | quickly, slowly, very |

### Example with Different POS
```python
lemmatizer.lemmatize("going", pos='v')       # Output: 'go'
lemmatizer.lemmatize("going", pos='n')       # Output: 'going'
lemmatizer.lemmatize("better", pos='a')      # Output: 'good'
lemmatizer.lemmatize("better", pos='r')      # Output: 'well'
```

---

## Batch Lemmatization Example

### Code
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ["eating", "eats", "eaten", "writing", "writes", 
         "programming", "programs", "history", "finally", "finalized"]

# Lemmatize with verb POS tag
for word in words:
    print(f"{word} ---> {lemmatizer.lemmatize(word, pos='v')}")

# Output:
# eating ---> eat
# eats ---> eat
# eaten ---> eat
# writing ---> write
# writes ---> write
# programming ---> program
# programs ---> program
# history ---> history (no change, already lemma)
# finally ---> finally (adverb, no change)
# finalized ---> finalize
```

---

## Combined Text Processing Pipeline

### Example: Clean and Lemmatize Text
```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Setup
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()

paragraph = """I have three visions for India. In 3000 years of our history, 
people from all over the world have come and invaded us..."""

# Tokenize sentences
sentences = nltk.sent_tokenize(paragraph)

# Apply stopwords removal and lemmatization
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    # Remove stopwords AND lemmatize
    words = [lemmatizer.lemmatize(word.lower(), pos='v') 
             for word in words 
             if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)

print(sentences)
```

---

## Lemmatization vs Stemming

| Aspect | Lemmatization | Stemming |
|--------|---------------|----------|
| **Output** | Valid dictionary word | May not be valid word |
| **Accuracy** | Higher | Lower |
| **Speed** | Slower | Faster |
| **Use Case** | Better for meaning | Volume processing |
| **Example** | "running" → "run" | "running" → "runn" |
| **Q&A Systems** | Better ✓ | Poor ✗ |

---

## Use Cases
- **Question-Answering systems**: Need accurate meaning
- **Chatbots**: Understand user intent properly
- **Text summarization**: Maintain semantic correctness
- **Information retrieval**: Match query with documents accurately
- **Sentiment analysis**: Understand emotion correctly

---

## Best Practices
1. Always specify POS tag for accurate results
2. Use after stopword removal for efficiency
3. Combine with tokenization for better results
4. Consider using both stemming and lemmatization based on need:
   - Fast processing → Stemming
   - Accuracy matters → Lemmatization
