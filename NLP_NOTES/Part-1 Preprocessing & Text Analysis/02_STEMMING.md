# NLP Stemming Guide

## Overview
**Stemming** is the process of reducing words to their root form or stem.
- Removes suffixes and prefixes
- Important for NLU (Natural Language Understanding) and NLP
- Useful for document classification and sentiment analysis

## Key Concept
```
Example: eating, eat, eaten → eat (root form)
Example: going, gone, goes → go (root form)
```

---

## 1. Porter Stemmer

### Purpose
Classic stemming algorithm that removes common suffixes using a set of rules.

### Code Example
```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["eating", "eats", "eaten", "writing", "writes", 
         "programming", "programs", "history", "finally", "finalized"]

# Stem each word
for word in words:
    print(f"{word} ---> {stemmer.stem(word)}")

# Output:
# eating ---> eat
# eats ---> eat
# eaten ---> eaten
# writing ---> write
# writes ---> write
# programming ---> program
# programs ---> program
# history ---> histori
# finally ---> final
# finalized ---> final

# Single word stemming
stemmer.stem('congratulate')  # Output: 'congratul'
stemmer.stem('sitting')       # Output: 'sit'
```

### Characteristics
- Rule-based approach
- Fast and efficient
- Sometimes produces non-real words (e.g., "histori" instead of "history")

---

## 2. Regexp Stemmer

### Purpose
Creates a stemmer using regular expression rules to remove suffixes.

### Code Example
```python
from nltk.stem import RegexpStemmer

# Define pattern: remove 'ing', 's', 'able', or 'e' suffixes (min word length 4)
stemmer = RegexpStemmer('ing|s$|able$|e$', min=4)

stemmer.stem("eating")      # Output: 'eat'
stemmer.stem("ingeating")   # Output: 'ingeat'
stemmer.stem("helping")     # Output: 'help'
```

### Parameters
- **Pattern**: Regex pattern for suffixes to remove
  - `'ing|s$|able$|e$'` - removes ing, s, able, or e at end
  - `$` = end of word anchor
  - `|` = OR operator
- **min**: Minimum word length (default 0)

### Characteristics
- Customizable using regex patterns
- More control than Porter Stemmer
- Can be tailored for specific domains

---

## 3. Snowball Stemmer

### Purpose
More advanced stemming algorithm, language-aware.
Improved version of Porter Stemmer that handles more cases better.

### Code Example
```python
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")

words = ["eating", "eats", "eaten", "writing", "writes", 
         "programming", "programs", "history", "finally", "finalized"]

# Stem each word
for word in words:
    print(f"{word} ---> {stemmer.stem(word)}")

# Output:
# eating ---> eat
# eats ---> eat
# eaten ---> eaten
# writing ---> write
# writes ---> write
# programming ---> program
# programs ---> program
# history ---> histori
# finally ---> final
# finalized ---> final
```

### Advantages over Porter Stemmer
- Better handling of irregular words
- Produces more meaningful stems
- Available for multiple languages

---

## Comparison Table

| Feature | Porter | Regexp | Snowball |
|---------|--------|--------|----------|
| **Speed** | Fast | Medium | Medium |
| **Accuracy** | Good | Variable | Better |
| **Customizable** | No | Yes | No |
| **Real words output** | Sometimes not | Yes | Better |
| **Use Case** | General | Domain-specific | Better quality |

---

## Use Cases
- **Product reviews**: Normalize variations (review, reviewing, reviewed)
- **Search engines**: Handle different word forms
- **Text classification**: Reduce feature space
- **Sentiment analysis**: Improve classification accuracy

---

## Important Note
Stemming produces non-word forms. If you need real dictionary words, use **Lemmatization** instead (see `03_LEMMATIZATION.md`).
