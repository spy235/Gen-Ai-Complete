# NLP Stop Words Guide

## Overview
**Stop words** are common words that carry little meaningful information.
- Examples: "the", "is", "and", "or", "in", "at", "a", "an"
- Usually removed before processing to reduce noise
- Can improve efficiency and accuracy of NLP tasks

## Purpose
Stop words are used to:
- Reduce noise in text data
- Focus on meaningful words only
- Improve processing speed
- Improve classification accuracy

---

## Setup Requirements
```python
import nltk
nltk.download('stopwords')  # Required for stopwords corpus
```

---

## Get English Stop Words

### Code Example
```python
from nltk.corpus import stopwords

# Get all English stop words
stop_words = stopwords.words('english')
print(stop_words)
# Output: ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
#          'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 
#          'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
#          "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 
#          'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
#          'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
#          'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
#          'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
#          'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', ...]

print(f"Total stop words: {len(stop_words)}")
# Output: Total stop words: 179
```

---

## Remove Stop Words from Text

### Basic Example
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = "The quick brown fox jumps over the lazy dog"
words = word_tokenize(text)

# Convert to set for fast lookup
stop_words = set(stopwords.words('english'))

# Remove stop words
filtered_words = [word for word in words 
                  if word.lower() not in stop_words]

print(filtered_words)
# Output: ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']
```

---

## Complete Pipeline: Tokenization → Stopword Removal → Stemming/Lemmatization

### Full Text Processing Example
```python
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Setup
nltk.download('stopwords')
stemmer = PorterStemmer()

paragraph = """I have three visions for India. In 3000 years of our history, 
people from all over the world have come and invaded us, captured our lands, 
conquered our minds."""

# Step 1: Tokenize into sentences
sentences = nltk.sent_tokenize(paragraph)

# Step 2: Process each sentence
for i in range(len(sentences)):
    # Tokenize words
    words = nltk.word_tokenize(sentences[i])
    
    # Apply stopwords removal AND stemming
    words = [stemmer.stem(word) 
             for word in words 
             if word.lower() not in set(stopwords.words('english'))]
    
    # Join back into sentence
    sentences[i] = ' '.join(words)

print(sentences)
```

---

## Advanced: Custom Stop Words

### Extend or Customize Stop Words
```python
from nltk.corpus import stopwords

# Get default English stop words
stop_words = set(stopwords.words('english'))

# Add custom stop words
custom_stop_words = {'domain', 'specific', 'words'}
stop_words.update(custom_stop_words)

# Or remove certain stop words if needed
stop_words.discard('not')  # Keep 'not' for sentiment analysis
stop_words.discard('no')   # Keep 'no' for sentiment analysis

# Use in filtering
filtered = [word for word in words 
            if word.lower() not in stop_words]
```

---

## Stop Words in Different Languages

### Code Example
```python
from nltk.corpus import stopwords

# Available languages
available_languages = stopwords.fileids()
print(available_languages)
# Output: ['arabic', 'asturian', 'basque', 'catalan', 'czech', 'danish', 
#          'dutch', 'english', 'finnish', 'french', 'galician', 'german', 
#          'greek', 'hungarian', 'italian', 'norwegian', 'portuguese', 
#          'romanian', 'russian', 'spanish', 'swedish', 'turkish']

# Get Spanish stop words
spanish_stops = stopwords.words('spanish')
print(spanish_stops[:10])
```

---

## Use Cases

| Use Case | Stop Words | Reason |
|----------|-----------|--------|
| **Document Classification** | Remove | Focus on meaningful features |
| **Sentiment Analysis** | Keep some* | Negations matter (not good) |
| **Information Retrieval** | Remove | Faster search, better ranking |
| **Machine Translation** | Remove | Reduce noise in training |
| **Text Summarization** | Remove | Important words more visible |

*For sentiment analysis, consider keeping negations like "not", "no", "don't"

---

## Important Considerations

### When NOT to Remove Stop Words
- Sentiment analysis (negations are important)
- Machine translation (word order matters)
- Semantic analysis (meaning preservation needed)
- N-gram analysis (context is important)

### When to Remove Stop Words
- Document classification
- Information retrieval
- Text clustering
- Topic modeling
- General text preprocessing for efficiency

---

## Common Combinations

### For Classification/Clustering
```python
# Remove stop words + Apply stemming
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

stemmer = PorterStemmer()
stops = set(stopwords.words('english'))

processed = [stemmer.stem(word) 
             for word in words 
             if word.lower() not in stops]
```

### For Sentiment Analysis
```python
# Remove stop words BUT keep negations
stops = set(stopwords.words('english'))
negations = {'not', 'no', "don't", "isn't", "aren't"}
stops = stops - negations

processed = [word for word in words 
             if word.lower() not in stops]
```
