# NLP Concepts Reference Guide

A comprehensive collection of structured notes on Natural Language Processing (NLP) fundamentals with working code examples.

## 📚 Contents

### Core NLP Concepts

1. **[01_TOKENIZATION.md](01_TOKENIZATION.md)**
   - Overview of tokenization
   - Sentence tokenization
   - Word tokenization
   - Wordpunct tokenization
   - TreeBank word tokenization
   - Comparison of methods

2. **[02_STEMMING.md](02_STEMMING.md)**
   - Stemming overview
   - Porter Stemmer
   - Regexp Stemmer
   - Snowball Stemmer
   - Comparison and use cases

3. **[03_LEMMATIZATION.md](03_LEMMATIZATION.md)**
   - Lemmatization overview
   - WordNet Lemmatizer
   - POS tags for accuracy
   - Complete pipeline example
   - Lemmatization vs Stemming

4. **[04_STOPWORDS.md](04_STOPWORDS.md)**
   - Stop words overview
   - English stop words
   - Custom stop words
   - Multi-language support
   - Integration in pipelines

5. **[05_POS_TAGGING.md](05_POS_TAGGING.md)**
   - Parts of Speech tagging
   - Complete POS tag reference
   - Basic tagging examples
   - Entity extraction by POS
   - Use cases and patterns

6. **[06_NAMED_ENTITY_RECOGNITION.md](06_NAMED_ENTITY_RECOGNITION.md)**
   - Named Entity Recognition (NER)
   - Entity types and examples
   - Complete NER pipeline
   - Entity extraction and visualization
   - Practical use cases

---

## 🚀 Quick Start

### Setup Dependencies
```bash
pip install nltk
```

### Download Required NLTK Data
```python
import nltk

# Essential downloads
nltk.download('punkt_tab')                    # Tokenization
nltk.download('averaged_perceptron_tagger_eng')  # POS Tagging
nltk.download('wordnet')                      # Lemmatization
nltk.download('stopwords')                    # Stop words
nltk.download('maxent_ne_chunker_tab')       # NER (newer version)
nltk.download('words')                        # Word corpus
```

---

## 📊 NLP Processing Pipeline

```
Raw Text
    ↓
[TOKENIZATION] → Break into words/sentences
    ↓
[STOPWORD REMOVAL] → Remove common words (optional)
    ↓
[STEMMING/LEMMATIZATION] → Normalize word forms
    ↓
[POS TAGGING] → Identify word types
    ↓
[NAMED ENTITY RECOGNITION] → Extract entities
    ↓
Processed Features → Ready for ML/Analysis
```

---

## 💡 Common Use Cases

### Document Classification
```
Tokenization → Stopword Removal → Stemming → Feature Extraction
```

### Sentiment Analysis
```
Tokenization → [Keep negations] → Lemmatization → Classification
```

### Question Answering
```
Tokenization → POS Tagging → NER → Extract Answer Entities
```

### Information Extraction
```
Tokenization → POS Tagging → NER → Relationship Extraction
```

### Text Summarization
```
Sentence Tokenization → Keyword Extraction → Ranking → Summary
```

---

## 🔧 Decision Tree

**Choose between Stemming and Lemmatization:**

```
Do you need real dictionary words?
├─ YES → Use Lemmatization
└─ NO  → Use Stemming

Need semantic accuracy?
├─ YES → Use Lemmatization (slower but accurate)
└─ NO  → Use Stemming (faster)

Domain-specific processing?
├─ YES → Consider custom stop words + appropriate stemmer
└─ NO  → Use standard English stopwords
```

---

## 📝 Example: Complete Text Processing

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Setup
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

text = """The quick brown foxes jumped over the lazy dogs."""

# Step 1: Tokenize
sentences = sent_tokenize(text)
words = word_tokenize(sentences[0])

# Step 2: Remove stopwords
filtered = [w for w in words if w.lower() not in stop_words]

# Step 3: Stem
stemmed = [stemmer.stem(w.lower()) for w in filtered]

print(stemmed)
# Output: ['quick', 'brown', 'fox', 'jump', 'lazi', 'dog']
```

---

## 🎯 File Organization

```
NLP_NOTES/
├── README.md (this file)
├── 01_TOKENIZATION.md
├── 02_STEMMING.md
├── 03_LEMMATIZATION.md
├── 04_STOPWORDS.md
├── 05_POS_TAGGING.md
└── 06_NAMED_ENTITY_RECOGNITION.md
```

---

## 📖 How to Use This Guide

1. **Quick Reference**: Use the specific markdown files for quick lookups
2. **Code Copy-Paste**: All examples are production-ready
3. **Learning Path**: Read in order (01→06) for complete understanding
4. **Integration**: Copy relevant code into your projects
5. **Customization**: Adapt examples for your specific domain

---

## 🔍 Key Concepts at a Glance

| Concept | Input | Output | Purpose |
|---------|-------|--------|---------|
| **Tokenization** | Text string | List of tokens | Break text into units |
| **Stemming** | Word | Root form | Normalize word variations |
| **Lemmatization** | Word + POS | Dictionary word | Accurate word normalization |
| **Stop words** | Word list | Filtered words | Remove noise |
| **POS Tagging** | Tokens | Word + POS tag | Identify word types |
| **NER** | POS-tagged tokens | Named entities | Extract important entities |

---

## ⚠️ Common Mistakes to Avoid

❌ **Not downloading required NLTK data**
✅ Always run nltk.download() for needed resources

❌ **Skipping POS tagging before lemmatization**
✅ POS tags are essential for accurate lemmatization

❌ **Using stemming for semantic tasks**
✅ Use lemmatization when meaning matters

❌ **Removing all stop words indiscriminately**
✅ Keep negations for sentiment analysis

❌ **Ignoring case sensitivity**
✅ Normalize case appropriately for your task

---

## 📚 Additional Resources

- [NLTK Official Documentation](https://www.nltk.org/)
- [NLTK Book Online](https://www.nltk.org/book/)
- [WordNet Documentation](https://wordnet.princeton.edu/)
- [Stanford NLP Tools](https://nlp.stanford.edu/software/)

---

## 🤝 Tips for Productivity

1. **Keep this guide bookmarked** for quick reference
2. **Create a utilities file** with your common functions
3. **Test on your domain** - NLP models may behave differently on specific text
4. **Version your stop words** - domain-specific lists may change
5. **Cache results** - avoid reprocessing identical text

---

## 📋 Checklist for NLP Project

- [ ] Installed nltk
- [ ] Downloaded required NLTK data
- [ ] Analyzed text domain and characteristics
- [ ] Decided on tokenization method
- [ ] Determined stemming vs lemmatization
- [ ] Identified custom stop words if needed
- [ ] Tested POS tagger on sample data
- [ ] Validated NER on domain-specific entities
- [ ] Created preprocessing pipeline
- [ ] Tested end-to-end pipeline

---

**Last Updated**: May 2026
**Source**: NLP_Basics1.ipynb & NLP_Basics2.ipynb
