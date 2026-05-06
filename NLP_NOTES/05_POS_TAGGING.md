# NLP Parts of Speech (POS) Tagging Guide

## Overview
**POS Tagging** identifies the grammatical role of each word in a sentence.
- Classifies words as nouns, verbs, adjectives, etc.
- Essential for parsing, semantic analysis, and many NLP tasks
- Helps understand sentence structure and meaning

## Setup Requirements
```python
import nltk
nltk.download('averaged_perceptron_tagger_eng')  # For POS tagger
```

---

## POS Tags Reference

### Complete POS Tag List

| Tag | Meaning | Example |
|-----|---------|---------|
| **CC** | Coordinating conjunction | and, or, but |
| **CD** | Cardinal digit | 1, 2, 3, twenty |
| **DT** | Determiner | the, a, an |
| **EX** | Existential there | there (as in "there is") |
| **FW** | Foreign word | café, naïve |
| **IN** | Preposition/Subordinating conjunction | in, on, at, from, by |
| **JJ** | Adjective | big, beautiful, small |
| **JJR** | Adjective, comparative | bigger, better |
| **JJS** | Adjective, superlative | biggest, best |
| **LS** | List marker | 1), 2), a), b) |
| **MD** | Modal | could, will, might, must |
| **NN** | Noun, singular | cat, book, desk |
| **NNS** | Noun, plural | cats, books, desks |
| **NNP** | Proper noun, singular | John, India, Paris |
| **NNPS** | Proper noun, plural | Americans, Indians |
| **PDT** | Predeterminer | all, both (as in "all the kids") |
| **POS** | Possessive ending | 's (as in "parent's") |
| **PRP** | Personal pronoun | I, he, she, we, they |
| **PRP$** | Possessive pronoun | my, his, her, our, their |
| **RB** | Adverb | very, slowly, quickly |
| **RBR** | Adverb, comparative | faster, better |
| **RBS** | Adverb, superlative | fastest, best |
| **RP** | Particle | up, down (as in "give up") |
| **TO** | 'to' | to (as in "to go") |
| **UH** | Interjection | uh, oh, errrrr |
| **VB** | Verb, base form | take, go, run |
| **VBD** | Verb, past tense | took, went, ran |
| **VBG** | Verb, gerund/present participle | taking, going, running |
| **VBN** | Verb, past participle | taken, gone, run |
| **VBP** | Verb, singular present, non-3rd person | take, go, run |
| **VBZ** | Verb, 3rd person singular present | takes, goes, runs |
| **WDT** | Wh-determiner | which, what |
| **WP** | Wh-pronoun | who, what, whom |
| **WP$** | Possessive wh-pronoun | whose |
| **WRB** | Wh-adverb | where, when, why |

---

## Basic POS Tagging

### Simple Example
```python
import nltk

# Simple sentence
sentence = "Taj Mahal is a beautiful Monument"
words = sentence.split()

# Tag POS
pos_tags = nltk.pos_tag(words)
print(pos_tags)

# Output:
# [('Taj', 'NNP'), ('Mahal', 'NNP'), ('is', 'VBZ'), 
#  ('a', 'DT'), ('beautiful', 'JJ'), ('Monument', 'NN')]
```

---

## POS Tagging on Tokenized Text

### Using Word Tokenizer
```python
from nltk.tokenize import word_tokenize
import nltk

text = "Taj Mahal is a beautiful Monument"

# Proper tokenization (handles punctuation)
tokens = word_tokenize(text)
print(tokens)
# Output: ['Taj', 'Mahal', 'is', 'a', 'beautiful', 'Monument']

# POS tagging
pos_tags = nltk.pos_tag(tokens)
print(pos_tags)
# Output: [('Taj', 'NNP'), ('Mahal', 'NNP'), ('is', 'VBZ'), 
#          ('a', 'DT'), ('beautiful', 'JJ'), ('Monument', 'NN')]
```

---

## Complete Pipeline: Sentence → Tokens → POS Tags

### Full Example with Stopword Removal
```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Setup
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

paragraph = """I have three visions for India. In 3000 years of our history, 
people from all over the world have come and invaded us."""

# Step 1: Tokenize into sentences
sentences = nltk.sent_tokenize(paragraph)

# Step 2: Process each sentence
for sentence in sentences:
    # Tokenize words
    words = word_tokenize(sentence)
    
    # Remove stopwords (optional, for focused analysis)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words 
             if word.lower() not in stop_words]
    
    # Get POS tags
    pos_tags = nltk.pos_tag(words)
    
    print(pos_tags)
```

---

## Extracting Specific POS Categories

### Extract Nouns, Verbs, Adjectives
```python
import nltk
from nltk.tokenize import word_tokenize

text = "The beautiful cat quickly jumped over the fence"
tokens = word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)

# Extract by POS tag
nouns = [word for word, pos in pos_tags if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
verbs = [word for word, pos in pos_tags if pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
adjectives = [word for word, pos in pos_tags if pos in ['JJ', 'JJR', 'JJS']]
adverbs = [word for word, pos in pos_tags if pos in ['RB', 'RBR', 'RBS']]

print(f"Nouns: {nouns}")           # Output: Nouns: ['cat', 'fence']
print(f"Verbs: {verbs}")           # Output: Verbs: ['jumped']
print(f"Adjectives: {adjectives}") # Output: Adjectives: ['beautiful']
print(f"Adverbs: {adverbs}")       # Output: Adverbs: ['quickly']
```

---

## Use Cases

| Application | Usage |
|-------------|-------|
| **Named Entity Recognition** | Identifies proper nouns |
| **Parsing** | Determines sentence structure |
| **Information Extraction** | Finds relationships between entities |
| **Lemmatization** | Needs POS for accurate results |
| **Sentiment Analysis** | Different weights for adj/adv |
| **Text Summarization** | Identifies important words |
| **Machine Translation** | Ensures grammatical accuracy |

---

## Common POS Patterns

### Identifying Key Phrases
```python
# Common noun phrase: DT + JJ + NN
# Example: "the beautiful cat"

# Verb phrase: VB/VBZ + ADV
# Example: "quickly jumped"

# Proper noun: NNP or NNPS
# Example: "India", "New Delhi"

# Noun with modifier: JJ + NN
# Example: "beautiful monument"
```

---

## Tips for Accurate Tagging

1. **Context matters**: "bank" can be NN (financial) or VB (direction)
2. **Case sensitivity**: Uppercase helps identify proper nouns
3. **Punctuation handling**: Use `word_tokenize()` not `split()`
4. **Stopwords**: Consider keeping for POS analysis (affects accuracy)
5. **Word order**: "run" vs "runs" → VB vs VBZ

### Example of Context
```python
text1 = "I will bank on you"        # "bank" is VB
text2 = "The bank is near the river" # "bank" is NN

for text in [text1, text2]:
    tokens = word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    bank_tag = [tag for word, tag in tags if word == 'bank'][0]
    print(f"{text} → bank is {bank_tag}")
```

---

## Performance Considerations
- POS tagging adds computational overhead
- Consider caching results for repeated texts
- For large corpora, batch process when possible
- The `averaged_perceptron_tagger_eng` is fast and accurate
