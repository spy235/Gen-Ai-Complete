# NLP Named Entity Recognition (NER) Guide

## Overview
**Named Entity Recognition (NER)** identifies and classifies named entities in text.
- Extracts important entities like persons, locations, organizations, dates, etc.
- Essential for information extraction and knowledge base construction
- Enables understanding of "who", "where", "when" in text

## Entity Types

### Common Entity Categories

| Entity Type | Tag | Examples |
|------------|-----|----------|
| **Person** | PERSON | Krish C Naik, Gustave Eiffel, Albert Einstein |
| **Location/Place** | GPE, LOCATION | India, Paris, New York, France |
| **Organization** | ORG | iNeuron Private Limited, Google, NASA |
| **Date** | DATE | September, 24-09-1989, 1887-1889 |
| **Time** | TIME | 4:30 PM, 3 o'clock, noon |
| **Money** | MONEY | 1 million dollars, $50, ₹1000 |
| **Percent** | PERCENT | 20%, twenty percent |
| **Facility** | FACILITY | Eiffel Tower, Empire State Building |
| **Product** | PRODUCT | iPhone, Windows |

---

## Setup Requirements
```python
import nltk

# Download required resources
nltk.download('averaged_perceptron_tagger_eng')  # For POS tagging
nltk.download('maxent_ne_chunker_tab')          # Modern NER model
nltk.download('words')                           # Word corpus
```

---

## Named Entity Chunking (ne_chunk)

### Purpose
Groups named entities based on POS tags using chunking rules.

### Basic Example
```python
import nltk
from nltk.tokenize import word_tokenize

sentence = "The Eiffel Tower was built from 1887 to 1889 by Gustave Eiffel, " \
           "whose company specialized in building metal frameworks."

# Step 1: Tokenize
tokens = word_tokenize(sentence)

# Step 2: POS tagging
pos_tags = nltk.pos_tag(tokens)

# Step 3: Named Entity Recognition
entities = nltk.ne_chunk(pos_tags)

print(entities)
# Output: Shows tree structure with recognized entities
```

---

## Complete NER Pipeline

### Full Example
```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Setup
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')

paragraph = """The Eiffel Tower was built from 1887 to 1889 by Gustave Eiffel, 
whose company specialized in building metal frameworks and structures."""

# Step 1: Tokenize into sentences
sentences = sent_tokenize(paragraph)

# Step 2: Process each sentence
for sentence in sentences:
    # Tokenize words
    words = word_tokenize(sentence)
    
    # Optional: Remove stopwords for focused analysis
    stop_words = set(stopwords.words('english'))
    words_filtered = [word for word in words 
                      if word.lower() not in stop_words]
    
    # POS tagging
    pos_tags = nltk.pos_tag(words_filtered)
    
    # Named Entity Recognition
    entities = nltk.ne_chunk(pos_tags)
    
    print(entities)
    # Can also draw tree: entities.draw()
```

---

## Extract and Display Entities

### Extract Named Entities
```python
import nltk
from nltk.tokenize import word_tokenize

sentence = "Apple was founded by Steve Jobs in California in 1976"

tokens = word_tokenize(sentence)
pos_tags = nltk.pos_tag(tokens)
entities = nltk.ne_chunk(pos_tags)

# Extract individual entities
named_entities = []
for subtree in entities:
    if hasattr(subtree, 'label'):  # Check if it's a named entity
        entity_name = ' '.join([word for word, tag in subtree.leaves()])
        entity_type = subtree.label()
        named_entities.append((entity_name, entity_type))

print(named_entities)
# Output: [('Apple', 'ORGANIZATION'), ('Steve Jobs', 'PERSON'), 
#          ('California', 'GPE'), ('1976', 'DATE')]
```

---

## Visualize Entity Trees

### Draw Tree Diagram
```python
import nltk
from nltk.tokenize import word_tokenize

sentence = "Gustave Eiffel built the Eiffel Tower in Paris during 1887-1889"

tokens = word_tokenize(sentence)
pos_tags = nltk.pos_tag(tokens)
entities = nltk.ne_chunk(pos_tags)

# Draw the tree (creates visual diagram)
entities.draw()

# Or get string representation
print(entities)
```

### Output Example
```
(S
  (PERSON Gustave/NNP Eiffel/NNP)
  built/VBD
  (FACILITY the/DT Eiffel/NNP Tower/NNP)
  in/IN
  (GPE Paris/NNP)
  during/IN
  1887-1889/CD)
```

---

## NER with Binary Mode

### Binary vs Multiclass
```python
import nltk

# Multiclass (default) - distinguishes entity types
nltk.ne_chunk(pos_tags, binary=False)

# Binary - just marks NAMED_ENTITY or not
nltk.ne_chunk(pos_tags, binary=True)
```

### Binary Example
```python
# Output with binary=True:
(S
  (NE Gustave/NNP Eiffel/NNP)
  built/VBD
  (NE the/DT Eiffel/NNP Tower/NNP)
  in/IN
  (NE Paris/NNP)
  during/IN
  1887-1889/CD)
```

---

## Practical Use Cases

### 1. Information Extraction
```python
# Extract key information from news articles
def extract_entities(text):
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    entities = nltk.ne_chunk(pos_tags)
    
    extracted = {'PERSON': [], 'GPE': [], 'ORGANIZATION': [], 'DATE': []}
    
    for subtree in entities:
        if hasattr(subtree, 'label'):
            entity = ' '.join([word for word, tag in subtree.leaves()])
            entity_type = subtree.label()
            if entity_type in extracted:
                extracted[entity_type].append(entity)
    
    return extracted

# Usage
result = extract_entities("Elon Musk founded Tesla in USA in 2003")
print(result)
# Output: {'PERSON': ['Elon Musk'], 'GPE': ['USA'], 
#          'ORGANIZATION': ['Tesla'], 'DATE': ['2003']}
```

### 2. Question Answering
```python
# Identify question targets for Q&A systems
def identify_question_target(question):
    tokens = word_tokenize(question)
    pos_tags = nltk.pos_tag(tokens)
    entities = nltk.ne_chunk(pos_tags)
    
    # Look for named entities - these are likely answers
    for subtree in entities:
        if hasattr(subtree, 'label'):
            print(f"Look for: {subtree.label()}")

identify_question_target("Where is the Eiffel Tower located?")
# Output: Look for: LOCATION (or GPE)
```

### 3. Knowledge Base Construction
```python
# Build relationships from text
def extract_relationships(text):
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    entities = nltk.ne_chunk(pos_tags)
    
    # Group entities and their context
    relationships = []
    for subtree in entities:
        if hasattr(subtree, 'label'):
            entity = ' '.join([word for word, tag in subtree.leaves()])
            entity_type = subtree.label()
            relationships.append({'entity': entity, 'type': entity_type})
    
    return relationships
```

---

## Limitations

| Limitation | Workaround |
|-----------|-----------|
| Domain-specific entities not recognized | Train custom model |
| Abbreviations not handled well | Create mapping dictionary |
| Ambiguous entities misclassified | Add context rules |
| Case sensitivity issues | Normalize case appropriately |

---

## Advanced: Custom Entity Recognition

### For Domain-Specific Entities
```python
# If default NER doesn't recognize your entities
# Create pattern-based rules

import re

def extract_custom_entities(text):
    # Pattern for email
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    
    # Pattern for phone
    phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
    
    # Pattern for URL
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    
    return {'emails': emails, 'phones': phones, 'urls': urls}

# Usage
text = "Contact John at john@example.com or 555-123-4567"
print(extract_custom_entities(text))
```

---

## Performance Tips

1. **Batch processing**: Process multiple sentences together
2. **Cache results**: Store NER outputs for repeated texts
3. **Reduce vocabulary**: Remove unnecessary words first
4. **Combine approaches**: Use regex for simple patterns, NER for complex ones
5. **Evaluate on domain**: Test NER accuracy on your specific domain

---

## Common Pitfalls

❌ **Don't**: Assume all entity types are recognized
✅ **Do**: Test on your specific domain

❌ **Don't**: Ignore case sensitivity
✅ **Do**: Handle proper nouns carefully

❌ **Don't**: Skip POS tagging
✅ **Do**: Always POS tag before NER

❌ **Don't**: Miss multi-word entities
✅ **Do**: Join entity phrases properly
