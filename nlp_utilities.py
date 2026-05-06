"""
NLP Utilities - Reusable functions for common NLP tasks

This module provides utility functions for tokenization, stemming, lemmatization,
stop word removal, POS tagging, and named entity recognition.

Import and use anywhere in your project:
    from nlp_utilities import *
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize, TreebankWordTokenizer
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer, RegexpStemmer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk


# ============================================================================
# TOKENIZATION FUNCTIONS
# ============================================================================

def sentence_tokenize(text):
    """
    Break text into sentences.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of sentences
        
    Example:
        >>> text = "Hello world. How are you?"
        >>> sentence_tokenize(text)
        ['Hello world.', 'How are you?']
    """
    return sent_tokenize(text)


def word_tokenize_text(text):
    """
    Break text into words.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of words
        
    Example:
        >>> text = "The quick brown fox"
        >>> word_tokenize_text(text)
        ['The', 'quick', 'brown', 'fox']
    """
    return word_tokenize(text)


def wordpunct_tokenize_text(text):
    """
    Tokenize text separating punctuation.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of tokens including punctuation
    """
    return wordpunct_tokenize(text)


def treebank_tokenize(text):
    """
    Advanced tokenization using TreebankWordTokenizer.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of tokens
    """
    tokenizer = TreebankWordTokenizer()
    return tokenizer.tokenize(text)


# ============================================================================
# STEMMING FUNCTIONS
# ============================================================================

def porter_stem(text):
    """
    Apply Porter Stemming to text.
    
    Args:
        text (str or list): Input text or list of words
        
    Returns:
        str or list: Stemmed text/words
        
    Example:
        >>> porter_stem("running")
        'run'
        >>> porter_stem(["eating", "eating", "eaten"])
        ['eat', 'eat', 'eaten']
    """
    stemmer = PorterStemmer()
    
    if isinstance(text, str):
        # Single word
        if ' ' in text:
            # Multiple words in string
            return ' '.join([stemmer.stem(word.lower()) for word in word_tokenize(text)])
        else:
            return stemmer.stem(text.lower())
    elif isinstance(text, list):
        return [stemmer.stem(word.lower()) for word in text]
    else:
        raise TypeError("Input must be string or list")


def snowball_stem(text, language='english'):
    """
    Apply Snowball Stemming to text.
    
    Args:
        text (str or list): Input text or list of words
        language (str): Language for stemming (default: 'english')
        
    Returns:
        str or list: Stemmed text/words
    """
    stemmer = SnowballStemmer(language)
    
    if isinstance(text, str):
        if ' ' in text:
            return ' '.join([stemmer.stem(word.lower()) for word in word_tokenize(text)])
        else:
            return stemmer.stem(text.lower())
    elif isinstance(text, list):
        return [stemmer.stem(word.lower()) for word in text]
    else:
        raise TypeError("Input must be string or list")


def regex_stem(text, pattern='ing|s$|able$|e$', min_length=4):
    """
    Apply Regex-based Stemming to text.
    
    Args:
        text (str or list): Input text or list of words
        pattern (str): Regex pattern for suffixes to remove
        min_length (int): Minimum word length
        
    Returns:
        str or list: Stemmed text/words
        
    Example:
        >>> regex_stem("eating", pattern='ing|s$')
        'eat'
    """
    stemmer = RegexpStemmer(pattern, min=min_length)
    
    if isinstance(text, str):
        if ' ' in text:
            return ' '.join([stemmer.stem(word.lower()) for word in word_tokenize(text)])
        else:
            return stemmer.stem(text.lower())
    elif isinstance(text, list):
        return [stemmer.stem(word.lower()) for word in text]
    else:
        raise TypeError("Input must be string or list")


# ============================================================================
# LEMMATIZATION FUNCTIONS
# ============================================================================

def lemmatize_word(word, pos='n'):
    """
    Lemmatize a single word.
    
    Args:
        word (str): Input word
        pos (str): Part of speech ('n'=noun, 'v'=verb, 'a'=adjective, 'r'=adverb)
        
    Returns:
        str: Lemmatized word
        
    Example:
        >>> lemmatize_word("running", pos='v')
        'run'
        >>> lemmatize_word("better", pos='a')
        'good'
    """
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word.lower(), pos=pos)


def lemmatize_text(text, pos='v'):
    """
    Lemmatize text with specified POS.
    
    Args:
        text (str or list): Input text or list of words
        pos (str): Part of speech for all words
        
    Returns:
        str or list: Lemmatized text/words
    """
    lemmatizer = WordNetLemmatizer()
    
    if isinstance(text, str):
        if ' ' in text:
            return ' '.join([lemmatizer.lemmatize(word.lower(), pos=pos) 
                           for word in word_tokenize(text)])
        else:
            return lemmatizer.lemmatize(text.lower(), pos=pos)
    elif isinstance(text, list):
        return [lemmatizer.lemmatize(word.lower(), pos=pos) for word in text]
    else:
        raise TypeError("Input must be string or list")


# ============================================================================
# STOP WORDS FUNCTIONS
# ============================================================================

def get_stop_words(language='english'):
    """
    Get stop words for a language.
    
    Args:
        language (str): Language code (default: 'english')
        
    Returns:
        set: Set of stop words
        
    Example:
        >>> stops = get_stop_words()
        >>> 'the' in stops
        True
    """
    return set(stopwords.words(language))


def remove_stopwords(text, language='english', custom_stopwords=None):
    """
    Remove stop words from text.
    
    Args:
        text (str or list): Input text or list of words
        language (str): Language for stop words
        custom_stopwords (set): Additional stop words to remove
        
    Returns:
        list: Filtered words
        
    Example:
        >>> text = "The quick brown fox"
        >>> remove_stopwords(text)
        ['quick', 'brown', 'fox']
    """
    stop_words = get_stop_words(language)
    
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    
    if isinstance(text, str):
        words = word_tokenize(text)
    else:
        words = text
    
    return [word for word in words if word.lower() not in stop_words]


# ============================================================================
# POS TAGGING FUNCTIONS
# ============================================================================

def get_pos_tags(text):
    """
    Get POS tags for text.
    
    Args:
        text (str or list): Input text or list of words
        
    Returns:
        list: List of (word, tag) tuples
        
    Example:
        >>> get_pos_tags("Taj Mahal is beautiful")
        [('Taj', 'NNP'), ('Mahal', 'NNP'), ('is', 'VBZ'), ('beautiful', 'JJ')]
    """
    if isinstance(text, str):
        tokens = word_tokenize(text)
    else:
        tokens = text
    
    return pos_tag(tokens)


def extract_by_pos(text, pos_tags_list):
    """
    Extract words by specific POS tags.
    
    Args:
        text (str or list): Input text or list of words
        pos_tags_list (list): List of POS tags to extract
        
    Returns:
        list: Words matching the specified POS tags
        
    Example:
        >>> text = "The beautiful cat runs"
        >>> extract_by_pos(text, ['NN', 'NNS', 'NNP', 'NNPS'])
        ['cat']
    """
    pos_tags = get_pos_tags(text)
    return [word for word, tag in pos_tags if tag in pos_tags_list]


def extract_nouns(text):
    """Extract all nouns from text."""
    return extract_by_pos(text, ['NN', 'NNS', 'NNP', 'NNPS'])


def extract_verbs(text):
    """Extract all verbs from text."""
    return extract_by_pos(text, ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])


def extract_adjectives(text):
    """Extract all adjectives from text."""
    return extract_by_pos(text, ['JJ', 'JJR', 'JJS'])


def extract_adverbs(text):
    """Extract all adverbs from text."""
    return extract_by_pos(text, ['RB', 'RBR', 'RBS'])


# ============================================================================
# NAMED ENTITY RECOGNITION FUNCTIONS
# ============================================================================

def extract_named_entities(text, binary=False):
    """
    Extract named entities from text.
    
    Args:
        text (str): Input text
        binary (bool): If True, just mark NAMED_ENTITY or not
        
    Returns:
        nltk.Tree: Named entity tree
        
    Example:
        >>> text = "Apple was founded by Steve Jobs in California"
        >>> entities = extract_named_entities(text)
    """
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return ne_chunk(pos_tags, binary=binary)


def extract_entities_list(text, binary=False):
    """
    Extract named entities as a list of (entity, type) tuples.
    
    Args:
        text (str): Input text
        binary (bool): If True, all entities marked as 'NAMED_ENTITY'
        
    Returns:
        list: List of (entity_name, entity_type) tuples
        
    Example:
        >>> text = "Eiffel Tower was built by Gustave Eiffel in Paris"
        >>> extract_entities_list(text)
        [('Eiffel Tower', 'FACILITY'), ('Gustave Eiffel', 'PERSON'), ('Paris', 'GPE')]
    """
    entities_tree = extract_named_entities(text, binary=binary)
    entities = []
    
    for subtree in entities_tree:
        if hasattr(subtree, 'label'):
            entity_name = ' '.join([word for word, tag in subtree.leaves()])
            entity_type = subtree.label()
            entities.append((entity_name, entity_type))
    
    return entities


# ============================================================================
# COMPLETE PIPELINE FUNCTIONS
# ============================================================================

def preprocess_text(text, remove_stops=True, stem_type='porter', language='english'):
    """
    Complete text preprocessing pipeline.
    
    Args:
        text (str): Input text
        remove_stops (bool): Remove stop words
        stem_type (str): 'porter', 'snowball', 'regex', or None
        language (str): Language for processing
        
    Returns:
        str: Processed text
        
    Example:
        >>> text = "The quick brown foxes are running"
        >>> preprocess_text(text)
        'quick brown fox run'
    """
    # Tokenize
    words = word_tokenize(text.lower())
    
    # Remove stop words
    if remove_stops:
        stop_words = get_stop_words(language)
        words = [w for w in words if w not in stop_words]
    
    # Apply stemming
    if stem_type == 'porter':
        words = [PorterStemmer().stem(w) for w in words]
    elif stem_type == 'snowball':
        words = [SnowballStemmer(language).stem(w) for w in words]
    elif stem_type == 'regex':
        words = [RegexpStemmer('ing|s$|able$|e$').stem(w) for w in words]
    
    return ' '.join(words)


def analyze_text(text):
    """
    Complete text analysis with all features.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary with analysis results
    """
    return {
        'original': text,
        'sentences': sentence_tokenize(text),
        'tokens': word_tokenize(text),
        'pos_tags': get_pos_tags(text),
        'nouns': extract_nouns(text),
        'verbs': extract_verbs(text),
        'named_entities': extract_entities_list(text),
        'preprocessed': preprocess_text(text),
    }


# ============================================================================
# UTILITY HELPERS
# ============================================================================

def download_nltk_resources():
    """Download all required NLTK resources."""
    resources = [
        'punkt_tab',
        'averaged_perceptron_tagger_eng',
        'wordnet',
        'stopwords',
        'maxent_ne_chunker_tab',
        'words'
    ]
    
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"✓ Downloaded {resource}")
        except Exception as e:
            print(f"✗ Failed to download {resource}: {e}")


if __name__ == '__main__':
    # Download resources if needed
    download_nltk_resources()
    
    # Example usage
    print("\n" + "="*60)
    print("NLP UTILITIES - EXAMPLES")
    print("="*60)
    
    sample_text = "The quick brown fox jumps over the lazy dog"
    
    print(f"\nOriginal: {sample_text}")
    print(f"Sentences: {sentence_tokenize(sample_text)}")
    print(f"Tokens: {word_tokenize_text(sample_text)}")
    print(f"Porter Stemming: {porter_stem(sample_text)}")
    print(f"Snowball Stemming: {snowball_stem(sample_text)}")
    print(f"Lemmatization: {lemmatize_text(sample_text)}")
    print(f"Stop words removed: {remove_stopwords(sample_text)}")
    print(f"POS Tags: {get_pos_tags(sample_text)}")
    print(f"Nouns: {extract_nouns(sample_text)}")
    print(f"Verbs: {extract_verbs(sample_text)}")
    print(f"Preprocessed: {preprocess_text(sample_text)}")
