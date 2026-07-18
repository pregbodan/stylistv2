"""
nlp_preprocessor.py
--------------------
Implements the NLP Pipeline described in Chapter 3.7 of the project report:
Text Input -> Tokenization -> Lemmatization -> (stop-word removal) -> feature-ready text.

This module is intentionally framework-light (NLTK) so it runs anywhere without
heavy downloads, while still demonstrating a genuine NLP pipeline rather than a
plain string match.
"""

import re
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources if missing
required_resources = {
    "corpora/stopwords": "stopwords",
    "tokenizers/punkt": "punkt",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4",
}

for resource_path, package_name in required_resources.items():
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package_name, quiet=True)

_lemmatizer = WordNetLemmatizer()
_stopwords = set(stopwords.words("english"))

# Keep negation and a few hardware-relevant words that default stopword
# lists would otherwise strip out and that change diagnostic meaning.
_KEEP_WORDS = {"not", "no", "off", "on", "down", "out"}
_stopwords = _stopwords - _KEEP_WORDS


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/extra whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str):
    return word_tokenize(text)


def remove_stopwords(tokens):
    return [t for t in tokens if t not in _stopwords and t not in string.punctuation]


def lemmatize(tokens):
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text: str, for_ml: bool = True) -> str:
    """
    Full pipeline: clean -> tokenize -> [stopword removal] -> lemmatize -> rejoin.

    for_ml=True  -> strips stopwords (better for TF-IDF/classifier input)
    for_ml=False -> keeps stopwords (better for rule/keyword matching, since
                    knowledge-base keyword phrases include words like "not")
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    if for_ml:
        tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def extract_entities(text: str):
    """
    Lightweight rule-based entity extraction for hardware-related nouns.
    Returns a list of recognized hardware component entities mentioned in the text.
    Acts as the 'Entity Recognition' stage of the NLP pipeline (3.7.1).
    """
    component_vocab = [
        "ram", "memory", "cpu", "processor", "gpu", "graphics card", "motherboard",
        "power supply", "psu", "hard drive", "hard disk", "ssd", "monitor", "screen",
        "display", "keyboard", "mouse", "fan", "battery", "usb", "speaker",
        "headphone", "microphone", "webcam", "bluetooth", "printer", "touchpad"
    ]
    text_lower = text.lower()
    found = [comp for comp in component_vocab if comp in text_lower]
    return found


if __name__ == "__main__":
    sample = "My computer won't turn on and there are no lights or fans spinning."
    print("Original :", sample)
    print("ML-ready :", preprocess(sample, for_ml=True))
    print("Rule-ready:", preprocess(sample, for_ml=False))
    print("Entities :", extract_entities(sample))
