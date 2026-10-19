"""
Text Preprocessor for Topic Modeling with Latent Dirichlet Allocation (LDA).
Implements tokenisation, stop-word removal (scikit-learn 318 + news domain stop words),
and morphological lemmatisation using NLTK WordNet.
"""

import re
from typing import List, Set, Optional
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
from nltk.stem import WordNetLemmatizer

# Regex matching alphabetic tokens of 3 or more letters
TOKEN_RE = re.compile(r"[a-z]{3,}")

# Domain-specific news stop words (said, told, mr, year, ...)
# 21 words not present in scikit-learn's 318 stop words, giving 339 words in all.
NEWS_STOP_WORDS = {
    "said", "told", "mr", "year", "people", "new", "time", "world",
    "make", "just", "like", "day", "say", "way", "come", "take",
    "good", "know", "set", "going", "think"
}


class Preprocessor:
    """Tokenises, strips stop words, and lemmatises text documents."""

    def __init__(self, extra_stops: Optional[Set[str]] = None, lemmatize: bool = True):
        self.stop: Set[str] = set(ENGLISH_STOP_WORDS).union(NEWS_STOP_WORDS)
        if extra_stops:
            self.stop.update(extra_stops)
        self.lemmatize = lemmatize
        
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.lemmatizer.lemmatize("jobs")
        except Exception:
            nltk.download("wordnet", quiet=True)
            self.lemmatizer = WordNetLemmatizer()

    def _lemma(self, word: str) -> str:
        """Applies WordNet noun lemmatisation (e.g. jobs -> job, profits -> profit)."""
        if not self.lemmatize:
            return word
        try:
            return self.lemmatizer.lemmatize(word, pos="n")
        except Exception:
            return word

    def __call__(self, text: str) -> List[str]:
        """Preprocesses a single document string into a list of cleaned tokens."""
        out = []
        for tok in TOKEN_RE.findall(text.lower()):
            if tok in self.stop:
                continue
            tok = self._lemma(tok)
            if tok not in self.stop and len(tok) >= 3:
                out.append(tok)
        return out
