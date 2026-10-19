"""
Topic Model wrapper using scikit-learn's LatentDirichletAllocation.
Implements bag-of-words vectorisation, multiple random restarts with log-likelihood selection,
topic-word distribution extraction, and document-topic mixture inference.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from src.preprocess import Preprocessor


class LDATopicModel:
    """Fits Latent Dirichlet Allocation with multi-restart selection on text documents."""

    def __init__(
        self,
        n_components: int = 5,
        min_df: int = 5,
        max_df: float = 0.3,
        n_restarts: int = 10,
        random_state: int = 0,
        max_iter: int = 100,
        preprocessor: Optional[Preprocessor] = None
    ):
        self.n_components = n_components
        self.min_df = min_df
        self.max_df = max_df
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.max_iter = max_iter
        self.preprocessor = preprocessor or Preprocessor()

        self.vectorizer = CountVectorizer(
            analyzer=self.preprocessor,
            min_df=min_df,
            max_df=max_df
        )

        self.lda: Optional[LatentDirichletAllocation] = None
        self.X = None
        self.vocab: List[str] = []
        self.theta: Optional[np.ndarray] = None
        self.topic_word: Optional[np.ndarray] = None
        self.best_seed: int = random_state
        self.best_ll: float = 0.0
        self.perplexity: float = 0.0

    def fit(self, docs: List[str]):
        """Fits CountVectorizer and runs N random restarts of LDA, selecting the highest log-likelihood."""
        self.docs = docs
        self.X = self.vectorizer.fit_transform(self.docs)
        self.vocab = list(self.vectorizer.get_feature_names_out())

        best = None
        for i in range(self.n_restarts):
            seed = self.random_state + i
            lda = LatentDirichletAllocation(
                n_components=self.n_components,
                learning_method="batch",
                max_iter=self.max_iter,
                random_state=seed
            )
            theta = lda.fit_transform(self.X)
            ll = lda.score(self.X)
            if best is None or ll > best[0]:
                best = (ll, lda, theta, seed)

        self.best_ll, self.lda, self.theta, self.best_seed = best
        self.perplexity = float(self.lda.perplexity(self.X))

        # Topic-word distribution beta: rows sum to 1
        comp = self.lda.components_
        self.topic_word = comp / comp.sum(axis=1, keepdims=True)
        return self

    def get_top_words(self, n: int = 5) -> List[List[Tuple[str, float]]]:
        """Returns the n most probable words and their probabilities for each topic."""
        out = []
        for row in self.topic_word:
            idx = np.argsort(row)[::-1][:n]
            out.append([(self.vocab[i], float(row[i])) for i in idx])
        return out

    def get_dominant_topics(self) -> np.ndarray:
        """Returns the index of the dominant topic (argmax) for each document."""
        return np.argmax(self.theta, axis=1)

    def get_topic_prevalence(self) -> np.ndarray:
        """Computes the average probability share of each topic across all documents."""
        return self.theta.mean(axis=0)
