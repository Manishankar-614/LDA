"""
Evaluation Metrics for Topic Modeling.
Calculates:
1. NPMI (Normalized Pointwise Mutual Information) coherence across top topic words.
2. Classification agreement metrics: Purity, NMI (Normalized Mutual Information), and ARI (Adjusted Rand Index).
3. Run-to-run topic stability (mean pairwise Jaccard similarity).
"""

from typing import List, Tuple, Dict, Set
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment


def compute_npmi_coherence(
    top_words: List[List[Tuple[str, float]]],
    reference_docs: List[List[str]],
    top_n: int = 5,
    eps: float = 1e-12
) -> Tuple[float, List[float]]:
    """
    Computes Normalized Pointwise Mutual Information (NPMI) coherence for top words.
    NPMI(a, b) = log(P(a, b) / (P(a) * P(b))) / (-log(P(a, b)))
    Ranges from -1 to 1; positive values denote genuine semantic co-occurrence.
    """
    doc_sets = [set(d) for d in reference_docs]
    n_docs = max(len(doc_sets), 1)

    topic_npmi = []
    for word_list in top_words:
        words = [w for w, _ in word_list[:top_n]]
        pair_npmis = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w1, w2 = words[i], words[j]
                d1 = sum(1 for s in doc_sets if w1 in s)
                d2 = sum(1 for s in doc_sets if w2 in s)
                d12 = sum(1 for s in doc_sets if w1 in s and w2 in s)

                if d12 == 0:
                    pair_npmis.append(-1.0)
                else:
                    p1 = d1 / n_docs
                    p2 = d2 / n_docs
                    p12 = d12 / n_docs
                    pmi = np.log(p12 / (p1 * p2 + eps))
                    npmi = pmi / (-np.log(p12) + eps)
                    pair_npmis.append(npmi)

        topic_npmi.append(float(np.mean(pair_npmis)) if pair_npmis else 0.0)

    mean_npmi = float(np.mean(topic_npmi)) if topic_npmi else 0.0
    return mean_npmi, topic_npmi


def compute_purity(y_true: List[str], dominant_topics: np.ndarray) -> float:
    """
    Computes clustering purity: share of documents whose dominant topic's majority category is their own.
    Purity = sum_k max_c |w_k cap c| / N
    """
    classes = list(set(y_true))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true_idx = np.array([class_to_idx[c] for c in y_true])

    k_clusters = len(np.unique(dominant_topics))
    c_classes = len(classes)
    contingency = np.zeros((k_clusters, c_classes), dtype=int)

    for cluster_id, class_id in zip(dominant_topics, y_true_idx):
        contingency[cluster_id, class_id] += 1

    return float(np.sum(np.max(contingency, axis=1)) / len(y_true))


def compute_clustering_agreement(y_true: List[str], dominant_topics: np.ndarray) -> Dict[str, float]:
    """Computes Purity, NMI, and ARI against reference category labels."""
    purity = compute_purity(y_true, dominant_topics)
    nmi = float(normalized_mutual_info_score(y_true, dominant_topics))
    ari = float(adjusted_rand_score(y_true, dominant_topics))
    return {
        "purity": round(purity, 3),
        "nmi": round(nmi, 3),
        "ari": round(ari, 3)
    }


def compute_top_word_stability(restarts_top_words: List[List[Set[str]]]) -> float:
    """
    Measures the mean pairwise Jaccard similarity across restarts.
    restarts_top_words is a list of [set_of_words_per_topic] for each restart run.
    """
    n_runs = len(restarts_top_words)
    if n_runs < 2:
        return 1.0

    jaccards = []
    for r1 in range(n_runs):
        for r2 in range(r1 + 1, n_runs):
            # Compute cost matrix between topics of run 1 and run 2
            topics1 = restarts_top_words[r1]
            topics2 = restarts_top_words[r2]
            k = len(topics1)
            sim_matrix = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    intersection = len(topics1[i].intersection(topics2[j]))
                    union = len(topics1[i].union(topics2[j]))
                    sim_matrix[i, j] = intersection / max(union, 1)
            
            # Maximum weight bipartite matching (Hungarian algorithm)
            row_ind, col_ind = linear_sum_assignment(-sim_matrix)
            matched_jaccard = sim_matrix[row_ind, col_ind].mean()
            jaccards.append(matched_jaccard)

    return float(np.mean(jaccards))
