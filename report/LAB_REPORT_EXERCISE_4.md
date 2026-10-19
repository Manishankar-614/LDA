# LAB EXERCISE: 4
## Topic Modeling with Latent Dirichlet Allocation (LDA) — Discovering Topics and Their Top Words in a Small News Document Collection

*A modular Streamlit web application built on scikit-learn that preprocesses a small collection of documents, fits an LDA topic model, and displays the top five words of every discovered topic together with each document's topic mixture.*

**Department of AIML | Advance NLP (AML23702)**  
**Student Name:** Manishankar  
**USN:** 1GA23AI025  
**GitHub Repository:** [https://github.com/Manishankar-614/LDA](https://github.com/Manishankar-614/LDA)

---

### Aim / Objective
This project applies Latent Dirichlet Allocation (LDA), an unsupervised probabilistic topic model, to a small collection of text documents and displays the five words most strongly associated with each topic it discovers. LDA treats every document as a mixture of a fixed number of hidden topics and every topic as a probability distribution over words; fitting the model recovers both from word counts alone, with no labels.

The pipeline is organised as small reusable modules in `src/`: a Preprocessor (`preprocess.py`) that tokenises, removes stop words and lemmatises; an LDATopicModel (`topic_model.py`) that builds the document–term matrix, fits scikit-learn's LDA and extracts the top words; and evaluation helpers (`metrics.py`) for topic coherence, agreement with known categories and run-to-run stability. The Streamlit front end (`app.py`) lets the user choose the corpus and the number of topics, and shows the top-word table, per-topic bar charts, the topic mixture of every document and a topic-versus-category heat map.

The corpus is 100 BBC News articles (20 from each of five sections). The section labels are hidden from the model and used only afterwards, to check whether the discovered topics correspond to real themes (see Results and Evaluation / Analysis).

---

### Expected Outcomes Addressed
- **1. Prepare a small collection of text documents.** A stratified random sample of 100 articles (43,141 words) is drawn from the BBC News dataset and stored in `data/bbc_sample_100.csv`; users can also paste their own documents or upload `.txt` / `.csv` files in the app.
- **2. Represent the documents for topic modeling.** Each document is lower-cased, tokenised, stripped of 339 stop words and lemmatised with WordNet, then converted to a bag-of-words count matrix by `CountVectorizer` (words in fewer than 5 or more than 30% of documents are dropped), giving a 100 × 677 matrix with 9,225 tokens.
- **3. Apply Latent Dirichlet Allocation.** scikit-learn's `LatentDirichletAllocation` is fitted with $K = 5$ topics, symmetric Dirichlet priors $\alpha = \eta = 1/K = 0.2$, batch variational Bayes and 100 iterations, from 10 random starts; the start with the highest log-likelihood is kept.
- **4. Display the top five words of each topic.** For each row of the topic–word matrix the five most probable words are shown with their probabilities, in a table with CSV export and as five bar charts; every value is reproduced in the Results section.
- **5. Interpret and evaluate the topics.** Topics are scored with NPMI coherence against 2,126 BBC articles, the number of topics is chosen by sweeping $K = 2–10$, and the dominant topics are compared with the hidden section labels (purity, NMI, ARI).

---

### Tools & Technologies Used
| Category | Tool / Library | Purpose in this pipeline |
| :--- | :--- | :--- |
| **Language / runtime** | Python 3 | Core implementation language for the whole app. |
| **Topic model** | scikit-learn `LatentDirichletAllocation` | Variational-Bayes LDA: learns the document–topic ($\theta$) and topic–word ($\beta$) distributions. |
| **Vectorisation** | scikit-learn `CountVectorizer` | Builds the document–term count matrix and applies the `min_df` / `max_df` vocabulary filter. |
| **Preprocessing** | Python `re`, scikit-learn stop-word list, NLTK WordNet lemmatiser | Tokenisation, stop-word removal and reduction of words to their dictionary form. |
| **Evaluation** | NumPy, SciPy (Hungarian matching), scikit-learn metrics | NPMI coherence, purity / NMI / ARI against labels, top-word stability across restarts. |
| **Web framework** | Streamlit | Corpus and parameter controls, tables, CSV downloads, charts. |
| **Charts** | Plotly (app), Matplotlib (report) | Per-topic word-weight bars, topic prevalence, category × topic heat map, K-selection plot. |
| **Tabular data** | pandas | Corpus loading, topic and document tables, evaluation outputs. |

---

### Dataset / Corpus Description
The documents come from the **BBC News dataset** (Greene and Cunningham, 2006): 2,225 articles from the BBC News website, 2004–2005, each labelled with one of five sections — business, entertainment, politics, sport and tech. The CSV copy used here stores the text lower-cased with most punctuation already removed. After dropping 99 exact duplicate texts, 2,126 articles remain.

To match the brief of a small collection, 20 articles per section were sampled at random (`random_state = 42`), giving 100 documents. The full 2,126 articles are used only as a reference corpus for measuring topic coherence.

| Property | Value |
| :--- | :--- |
| **Documents in the working corpus** | 100 (business 20, entertainment 20, politics 20, sport 20, tech 20) |
| **Raw length** | 43,141 words in total; mean 431, minimum 130, maximum 4,492 words per article |
| **After preprocessing + vocabulary filter** | 677 distinct terms, 9,225 tokens (21% of raw words kept) |
| **Reference corpus for coherence** | 2,126 de-duplicated BBC articles (all five sections) |
| **Labels** | Section names — hidden from LDA, used only for evaluation |

**Representative sample record (`data/bbc_sample_100.csv`, first 240 characters):**
```
doc_id: 409  category: business
text: dutch bank to lay off 2 850 staff abn amro the netherlands largest bank is to cut 2 850 jobs as a result of falling profits. the cuts - amounting to 3% of the bank s workforce - will result in a one-off charge of 790m euros ($1.1bn). ...
```

---

### System Architecture / Pipeline Design

| Stage | What happens |
| :--- | :--- |
| **1. Corpus input** | The sidebar selects the bundled 100-article sample, pasted text (documents separated by blank lines) or uploaded files, plus K, the number of top words and the number of restarts. |
| **2. Tokenisation** | `Preprocessor.__call__` lower-cases the text and keeps alphabetic tokens of three or more letters, discarding numbers and punctuation. |
| **3. Stop-word removal** | Removes scikit-learn's 318 English stop words plus a 38-word news list (21 of them new: said, told, mr, year, …) — 339 words in all. |
| **4. Lemmatisation** | NLTK's WordNet lemmatiser maps plural nouns to their singular form (jobs → job, profits → profit), merging counts for the same concept. |
| **5. Bag of words** | `CountVectorizer` keeps words found in at least 5 documents and at most 30% of them, producing a 100 × 677 sparse count matrix. |
| **6. LDA fitting** | `LatentDirichletAllocation` ($K = 5, \alpha = \eta = 0.2$, batch variational Bayes, 100 iterations) is run from 10 random seeds; the run with the highest log-likelihood (variational bound) is kept. |
| **7. Topic extraction** | Each row of `components_` is normalised to a word distribution $\beta$; its five largest entries are the topic's top words. `fit_transform` returns $\theta$, the topic mixture of each document. |
| **8. Streamlit UI** | Top-word table and bar charts, document–topic table with dominant topic, topic prevalence and (if labels exist) a category × topic heat map; CSV downloads for both tables. |

---

### Key Results (Deliverable-Specific Outputs)

#### Top five words per topic ($K = 5$, best of 10 restarts: seed 6)
| Topic | Top 5 words (probability) | Docs | NPMI@5 | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Topic 1** | firm (0.026), bank (0.022), company (0.019), profit (0.017), machine (0.016) | 22 | 0.134 | Business & finance |
| **Topic 2** | labour (0.018), election (0.016), plan (0.016), government (0.013), group (0.012) | 18 | 0.249 | UK politics & elections |
| **Topic 3** | gadget (0.042), mobile (0.028), phone (0.022), list (0.021), device (0.020) | 11 | 0.411 | Consumer technology / gadgets |
| **Topic 4** | government (0.032), court (0.016), right (0.015), threat (0.013), lord (0.013) | 13 | 0.155 | Law, courts & security |
| **Topic 5** | game (0.027), film (0.027), best (0.019), player (0.017), award (0.015) | 36 | 0.138 | Sport & entertainment (merged) |

#### Aggregate Metrics
| Metric | Value | Definition |
| :--- | :--- | :--- |
| **Vocabulary / tokens** | 677 / 9,225 | After preprocessing and the min_df / max_df filter. |
| **NPMI coherence** | 0.196 (top-10), 0.217 (top-5) | Mean pairwise NPMI of each topic's top words in 2,126 BBC articles (−1 to 1). |
| **Perplexity** | 539.8 | Training perplexity of the selected model (lower = better fit). |
| **Purity** | 0.63 | Share of documents whose dominant topic's majority category is their own. |
| **NMI / ARI** | 0.516 / 0.383 | Agreement between dominant topics and the hidden labels (1 = identical, 0 = chance). |
| **Confident documents** | 35 with p > 0.9; 15 with p < 0.5 | Dominant-topic probability; mean 0.743. |
| **Top-5 stability** | 0.399 mean Jaccard | Overlap of matched top-5 word sets across the 45 pairs of the 10 restarts. |

#### Dominant topic versus hidden section label (counts of documents)
| Category \ Dominant topic | T1 | T2 | T3 | T4 | T5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **business** | 14 | 3 | 0 | 3 | 0 |
| **entertainment** | 2 | 1 | 0 | 2 | 15 |
| **politics** | 0 | 14 | 0 | 6 | 0 |
| **sport** | 0 | 0 | 0 | 2 | 18 |
| **tech** | 6 | 0 | 11 | 0 | 3 |

---

### ADVANCE NLP (AML23702) — Lab Exercise Evaluation Sheet

| Field | Value |
| :--- | :--- |
| **Student Name** | **Manishankar** |
| **USN** | **1GA23AI025** |
| **Lab Exercise No.** | 4 |
| **Experiment Title** | Topic Modeling with Latent Dirichlet Allocation (LDA) — Discovering Topics and Their Top Words in a Small News Document Collection |

#### Marks Awarded
| Sl. No. | Evaluation Criteria | Max Marks | Marks Obtained |
| :---: | :--- | :---: | :---: |
| 1 | Problem Understanding & Scope | 5 | |
| 2 | Pipeline Design & Methodology | 10 | |
| 3 | Implementation Correctness | 10 | |
| 4 | Use of NLP Tools / Models | 5 | |
| 5 | Output Quality / Task Performance | 10 | |
| 6 | Analysis & Interpretation | 5 | |
| 7 | Documentation & Deliverables | 2.5 | |
| 8 | Presentation & Reproducibility | 2.5 | |
| **Total** | **Comprehensive Lab Assessment** | **50** | |

**Overall Comments:**
________________________________________________________________________________  
________________________________________________________________________________  

**Date of Submission:** ______________ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Signature of Faculty:** ______________
