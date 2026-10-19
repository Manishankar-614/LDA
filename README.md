# LAB EXERCISE: 4
# Topic Modeling with Latent Dirichlet Allocation (LDA) — Discovering Topics and Their Top Words in a Small News Document Collection

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![Repository](https://img.shields.io/badge/GitHub-Manishankar--614%2FLDA-black?logo=github)](https://github.com/Manishankar-614/LDA)

**Department of AIML | Advance NLP (AML23702)**  
**Student Name:** Manishankar  
**USN:** 1GA23AI025  
**GitHub Repository:** [https://github.com/Manishankar-614/LDA](https://github.com/Manishankar-614/LDA)

---

## 📌 Executive Summary

A modular Streamlit web application built on **scikit-learn** that preprocesses a small collection of documents, fits an LDA topic model, and displays the top five words of every discovered topic together with each document's topic mixture.

The corpus is **100 BBC News articles** (20 from each of five sections: *business, entertainment, politics, sport, tech*). Section labels are hidden from the model during training and used only afterwards to assess whether discovered topics correspond to real semantic themes.

---

## 📁 Repository Structure

```
LDA/
├── app.py                         # Streamlit interactive web application
├── src/
│   ├── __init__.py
│   ├── preprocess.py              # Preprocessor (tokenisation, 339 stop words, WordNet lemmatiser)
│   ├── topic_model.py             # LDATopicModel (restarts loop, best log-likelihood, topic-word distributions)
│   └── metrics.py                 # Evaluation metrics (NPMI coherence, purity, NMI, ARI, stability)
├── data/
│   └── bbc_sample_100.csv         # Stratified sample of 100 BBC News articles
├── eval/
│   ├── __init__.py
│   ├── run_eval.py                # Standalone evaluation reproducing report metrics
│   ├── top_five_words.csv         # Discovered top 5 words per topic with probabilities
│   ├── aggregate_metrics.csv      # Perplexity, coherence, purity, NMI, ARI
│   └── category_vs_topic.csv      # Contingency matrix of labels vs discovered topics
├── report/
│   ├── LAB_REPORT_EXERCISE_4.pdf  # Exact 10-page formal Lab Report
│   ├── LAB_REPORT_EXERCISE_4.md   # Markdown transcript of the 10-page Lab Report
│   └── assets/                    # High-resolution diagrams and UI screenshots
├── generate_report_pdf.py         # ReportLab compilation script for the 10-page report
├── create_report_assets.py        # Figure and diagram generator
├── requirements.txt               # Pinned Python package dependencies
├── .gitignore                     # Git ignore rules for clean repository
└── README.md                      # Project documentation and guide
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Manishankar-614/LDA.git
cd LDA
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Web Application
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

### 4. Run the Standalone Evaluation
```bash
python -m eval.run_eval
```

---

## 🔬 Key Results & Deliverables

### Top 5 Words per Discovered Topic ($K = 5$, Best of 10 Restarts: Seed 6)

| Topic | Top 5 Words (Probability) | Docs Dominant | Interpretation |
| :--- | :--- | :---: | :--- |
| **Topic 1** | firm (0.026), bank (0.022), company (0.019), profit (0.017), machine (0.016) | 22 | Business & finance |
| **Topic 2** | labour (0.018), election (0.016), plan (0.016), government (0.013), group (0.012) | 18 | UK politics & elections |
| **Topic 3** | gadget (0.042), mobile (0.028), phone (0.022), list (0.021), device (0.020) | 11 | Consumer technology / gadgets |
| **Topic 4** | government (0.032), court (0.016), right (0.015), threat (0.013), lord (0.013) | 13 | Law, courts & security |
| **Topic 5** | game (0.027), film (0.027), best (0.019), player (0.017), award (0.015) | 36 | Sport & entertainment (merged) |

### Aggregate Performance Metrics

| Metric | Value | Meaning |
| :--- | :--- | :--- |
| **Vocabulary / Tokens** | 677 / 9,225 | Filtered count matrix after `min_df=5`, `max_df=0.3` |
| **NPMI Coherence** | 0.196 (top-10), 0.217 (top-5) | Mean pairwise normalized PMI in 2,126 reference articles |
| **Perplexity** | 539.8 | Model training perplexity |
| **Purity** | 0.63 | Dominant topic agreement with editorial category |
| **NMI / ARI** | 0.516 / 0.383 | Normalized Mutual Information / Adjusted Rand Index |
| **Confident Docs** | 35 ($p > 0.9$) | Documents strongly assigned to a single dominant topic |

---

## 📑 10-Page Lab Report

The complete, formal 10-page lab report conforming to the department evaluation rubric is located at:
- PDF version: [`report/LAB_REPORT_EXERCISE_4.pdf`](file:///c:/ANLP/Project4/report/LAB_REPORT_EXERCISE_4.pdf)
- Markdown version: [`report/LAB_REPORT_EXERCISE_4.md`](file:///c:/ANLP/Project4/report/LAB_REPORT_EXERCISE_4.md)

---

## 👥 Student & Course Information

- **Course:** Advance NLP (`AML23702`)
- **Department:** Department of AIML
- **Student Name:** Manishankar
- **USN:** 1GA23AI025
- **Lab Exercise No:** 4
