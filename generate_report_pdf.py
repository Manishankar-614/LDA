"""
Generates the exact 10-page Lab Report PDF for Lab Exercise 4:
Topic Modeling with Latent Dirichlet Allocation (LDA) — Discovering Topics and Their Top Words in a Small News Document Collection.
Configured for:
Student Name: Manishankar
USN: 1GA23AI025
GitHub URL: https://github.com/Manishankar-614/LDA
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

os.makedirs("report", exist_ok=True)
pdf_path = "report/LAB_REPORT_EXERCISE_4.pdf"


class NumberedCanvas(canvas.Canvas):
    """Draws custom header and Page X of 10 footer on every page."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica-Oblique", 8.5)
        self.setFillColor(colors.HexColor("#334155"))

        # Header (Top of page)
        self.drawString(54, 755, "Department of AIML")
        self.drawRightString(558, 755, "Advance NLP (AML23702)")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(54, 748, 558, 748)

        # Footer (Bottom of page)
        self.setFont("Helvetica", 8.5)
        self.drawCentredString(306, 32, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


def build_pdf():
    # Document setup: margins 54pt = 0.75in
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    style_lab_header = ParagraphStyle(
        "LabHeader",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        alignment=1, # Center
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4
    )
    style_main_title = ParagraphStyle(
        "MainTitle",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        alignment=1, # Center
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=6
    )
    style_subtitle = ParagraphStyle(
        "Subtitle",
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12,
        alignment=1, # Center
        textColor=colors.HexColor("#334155"),
        spaceAfter=14
    )
    style_h1 = ParagraphStyle(
        "Heading1_Custom",
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=10,
        spaceAfter=4
    )
    style_h2 = ParagraphStyle(
        "Heading2_Custom",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=6,
        spaceAfter=3
    )
    style_body = ParagraphStyle(
        "Body_Custom",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=5
    )
    style_body_bold = ParagraphStyle(
        "BodyBold_Custom",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=5
    )
    style_code = ParagraphStyle(
        "Code_Custom",
        fontName="Courier",
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#0F172A")
    )
    style_caption = ParagraphStyle(
        "Caption_Custom",
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#475569"),
        spaceAfter=6
    )
    style_table_header = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0F172A")
    )
    style_table_cell = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=7.8,
        leading=10,
        textColor=colors.HexColor("#1E293B")
    )
    style_table_cell_code = ParagraphStyle(
        "TableCellCode",
        fontName="Courier",
        fontSize=7.2,
        leading=9,
        textColor=colors.HexColor("#0F172A")
    )

    story = []

    # =========================================================
    # PAGE 1: Aim, Expected Outcomes
    # =========================================================
    story.append(Paragraph("LAB EXERCISE: 4", style_lab_header))
    story.append(Paragraph("Topic Modeling with Latent Dirichlet Allocation (LDA) — Discovering Topics and Their Top Words in a Small News Document Collection", style_main_title))
    story.append(Paragraph("A modular Streamlit web application built on scikit-learn that preprocesses a small collection of documents, fits an LDA topic model, and displays the top five words of every discovered topic together with each document's topic mixture.", style_subtitle))

    story.append(Paragraph("Aim / Objective", style_h1))
    story.append(Paragraph(
        "This project applies Latent Dirichlet Allocation (LDA), an unsupervised probabilistic topic model, to a small "
        "collection of text documents and displays the five words most strongly associated with each topic it discovers. "
        "LDA treats every document as a mixture of a fixed number of hidden topics and every topic as a probability "
        "distribution over words; fitting the model recovers both from word counts alone, with no labels.",
        style_body
    ))
    story.append(Paragraph(
        "The pipeline is organised as small reusable modules in <code>src/</code>: a Preprocessor (<code>preprocess.py</code>) that tokenises, "
        "removes stop words and lemmatises; an LDATopicModel (<code>topic_model.py</code>) that builds the document–term "
        "matrix, fits scikit-learn's LDA and extracts the top words; and evaluation helpers (<code>metrics.py</code>) for topic "
        "coherence, agreement with known categories and run-to-run stability. The Streamlit front end (<code>app.py</code>) lets the "
        "user choose the corpus and the number of topics, and shows the top-word table, per-topic bar charts, the topic "
        "mixture of every document and a topic-versus-category heat map.",
        style_body
    ))
    story.append(Paragraph(
        "The corpus is 100 BBC News articles (20 from each of five sections). The section labels are hidden from the "
        "model and used only afterwards, to check whether the discovered topics correspond to real themes (see Results and "
        "Evaluation / Analysis).",
        style_body
    ))

    story.append(Paragraph("Expected Outcomes Addressed", style_h1))
    story.append(Paragraph(
        "• <b>1. Prepare a small collection of text documents.</b> A stratified random sample of 100 articles (43,141 words) is "
        "drawn from the BBC News dataset and stored in <code>data/bbc_sample_100.csv</code>; users can also paste their own "
        "documents or upload .txt / .csv files in the app.",
        style_body
    ))
    story.append(Paragraph(
        "• <b>2. Represent the documents for topic modeling.</b> Each document is lower-cased, tokenised, stripped of 339 "
        "stop words and lemmatised with WordNet, then converted to a bag-of-words count matrix by "
        "CountVectorizer (words in fewer than 5 or more than 30% of documents are dropped), giving a 100 × 677 "
        "matrix with 9,225 tokens.",
        style_body
    ))
    story.append(Paragraph(
        "• <b>3. Apply Latent Dirichlet Allocation.</b> scikit-learn's <code>LatentDirichletAllocation</code> is fitted with K = 5 topics, "
        "symmetric Dirichlet priors α = η = 1/K, batch variational Bayes and 100 iterations, from 10 random starts; the "
        "start with the highest log-likelihood is kept.",
        style_body
    ))
    story.append(Paragraph(
        "• <b>4. Display the top five words of each topic.</b> For each row of the topic–word matrix the five most probable "
        "words are shown with their probabilities, in a table with CSV export and as five bar charts; every value is "
        "reproduced in the Results section.",
        style_body
    ))
    story.append(Paragraph(
        "• <b>5. Interpret and evaluate the topics.</b> Topics are scored with NPMI coherence against 2,126 BBC articles, the "
        "number of topics is chosen by sweeping K = 2–10, and the dominant topics are compared with the hidden "
        "section labels (purity, NMI, ARI).",
        style_body
    ))

    story.append(PageBreak())

    # =========================================================
    # PAGE 2: Tools & Technologies, Dataset Description
    # =========================================================
    story.append(Paragraph("Tools & Technologies Used", style_h1))

    tools_data = [
        [Paragraph("Category", style_table_header), Paragraph("Tool / Library", style_table_header), Paragraph("Purpose in this pipeline", style_table_header)],
        [Paragraph("Language / runtime", style_table_cell), Paragraph("Python 3", style_table_cell), Paragraph("Core implementation language for the whole app.", style_table_cell)],
        [Paragraph("Topic model", style_table_cell), Paragraph("scikit-learn<br/>LatentDirichletAllocation", style_table_cell), Paragraph("Variational-Bayes LDA: learns the document–topic (θ) and topic–word (β) distributions.", style_table_cell)],
        [Paragraph("Vectorisation", style_table_cell), Paragraph("scikit-learn CountVectorizer", style_table_cell), Paragraph("Builds the document–term count matrix and applies the min_df / max_df vocabulary filter.", style_table_cell)],
        [Paragraph("Preprocessing", style_table_cell), Paragraph("Python re, scikit-learn stop-word list, NLTK WordNet lemmatiser", style_table_cell), Paragraph("Tokenisation, stop-word removal and reduction of words to their dictionary form.", style_table_cell)],
        [Paragraph("Evaluation", style_table_cell), Paragraph("NumPy, SciPy (Hungarian matching), scikit-learn metrics", style_table_cell), Paragraph("NPMI coherence, purity / NMI / ARI against labels, top-word stability across restarts.", style_table_cell)],
        [Paragraph("Web framework", style_table_cell), Paragraph("Streamlit", style_table_cell), Paragraph("Corpus and parameter controls, tables, CSV downloads, charts.", style_table_cell)],
        [Paragraph("Charts", style_table_cell), Paragraph("Plotly (app), Matplotlib (report)", style_table_cell), Paragraph("Per-topic word-weight bars, topic prevalence, category × topic heat map, K-selection plot.", style_table_cell)],
        [Paragraph("Tabular data", style_table_cell), Paragraph("pandas", style_table_cell), Paragraph("Corpus loading, topic and document tables, evaluation outputs.", style_table_cell)]
    ]
    t_tools = Table(tools_data, colWidths=[100, 160, 244])
    t_tools.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_tools)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Dataset / Corpus Description", style_h1))
    story.append(Paragraph(
        "The documents come from the <b>BBC News dataset</b> (Greene and Cunningham, 2006): 2,225 articles from the BBC "
        "News website, 2004–2005, each labelled with one of five sections — business, entertainment, politics, sport and "
        "tech. The CSV copy used here stores the text lower-cased with most punctuation already removed. After dropping "
        "99 exact duplicate texts, 2,126 articles remain.",
        style_body
    ))
    story.append(Paragraph(
        "To match the brief of a small collection, 20 articles per section were sampled at random (random_state = 42), "
        "giving 100 documents. The full 2,126 articles are used only as a reference corpus for measuring topic coherence.",
        style_body
    ))

    corpus_data = [
        [Paragraph("Property", style_table_header), Paragraph("Value", style_table_header)],
        [Paragraph("Documents in the working corpus", style_table_cell), Paragraph("100 (business 20, entertainment 20, politics 20, sport 20, tech 20)", style_table_cell)],
        [Paragraph("Raw length", style_table_cell), Paragraph("43,141 words in total; mean 431, minimum 130, maximum 4,492 words per article", style_table_cell)],
        [Paragraph("After preprocessing + vocabulary filter", style_table_cell), Paragraph("677 distinct terms, 9,225 tokens (21% of raw words kept)", style_table_cell)],
        [Paragraph("Reference corpus for coherence", style_table_cell), Paragraph("2,126 de-duplicated BBC articles (all five sections)", style_table_cell)],
        [Paragraph("Labels", style_table_cell), Paragraph("Section names — hidden from LDA, used only for evaluation", style_table_cell)],
    ]
    t_corpus = Table(corpus_data, colWidths=[180, 324])
    t_corpus.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_corpus)
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Representative sample record (data/bbc_sample_100.csv, first 240 characters)</b>", style_body))
    sample_box = [
        [Paragraph("<b>doc_id:</b> 409 &nbsp;&nbsp; <b>category:</b> business<br/><b>text:</b> dutch bank to lay off 2 850 staff abn amro the netherlands largest bank is to cut 2 850 jobs as a result of falling profits. the cuts - amounting to 3% of the bank s workforce - will result in a one-off charge of 790m euros ($1.1bn). ...", style_table_cell_code)]
    ]
    t_sample = Table(sample_box, colWidths=[504])
    t_sample.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor("#94A3B8")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_sample)

    story.append(PageBreak())

    # =========================================================
    # PAGE 3: System Architecture & Figure 1
    # =========================================================
    story.append(Paragraph("System Architecture / Pipeline Design", style_h1))

    arch_data = [
        [Paragraph("Stage", style_table_header), Paragraph("What happens", style_table_header)],
        [Paragraph("1. Corpus input", style_table_cell), Paragraph("The sidebar selects the bundled 100-article sample, pasted text (documents separated by blank lines) or uploaded files, plus K, the number of top words and the number of restarts.", style_table_cell)],
        [Paragraph("2. Tokenisation", style_table_cell), Paragraph("Preprocessor.__call__ lower-cases the text and keeps alphabetic tokens of three or more letters, discarding numbers and punctuation.", style_table_cell)],
        [Paragraph("3. Stop-word removal", style_table_cell), Paragraph("Removes scikit-learn's 318 English stop words plus a 38-word news list (21 of them new: said, told, mr, year, …) — 339 words in all.", style_table_cell)],
        [Paragraph("4. Lemmatisation", style_table_cell), Paragraph("NLTK's WordNet lemmatiser maps plural nouns to their singular form (jobs → job, profits → profit), merging counts for the same concept.", style_table_cell)],
        [Paragraph("5. Bag of words", style_table_cell), Paragraph("CountVectorizer keeps words found in at least 5 documents and at most 30% of them, producing a 100 × 677 sparse count matrix.", style_table_cell)],
        [Paragraph("6. LDA fitting", style_table_cell), Paragraph("LatentDirichletAllocation (K = 5, α = η = 0.2, batch variational Bayes, 100 iterations) is run from 10 random seeds; the run with the highest log-likelihood (variational bound) is kept.", style_table_cell)],
        [Paragraph("7. Topic extraction", style_table_cell), Paragraph("Each row of components_ is normalised to a word distribution β; its five largest entries are the topic's top words. fit_transform returns θ, the topic mixture of each document.", style_table_cell)],
        [Paragraph("8. Streamlit UI", style_table_cell), Paragraph("Top-word table and bar charts, document–topic table with dominant topic, topic prevalence and (if labels exist) a category × topic heat map; CSV downloads for both tables.", style_table_cell)],
    ]
    t_arch = Table(arch_data, colWidths=[110, 394])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 10))

    if os.path.exists("report/assets/figure1_pipeline.png"):
        story.append(Image("report/assets/figure1_pipeline.png", width=490, height=180))
        story.append(Paragraph(
            "<b>Figure 1.</b> End-to-end pipeline: preprocessing stages (blue) turn raw articles into a count matrix, LDA (amber) learns topics and "
            "document mixtures, and Streamlit (green) displays them. Two files of logic (app.py and src/), no database or external API.",
            style_caption
        ))

    story.append(Paragraph("Backend Implementation", style_h1))
    story.append(Paragraph("6.1 Approach / Algorithm", style_h2))
    story.append(Paragraph(
        "LDA assumes a simple generative story: to write a document, first draw its topic mixture θ from a Dirichlet(α) "
        "distribution; then, for every word position, pick a topic from θ and a word from that topic's distribution β, itself "
        "drawn from a Dirichlet(η). Fitting the model inverts this story — given only the word counts, it estimates β (which "
        "words belong together) and θ (how much of each topic each document contains). scikit-learn uses variational "
        "Bayes, which maximises a lower bound on the log-likelihood.",
        style_body
    ))
    story.append(Paragraph(
        "The priors are left at scikit-learn's defaults, α = η = 1/K = 0.2. Values below 1 encourage each document to focus "
        "on few topics and each topic on few words, which suits short news articles. Because variational inference "
        "converges to a local optimum, the model is fitted from 10 random starts and the run with the highest log-likelihood "
        "is kept — a choice that uses no labels.",
        style_body
    ))

    story.append(PageBreak())

    # =========================================================
    # PAGE 4: Coherence, Code Snippets, Frontend
    # =========================================================
    story.append(Paragraph(
        "The vocabulary filter was chosen by topic coherence, also without labels (<code>eval/tune_grid.py</code>): over min_df ∈ "
        "{3, 4, 5}, max_df ∈ {0.2, 0.3} and topic–word prior η ∈ {0.2, 0.01}, five single runs each, two settings tied for the "
        "highest mean NPMI (0.146 and 0.144); the one with default priors, min_df = 5 and max_df = 0.3, was kept "
        "because its spread across seeds was smaller (0.023 vs 0.034). An earlier coarse search showed min_df = 2 clearly "
        "worse. K = 5 was then chosen from the coherence sweep in Results.",
        style_body
    ))
    story.append(Paragraph(
        "Topic quality is measured with <b>NPMI coherence</b>: for every pair of a topic's top words, NPMI(a, b) = log[P(a,b) / "
        "(P(a)P(b))] / −log P(a,b), estimated from document co-occurrence in the 2,126-article reference corpus and "
        "averaged. It ranges from −1 to 1; positive values mean the top words genuinely appear together.",
        style_body
    ))

    story.append(Paragraph("6.2 Key Code Snippets", style_h2))

    code1 = [
        [Paragraph("<b>Preprocessing — tokenise, remove stop words, lemmatise (src/preprocess.py):</b>", style_table_header)],
        [Paragraph(
            "def __call__(self, text):<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;out = []<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;for tok in TOKEN_RE.findall(text.lower()):&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# [a-z]{3,}<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if tok in self.stop:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 318 + 38 stop words<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;continue<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tok = self._lemma(tok)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# WordNet: games -> game<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if tok not in self.stop and len(tok) >= 3:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;out.append(tok)<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;return out",
            style_code
        )]
    ]
    t_code1 = Table(code1, colWidths=[504])
    t_code1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_code1)
    story.append(Spacer(1, 6))

    code2 = [
        [Paragraph("<b>LDA fitting — best of N random restarts by log-likelihood (src/topic_model.py):</b>", style_table_header)],
        [Paragraph(
            "self.X = self.vectorizer.fit_transform(self.docs)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# CountVectorizer(min_df=5, max_df=0.3)<br/>"
            "for i in range(self.n_restarts):<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;lda = LatentDirichletAllocation(n_components=5, learning_method=\"batch\",<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;max_iter=100, random_state=self.random_state + i)<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;theta = lda.fit_transform(self.X)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# document-topic mixture<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;ll = lda.score(self.X)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# approximate log-likelihood<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;if best is None or ll > best[0]:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;best = (ll, lda, theta, self.random_state + i)",
            style_code
        )]
    ]
    t_code2 = Table(code2, colWidths=[504])
    t_code2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_code2)
    story.append(Spacer(1, 6))

    code3 = [
        [Paragraph("<b>Top words — the five most probable words of each topic (src/topic_model.py):</b>", style_table_header)],
        [Paragraph(
            "comp = self.lda.components_<br/>"
            "self.topic_word = comp / comp.sum(axis=1, keepdims=True)&nbsp;&nbsp;# beta, rows sum to 1<br/>"
            "for row in self.topic_word:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;idx = np.argsort(row)[::-1][:n]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# n = 5<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;out.append([(self.vocab[i], float(row[i])) for i in idx])",
            style_code
        )]
    ]
    t_code3 = Table(code3, colWidths=[504])
    t_code3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_code3)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Frontend / UI Implementation", style_h1))
    story.append(Paragraph("7.1 Framework", style_h2))
    story.append(Paragraph(
        "Streamlit was chosen because the whole interactive app fits in one Python script with no JavaScript build step, so "
        "the UI imports the pipeline classes directly. <code>layout=\"wide\"</code> widens the page, and <code>st.cache_resource</code> caches "
        "the fitted model for each combination of corpus and parameters, so changing only the number of displayed words "
        "does not refit.",
        style_body
    ))
    story.append(Paragraph("7.2 UI Capabilities", style_h2))
    story.append(Paragraph(
        "• Choose the corpus (bundled BBC sample, pasted text or uploaded .txt / .csv files), the number of topics K "
        "(2–10), the number of top words (3–15, default 5) and the number of random restarts; an expander exposes "
        "min_df, max_df and the two preprocessing switches.<br/>"
        "• Start the analysis with one primary-styled “Discover Topics” button; a success banner reports the vocabulary "
        "size, token count, chosen seed and perplexity.<br/>"
        "• Read the top-word table (topic, top words, their probabilities, number of documents dominated by the topic) "
        "and download it as CSV; compare topics in a row of bar charts.<br/>"
        "• Browse the topic mixture of every document (dominant topic, its probability and all K proportions), download it "
        "as CSV, and view topic prevalence and — when labels are available — a category × dominant-topic heat map.",
        style_body
    ))

    story.append(PageBreak())

    # =========================================================
    # PAGE 5: UI Screenshots 7.3.1 and 7.3.2
    # =========================================================
    story.append(Paragraph("7.3 Screenshots", style_h1))

    if os.path.exists("report/assets/figure_7_3_1.png"):
        story.append(Image("report/assets/figure_7_3_1.png", width=490, height=225))
        story.append(Paragraph(
            "<b>Figure 7.3.1.</b> The app's start screen: corpus selector and model sliders in the sidebar (K = 5, top words = 5, 10 restarts) and the "
            "bundled corpus summary (100 documents, 43,141 words), ready for “Discover Topics”.",
            style_caption
        ))

    story.append(Spacer(1, 10))

    if os.path.exists("report/assets/figure_7_3_2.png"):
        story.append(Image("report/assets/figure_7_3_2.png", width=490, height=225))
        story.append(Paragraph(
            "<b>Figure 7.3.2.</b> After clicking “Discover Topics”: the fit summary (677-word vocabulary, 9,225 tokens, best of 10 restarts = seed "
            "6, perplexity 540), the top-5-words table and the start of the per-topic bar charts.",
            style_caption
        ))

    story.append(PageBreak())

    # =========================================================
    # PAGE 6: Figure 7.3.3 and Sample Input & Output
    # =========================================================
    if os.path.exists("report/assets/figure_7_3_3.png"):
        story.append(Image("report/assets/figure_7_3_3.png", width=490, height=230))
        story.append(Paragraph(
            "<b>Figure 7.3.3.</b> Lower half of the same page: the document–topic table (label, dominant topic and its probability, T1–T5 "
            "proportions) and the topic-prevalence bar chart next to the category × dominant-topic heat map.",
            style_caption
        ))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Sample Input & Output", style_h1))

    sample_io_data = [
        [Paragraph("Input (excerpt)", style_table_header), Paragraph("Pipeline stage", style_table_header), Paragraph("Output produced", style_table_header)],
        [
            Paragraph("“dutch bank to lay off 2 850 staff abn amro the netherlands largest bank is to cut 2 850 jobs as a result of falling profits”", style_table_cell),
            Paragraph("Preprocessing", style_table_cell),
            Paragraph("dutch, bank, lay, staff, abn, amro, netherlands, largest, bank, cut, job, result, falling, profit — numbers and stop words removed, jobs → job, profits → profit.", style_table_cell)
        ],
        [
            Paragraph("Same article (full text)", style_table_cell),
            Paragraph("LDA document mixture", style_table_cell),
            Paragraph("T1 0.989, T2 0.003, T3 0.003, T4 0.003, T5 0.003 → dominant Topic 1 (business & finance) with probability 0.989.", style_table_cell)
        ],
        [
            Paragraph("“lacroix label bought by us firm luxury goods group lvmh …” (business)", style_table_cell),
            Paragraph("LDA document mixture", style_table_cell),
            Paragraph("T1 0.446, T2 0.535, T3 0.006, T4 0.006, T5 0.006 → split between Topic 2 and Topic 1, although it is a business story — an example of a document LDA treats as a mixture.", style_table_cell)
        ],
        [
            Paragraph("“rich pickings for hi-tech thieves viruses trojans …” (tech)", style_table_cell),
            Paragraph("LDA document mixture", style_table_cell),
            Paragraph("T1 0.341, T2 0.283, T3 0.372, T4 0.002, T5 0.002 → no topic above 0.38: an online-fraud story mixes gadget, banking and policy vocabulary.", style_table_cell)
        ],
        [
            Paragraph("All 100 articles, K = 5", style_table_cell),
            Paragraph("Top-word extraction", style_table_cell),
            Paragraph("Five topics with five words each — reproduced in full in the Results table below.", style_table_cell)
        ],
    ]
    t_sample_io = Table(sample_io_data, colWidths=[140, 100, 264])
    t_sample_io.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_sample_io)

    story.append(PageBreak())

    # =========================================================
    # PAGE 7: Results (Deliverables, Top Words, Aggregate Metrics, Figure 2)
    # =========================================================
    story.append(Paragraph("Results (Deliverable-Specific Outputs)", style_h1))
    story.append(Paragraph(
        "<b>Deliverable:</b> the five topics discovered in the 100-article corpus with their top five words, produced by the app "
        "(Figure 7.3.2) and reproduced exactly by <code>python -m eval.run_eval</code>, which also writes every table below to "
        "<code>eval/</code>. The interpretation column is our reading of the words and the articles each topic dominates; LDA itself only "
        "outputs numbered topics.",
        style_body
    ))

    story.append(Paragraph("<b>Top five words per topic (K = 5, best of 10 restarts: seed 6)</b>", style_h2))
    res_topics = [
        [Paragraph("Topic", style_table_header), Paragraph("Top 5 words (probability)", style_table_header), Paragraph("Docs", style_table_header), Paragraph("NPMI@5", style_table_header), Paragraph("Interpretation", style_table_header)],
        [Paragraph("Topic 1", style_table_cell), Paragraph("firm (0.026), bank (0.022), company (0.019), profit (0.017), machine (0.016)", style_table_cell), Paragraph("22", style_table_cell), Paragraph("0.134", style_table_cell), Paragraph("Business & finance", style_table_cell)],
        [Paragraph("Topic 2", style_table_cell), Paragraph("labour (0.018), election (0.016), plan (0.016), government (0.013), group (0.012)", style_table_cell), Paragraph("18", style_table_cell), Paragraph("0.249", style_table_cell), Paragraph("UK politics & elections", style_table_cell)],
        [Paragraph("Topic 3", style_table_cell), Paragraph("gadget (0.042), mobile (0.028), phone (0.022), list (0.021), device (0.020)", style_table_cell), Paragraph("11", style_table_cell), Paragraph("0.411", style_table_cell), Paragraph("Consumer technology / gadgets", style_table_cell)],
        [Paragraph("Topic 4", style_table_cell), Paragraph("government (0.032), court (0.016), right (0.015), threat (0.013), lord (0.013)", style_table_cell), Paragraph("13", style_table_cell), Paragraph("0.155", style_table_cell), Paragraph("Law, courts & security", style_table_cell)],
        [Paragraph("Topic 5", style_table_cell), Paragraph("game (0.027), film (0.027), best (0.019), player (0.017), award (0.015)", style_table_cell), Paragraph("36", style_table_cell), Paragraph("0.138", style_table_cell), Paragraph("Sport & entertainment (merged)", style_table_cell)]
    ]
    t_res_topics = Table(res_topics, colWidths=[48, 204, 34, 48, 170])
    t_res_topics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_res_topics)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Aggregate metrics</b>", style_h2))
    res_agg = [
        [Paragraph("Metric", style_table_header), Paragraph("Value", style_table_header), Paragraph("Definition", style_table_header)],
        [Paragraph("Vocabulary / tokens", style_table_cell), Paragraph("677 / 9,225", style_table_cell), Paragraph("After preprocessing and the min_df / max_df filter.", style_table_cell)],
        [Paragraph("NPMI coherence", style_table_cell), Paragraph("0.196 (top-10), 0.217 (top-5)", style_table_cell), Paragraph("Mean pairwise NPMI of each topic's top words in 2,126 BBC articles (−1 to 1).", style_table_cell)],
        [Paragraph("Perplexity", style_table_cell), Paragraph("539.8", style_table_cell), Paragraph("Training perplexity of the selected model (lower = better fit).", style_table_cell)],
        [Paragraph("Purity", style_table_cell), Paragraph("0.63", style_table_cell), Paragraph("Share of documents whose dominant topic's majority category is their own.", style_table_cell)],
        [Paragraph("NMI / ARI", style_table_cell), Paragraph("0.516 / 0.383", style_table_cell), Paragraph("Agreement between dominant topics and the hidden labels (1 = identical, 0 = chance).", style_table_cell)],
        [Paragraph("Confident documents", style_table_cell), Paragraph("35 with p > 0.9; 15 with p < 0.5", style_table_cell), Paragraph("Dominant-topic probability; mean 0.743.", style_table_cell)],
        [Paragraph("Top-5 stability", style_table_cell), Paragraph("0.399 mean Jaccard", style_table_cell), Paragraph("Overlap of matched top-5 word sets across the 45 pairs of the 10 restarts.", style_table_cell)]
    ]
    t_res_agg = Table(res_agg, colWidths=[120, 140, 244])
    t_res_agg.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_res_agg)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Choosing the number of topics</b>", style_h2))
    if os.path.exists("report/assets/figure2_k_selection.png"):
        story.append(Image("report/assets/figure2_k_selection.png", width=490, height=170))
        story.append(Paragraph(
            "<b>Figure 2.</b> K = 2–10, each the best of 10 restarts. Coherence peaks at K = 5 (0.196), which was selected; training perplexity is "
            "lowest at K = 4 (528.1) and rises steadily after that.",
            style_caption
        ))

    story.append(PageBreak())

    # =========================================================
    # PAGE 8: Contingency, Preprocessing Ablation, Analysis
    # =========================================================
    story.append(Paragraph("<b>Dominant topic versus hidden section label (counts of documents)</b>", style_h2))
    res_cont = [
        [Paragraph("Category \\ Dominant topic", style_table_header), Paragraph("T1", style_table_header), Paragraph("T2", style_table_header), Paragraph("T3", style_table_header), Paragraph("T4", style_table_header), Paragraph("T5", style_table_header)],
        [Paragraph("business", style_table_cell), Paragraph("14", style_table_cell), Paragraph("3", style_table_cell), Paragraph("0", style_table_cell), Paragraph("3", style_table_cell), Paragraph("0", style_table_cell)],
        [Paragraph("entertainment", style_table_cell), Paragraph("2", style_table_cell), Paragraph("1", style_table_cell), Paragraph("0", style_table_cell), Paragraph("2", style_table_cell), Paragraph("15", style_table_cell)],
        [Paragraph("politics", style_table_cell), Paragraph("0", style_table_cell), Paragraph("14", style_table_cell), Paragraph("0", style_table_cell), Paragraph("6", style_table_cell), Paragraph("0", style_table_cell)],
        [Paragraph("sport", style_table_cell), Paragraph("0", style_table_cell), Paragraph("0", style_table_cell), Paragraph("0", style_table_cell), Paragraph("2", style_table_cell), Paragraph("18", style_table_cell)],
        [Paragraph("tech", style_table_cell), Paragraph("6", style_table_cell), Paragraph("0", style_table_cell), Paragraph("11", style_table_cell), Paragraph("0", style_table_cell), Paragraph("3", style_table_cell)],
    ]
    t_res_cont = Table(res_cont, colWidths=[154, 70, 70, 70, 70, 70])
    t_res_cont.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_res_cont)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Effect of each preprocessing step (K = 5, best of 10 restarts each)</b>", style_h2))
    res_ablation = [
        [Paragraph("Configuration", style_table_header), Paragraph("Vocab.", style_table_header), Paragraph("NPMI@10", style_table_header), Paragraph("NMI", style_table_header), Paragraph("Example top-5 lists", style_table_header)],
        [
            Paragraph("A. Generic stop words only", style_table_cell),
            Paragraph("6,290", style_table_cell),
            Paragraph("0.052", style_table_cell),
            Paragraph("0.122", style_table_cell),
            Paragraph("said, government, people, new, year | said, people, cash, year, machines", style_table_cell)
        ],
        [
            Paragraph("B. + news stop words + lemmatisation", style_table_cell),
            Paragraph("5,631", style_table_cell),
            Paragraph("0.102", style_table_cell),
            Paragraph("0.164", style_table_cell),
            Paragraph("labour, election, people, government, profit | film, world, people, team, international", style_table_cell)
        ],
        [
            Paragraph("C. + document-frequency filter (final)", style_table_cell),
            Paragraph("677", style_table_cell),
            Paragraph("0.196", style_table_cell),
            Paragraph("0.516", style_table_cell),
            Paragraph("firm, bank, company, profit, machine | labour, election, plan, government, group", style_table_cell)
        ],
    ]
    t_res_ablation = Table(res_ablation, colWidths=[130, 44, 52, 44, 234])
    t_res_ablation.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_res_ablation)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Evaluation / Analysis", style_h1))
    story.append(Paragraph(
        "Business (14/20), politics (14/20) and tech (11/20) each have a clear home topic and 18 of 20 sport articles fall in "
        "Topic 5, which, however, also takes 15 entertainment articles. All observations below come from <code>eval/</code>; labels "
        "were never used to fit or select a model.",
        style_body
    ))
    story.append(Paragraph(
        "• <b>Preprocessing decides whether the topics are readable.</b> With only a generic stop-word list, “said” appears in "
        "the top five of all five topics and “people”, “year” and “new” recur (NPMI 0.052, NMI 0.122). Adding the news "
        "stop words and lemmatisation removes the reporting verbs but leaves “people” in four topics (NPMI 0.102). "
        "The document-frequency filter — dropping words in more than 30% or fewer than 5 documents — shrinks the "
        "vocabulary from 5,631 to 677 words and roughly doubles coherence (0.196) and triples NMI (0.516). On a "
        "100-document corpus, choosing which words the model may see matters more than any LDA hyper-parameter.",
        style_body
    ))
    story.append(Paragraph(
        "• <b>LDA finds co-occurrence themes, not editorial sections.</b> Sport and entertainment articles share vocabulary "
        "such as best, award, win, play and game, so with K = 5 they merge into Topic 5, while politics splits into "
        "elections (Topic 2) and a law-and-security theme (Topic 4: terror laws, the Lord Chancellor, Guantanamo "
        "detainees) that also attracts court stories from other sections, such as a singer jailed and a musician missing bail. "
        "Both splits are reasonable readings of the text even though they disagree with the labels.",
        style_body
    ))
    story.append(Paragraph(
        "• <b>Ambiguous words blur topic boundaries.</b> “game” occurs in 11 sport and 8 tech articles (video games), and "
        "“machine” — found only in tech articles (cash-machine fraud, a virus story, a robot) — enters the business "
        "topic; together with banking words this is why 6 tech articles have Topic 1 as their dominant topic. Unigram "
        "bag-of-words cannot tell “cash machine” from “slot machine”.",
        style_body
    ))
    story.append(Paragraph(
        "• <b>A small corpus makes LDA unstable.</b> The 10 single runs reach NMI between 0.428 and 0.557, and their top-5 "
        "word sets overlap by only 0.399 (mean Jaccard). Keeping the run with the highest log-likelihood selected the "
        "most coherent run of the ten (NPMI 0.196 against a mean of 0.145), but only the 4th-best by NMI — likelihood "
        "and coherence track each other more closely than either tracks the labels.",
        style_body
    ))
    story.append(Paragraph(
        "• <b>The number of topics is only weakly determined.</b> Coherence is jagged across K (0.196 at K = 5 but 0.125 at "
        "K = 6), perplexity prefers K = 4, and NMI is actually highest at K = 4 (0.602) because sport and entertainment "
        "merge into one topic at K = 3, 4 and 5 alike. K = 5 is defensible, not uniquely correct.",
        style_body
    ))
    story.append(Paragraph(
        "• <b>Mixed membership is real and informative.</b> 35 articles are assigned to one topic with probability above 0.9, "
        "but 15 have no topic above 0.5 (e.g. the online-fraud article split 0.37 / 0.34 / 0.28). Hard dominant-topic "
        "metrics such as purity ignore this; the document–topic table in the app shows it.",
        style_body
    ))

    story.append(PageBreak())

    # =========================================================
    # PAGE 9: Recommendations, Conclusion, GitHub Link, References
    # =========================================================
    story.append(Paragraph(
        "• <b>Dataset quirk.</b> One tech article (“apple laptop is greatest gadget”) appears twice in the sample with different "
        "spacing, because exact-text de-duplication missed it. It slightly over-weights gadget vocabulary in Topic 3, "
        "whose coherence is the highest (0.411).",
        style_body
    ))

    story.append(Paragraph("Design Decisions, Limitations & Recommendations", style_h1))
    rec_data = [
        [Paragraph("Observation", style_table_header), Paragraph("Recommendation", style_table_header)],
        [Paragraph("With 100 documents, top-5 words change substantially between random starts (mean Jaccard 0.399).", style_table_cell), Paragraph("Fit on the full 2,126 articles where possible, or report topics that recur across several restarts (an ensemble), and always publish the stability figure alongside the topics.", style_table_cell)],
        [Paragraph("Sport and entertainment merge into one topic; politics splits by sub-theme.", style_table_cell), Paragraph("Try K = 6–8 with coherence checks, or guided/seeded LDA with a few anchor words per expected theme when topics must align with known categories.", style_table_cell)],
        [Paragraph("Ambiguous unigrams (game, machine) pull articles into the wrong topic.", style_table_cell), Paragraph("Add frequent bigrams as tokens (video_game, cash_machine) with gensim Phrases or CountVectorizer(ngram_range=(1, 2)) before fitting.", style_table_cell)],
        [Paragraph("WordNet lemmatisation is applied in noun form only, so verbs such as “falling” are unchanged.", style_table_cell), Paragraph("Use part-of-speech-aware lemmatisation (e.g. spaCy) or keep only nouns and adjectives, which usually gives cleaner topics.", style_table_cell)],
        [Paragraph("Exact-match de-duplication missed a near-duplicate article.", style_table_cell), Paragraph("Normalise whitespace before de-duplicating, or use MinHash near-duplicate detection.", style_table_cell)],
        [Paragraph("Topic quality is judged by one automatic metric (NPMI) and by our own interpretation.", style_table_cell), Paragraph("Add a human word-intrusion test, a second coherence measure (C_v), and an NMF baseline for comparison.", style_table_cell)],
    ]
    t_rec = Table(rec_data, colWidths=[180, 324])
    t_rec.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_rec)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Conclusion", style_h1))
    story.append(Paragraph(
        "The project applies Latent Dirichlet Allocation to a collection of 100 BBC News articles and displays the top five "
        "words of each discovered topic in an interactive Streamlit app. With K = 5, LDA recovers four recognisable "
        "themes from word counts alone — business and finance (firm, bank, company, profit), UK elections (labour, "
        "election, government), consumer gadgets (gadget, mobile, phone, device) and law and security (court, right, threat, "
        "lord) — plus a combined sport-and-entertainment topic (game, film, best, player, award). Coherence was 0.196 "
        "NPMI and the topics agree moderately with the hidden section labels (NMI 0.516, purity 0.63).",
        style_body
    ))
    story.append(Paragraph(
        "The evaluation shows that on a small corpus, preprocessing — especially removing very common and very rare "
        "words — matters more than model settings, that results vary noticeably between random starts, and that LDA "
        "groups words by how they co-occur rather than by editorial category. Each of these is addressable with more "
        "documents, phrase detection, restart ensembles or guided topics, without changing the overall preprocess → count "
        "→ LDA → top-words pipeline.",
        style_body
    ))

    story.append(Paragraph("Project Repository (GitHub Link)", style_h1))
    repo_data = [
        [Paragraph("Field", style_table_header), Paragraph("Value", style_table_header)],
        [Paragraph("GitHub Repository URL", style_table_cell), Paragraph("<b>https://github.com/Manishankar-614/LDA.git</b>", style_table_cell)],
        [Paragraph("Repository Visibility", style_table_cell), Paragraph("Public", style_table_cell)],
        [Paragraph("Branch / Commit Used for Submission", style_table_cell), Paragraph("main", style_table_cell)]
    ]
    t_repo = Table(repo_data, colWidths=[180, 324])
    t_repo.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_repo)
    story.append(Spacer(1, 6))

    story.append(Paragraph("References", style_h1))
    story.append(Paragraph(
        "• D. M. Blei, A. Y. Ng and M. I. Jordan, “Latent Dirichlet Allocation,” JMLR 3, 2003.<br/>"
        "• M. Hoffman, D. Blei and F. Bach, “Online Learning for Latent Dirichlet Allocation,” NIPS 2010.<br/>"
        "• D. Greene and P. Cunningham, “Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering,” ICML 2006 (BBC News dataset).<br/>"
        "• G. Bouma, “Normalized (Pointwise) Mutual Information in Collocation Extraction,” GSCL 2009.<br/>"
        "• scikit-learn LDA user guide — https://scikit-learn.org/stable/modules/decomposition.html<br/>"
        "• NLTK WordNet interface — https://www.nltk.org/howto/wordnet.html<br/>"
        "• Streamlit docs — https://docs.streamlit.io · Plotly Python — https://plotly.com/python/",
        style_caption
    ))

    story.append(PageBreak())

    # =========================================================
    # PAGE 10: Evaluation Sheet (Manishankar, 1GA23AI025)
    # =========================================================
    story.append(Paragraph("ADVANCE NLP (AML23702) — Lab Exercise Evaluation Sheet", style_h1))

    eval_meta = [
        [Paragraph("Field", style_table_header), Paragraph("Value", style_table_header)],
        [Paragraph("Student Name", style_table_cell), Paragraph("<b>Manishankar</b>", style_table_cell)],
        [Paragraph("USN", style_table_cell), Paragraph("<b>1GA23AI025</b>", style_table_cell)],
        [Paragraph("Lab Exercise No.", style_table_cell), Paragraph("4", style_table_cell)],
        [Paragraph("Experiment Title", style_table_cell), Paragraph("Topic Modeling with Latent Dirichlet Allocation (LDA) — Discovering Topics and Their Top Words in a Small News Document Collection", style_table_cell)],
    ]
    t_eval_meta = Table(eval_meta, colWidths=[150, 354])
    t_eval_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_eval_meta)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Marks Awarded", style_h1))
    marks_data = [
        [Paragraph("Sl. No.", style_table_header), Paragraph("Evaluation Criteria", style_table_header), Paragraph("Max Marks", style_table_header), Paragraph("Marks Obtained", style_table_header)],
        [Paragraph("1", style_table_cell), Paragraph("Problem Understanding & Scope", style_table_cell), Paragraph("5", style_table_cell), Paragraph("", style_table_cell)],
        [Paragraph("2", style_table_cell), Paragraph("Pipeline Design & Methodology", style_table_cell), Paragraph("10", style_table_cell), Paragraph("", style_table_cell)],
        [Paragraph("3", style_table_cell), Paragraph("Implementation Correctness", style_table_cell), Paragraph("10", style_table_cell), Paragraph("", style_table_cell)],
        [Paragraph("4", style_table_cell), Paragraph("Use of NLP Tools / Models", style_table_cell), Paragraph("5", style_table_cell), Paragraph("", style_table_cell)],
        [Paragraph("5", style_table_cell), Paragraph("Output Quality / Task Performance", style_table_cell), Paragraph("10", style_table_cell), Paragraph("", style_table_cell)],
        [Paragraph("6", style_table_cell), Paragraph("Analysis & Interpretation", style_table_cell), Paragraph("5", style_table_cell), Paragraph("", style_table_cell)],
        [Paragraph("7", style_table_cell), Paragraph("Documentation & Deliverables", style_table_cell), Paragraph("2.5", style_table_cell), Paragraph("", style_table_cell)],
        [Paragraph("8", style_table_cell), Paragraph("Presentation & Reproducibility", style_table_cell), Paragraph("2.5", style_table_cell), Paragraph("", style_table_cell)],
        [Paragraph("<b>Total</b>", style_table_header), Paragraph("<b>Comprehensive Lab Assessment</b>", style_table_header), Paragraph("<b>50</b>", style_table_header), Paragraph("", style_table_header)]
    ]
    t_marks = Table(marks_data, colWidths=[40, 274, 90, 100])
    t_marks.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#F8FAFC")),
    ]))
    story.append(t_marks)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Overall Comments", style_h1))
    comment_box = [
        [Paragraph("<br/><br/><br/><br/>", style_table_cell)]
    ]
    t_comment = Table(comment_box, colWidths=[504])
    t_comment.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor("#94A3B8")),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
    ]))
    story.append(t_comment)
    story.append(Spacer(1, 25))

    sig_line = [
        [Paragraph("Date of Submission: ______________", style_body_bold), Paragraph("Signature of Faculty: ______________", style_body_bold)]
    ]
    t_sig = Table(sig_line, colWidths=[250, 254])
    t_sig.setStyle(TableStyle([
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
    ]))
    story.append(t_sig)

    # Build PDF using NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {pdf_path}")


if __name__ == "__main__":
    build_pdf()
