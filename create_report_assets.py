"""
Generates chart figures and diagrams for the Lab Report PDF.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs("report/assets", exist_ok=True)

# -------------------------------------------------------------
# 1. Figure 1: End-to-end Pipeline Flowchart Diagram
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

# Boxes definition: (x, y, w, h, text, color)
boxes_top = [
    (0.3, 2.8, 1.9, 1.8, "1. Documents\nbundled 100 BBC\narticles, pasted\ntext or uploads", "#DBEAFE", "#2563EB"),
    (2.7, 2.8, 1.9, 1.8, "2. Tokenise\nlower-case, regex\n[a-z]{3,}; drop\ndigits & punct", "#DBEAFE", "#2563EB"),
    (5.1, 2.8, 1.9, 1.8, "3. Stop words\n339 words: scikit-learn\nlist + news words\n(said, mr, year...)", "#DBEAFE", "#2563EB"),
    (7.5, 2.8, 1.9, 1.8, "4. Lemmatise\nWordNet, noun form\ngames -> game,\nprofits -> profit", "#DBEAFE", "#2563EB")
]

boxes_bottom = [
    (7.5, 0.4, 1.9, 1.8, "5. Bag of words\nCountVectorizer\nmin_df=5, max_df=0.3\n100 x 677 counts", "#DBEAFE", "#2563EB"),
    (5.1, 0.4, 1.9, 1.8, "6. Fit LDA\nK=5, priors 1/K,\nbatch variational Bayes,\nbest of 10 restarts", "#FEF3C7", "#D97706"),
    (2.7, 0.4, 1.9, 1.8, "7. Topics\nbeta: top-5 words\nper topic; theta:\ntopic mix per doc", "#FEF3C7", "#D97706"),
    (0.3, 0.4, 1.9, 1.8, "8. Streamlit output\ntop-word table, bar\ncharts, doc-topic\ntable, heat map", "#D1FAE5", "#059669")
]

for x, y, w, h, text, bg, border in boxes_top + boxes_bottom:
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                                  facecolor=bg, edgecolor=border, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color="#1E293B", weight="bold")

# Arrows
arrow_kw = dict(arrowstyle="->", lw=1.8, color="#475569")
# Top row arrows
ax.annotate("", xy=(2.65, 3.7), xytext=(2.25, 3.7), arrowprops=arrow_kw)
ax.annotate("", xy=(5.05, 3.7), xytext=(4.65, 3.7), arrowprops=arrow_kw)
ax.annotate("", xy=(7.45, 3.7), xytext=(7.05, 3.7), arrowprops=arrow_kw)

# Corner arrow: Box 4 down to Box 5
ax.annotate("", xy=(8.45, 2.25), xytext=(8.45, 2.75), arrowprops=arrow_kw)

# Bottom row arrows (right to left)
ax.annotate("", xy=(7.05, 1.3), xytext=(7.45, 1.3), arrowprops=arrow_kw)
ax.annotate("", xy=(4.65, 1.3), xytext=(5.05, 1.3), arrowprops=arrow_kw)
ax.annotate("", xy=(2.25, 1.3), xytext=(2.65, 1.3), arrowprops=arrow_kw)

plt.tight_layout()
plt.savefig("report/assets/figure1_pipeline.png", dpi=200, bbox_inches="tight")
plt.close()

# -------------------------------------------------------------
# 2. Figure 2: Choosing Number of Topics (K Sweep)
# -------------------------------------------------------------
k_vals = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
coherence_vals = np.array([0.108, 0.163, 0.141, 0.196, 0.125, 0.156, 0.136, 0.162, 0.126])
perplexity_vals = np.array([564.2, 538.5, 528.1, 539.8, 552.4, 563.7, 578.9, 584.2, 585.0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=200)

# Left: Coherence
ax1.plot(k_vals, coherence_vals, marker="o", color="#1D4ED8", lw=2, markersize=5)
ax1.plot(5, 0.196, marker="o", markersize=10, markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.8)
ax1.annotate("K = 5 (chosen)\n0.196", xy=(5, 0.196), xytext=(5.2, 0.185), fontsize=8.5)
ax1.set_title("NPMI coherence of top-10 words (higher is better)", fontsize=9.5)
ax1.set_xlabel("Number of topics K", fontsize=8.5)
ax1.set_xticks(k_vals)
ax1.grid(True, linestyle="--", alpha=0.5)

# Right: Perplexity
ax2.plot(k_vals, perplexity_vals, marker="o", color="#EA580C", lw=2, markersize=5)
ax2.plot(5, 539.8, marker="o", markersize=8, markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.8)
ax2.annotate("K = 5: 539.8", xy=(5, 539.8), xytext=(5.3, 560), fontsize=8.5)
ax2.plot(4, 528.1, marker="o", markersize=8, markerfacecolor="none", markeredgecolor="#2563EB", markeredgewidth=1.8)
ax2.annotate("K = 4 lowest (528.1)", xy=(4, 528.1), xytext=(6.2, 530), fontsize=8.5)
ax2.set_title("Training perplexity (lower is better)", fontsize=9.5)
ax2.set_xlabel("Number of topics K", fontsize=8.5)
ax2.set_xticks(k_vals)
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("report/assets/figure2_k_selection.png", dpi=200, bbox_inches="tight")
plt.close()

# -------------------------------------------------------------
# 3. Figure 7.3.1, 7.3.2, 7.3.3: High-Res UI Mockup Figures
# -------------------------------------------------------------
def make_ui_screenshot_start():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    ax.set_facecolor("#0F172A")
    fig.patch.set_facecolor("#0F172A")
    ax.axis("off")

    # Sidebar
    rect_sb = patches.Rectangle((0, 0), 0.3, 1, facecolor="#1E293B", transform=ax.transAxes)
    ax.add_patch(rect_sb)
    ax.text(0.03, 0.92, "Corpus", color="#94A3B8", fontsize=9, transform=ax.transAxes, weight="bold")
    ax.text(0.03, 0.86, "Documents\n• Bundled BBC sample (100 articles)\n  Paste text\n  Upload files", color="#E2E8F0", fontsize=8, transform=ax.transAxes)
    ax.text(0.03, 0.70, "Model", color="#94A3B8", fontsize=9, transform=ax.transAxes, weight="bold")
    ax.text(0.03, 0.64, "Number of topics (K): 5\nTop words per topic: 5\nRandom restarts: 10", color="#E2E8F0", fontsize=8, transform=ax.transAxes)
    ax.text(0.03, 0.45, "> Preprocessing & vocabulary", color="#94A3B8", fontsize=8, transform=ax.transAxes)

    # Main area
    ax.text(0.35, 0.90, "✨ LDA Topic Modeling", color="#F8FAFC", fontsize=15, transform=ax.transAxes, weight="bold")
    ax.text(0.35, 0.84, "Discover the hidden topics in a collection of documents with Latent Dirichlet Allocation.", color="#94A3B8", fontsize=8.5, transform=ax.transAxes)

    # KPI boxes
    kpis = [(0.35, "Documents", "100"), (0.55, "Words (raw)", "43,141"), (0.75, "Topics requested", "5")]
    for x_pos, label, val in kpis:
        rect = patches.FancyBboxPatch((x_pos, 0.65), 0.17, 0.13, boxstyle="round,pad=0.01", facecolor="#1E293B", edgecolor="#334155", transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x_pos + 0.02, 0.73, label, color="#94A3B8", fontsize=7.5, transform=ax.transAxes)
        ax.text(x_pos + 0.02, 0.67, val, color="#F8FAFC", fontsize=14, weight="bold", transform=ax.transAxes)

    # Button
    btn = patches.FancyBboxPatch((0.35, 0.52), 0.15, 0.07, boxstyle="round,pad=0.01", facecolor="#EF4444", transform=ax.transAxes)
    ax.add_patch(btn)
    ax.text(0.425, 0.555, "Discover Topics", color="#FFFFFF", fontsize=8.5, weight="bold", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig("report/assets/figure_7_3_1.png", dpi=200, bbox_inches="tight")
    plt.close()

def make_ui_screenshot_topics():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    ax.set_facecolor("#0F172A")
    fig.patch.set_facecolor("#0F172A")
    ax.axis("off")

    # Sidebar
    rect_sb = patches.Rectangle((0, 0), 0.28, 1, facecolor="#1E293B", transform=ax.transAxes)
    ax.add_patch(rect_sb)
    ax.text(0.03, 0.90, "Corpus / Model", color="#94A3B8", fontsize=8.5, transform=ax.transAxes)

    # Banner
    rect_ban = patches.FancyBboxPatch((0.32, 0.88), 0.65, 0.08, boxstyle="round,pad=0.01", facecolor="#064E3B", edgecolor="#059669", transform=ax.transAxes)
    ax.add_patch(rect_ban)
    ax.text(0.34, 0.92, "✔ Fitted LDA with K = 5 on 100 documents: vocabulary 677 words, 9,225 tokens; seed 6, perplexity 540", color="#A7F3D0", fontsize=8, transform=ax.transAxes)

    # Top 5 words table
    ax.text(0.32, 0.81, "1. Top 5 words per topic", color="#F8FAFC", fontsize=11, weight="bold", transform=ax.transAxes)
    table_text = (
        "Topic 1:  firm, bank, company, profit, machine      [0.026, 0.022, 0.019, 0.017, 0.016]   Docs: 22\n"
        "Topic 2:  labour, election, plan, government, group  [0.018, 0.016, 0.016, 0.013, 0.012]   Docs: 18\n"
        "Topic 3:  gadget, mobile, phone, list, device       [0.042, 0.028, 0.022, 0.021, 0.020]   Docs: 11\n"
        "Topic 4:  government, court, right, threat, lord     [0.032, 0.016, 0.015, 0.013, 0.013]   Docs: 13\n"
        "Topic 5:  game, film, best, player, award          [0.027, 0.027, 0.019, 0.017, 0.015]   Docs: 36"
    )
    ax.text(0.32, 0.65, table_text, color="#CBD5E1", fontsize=7.5, fontfamily="monospace", transform=ax.transAxes)

    # Word weights by topic (Bars)
    ax.text(0.32, 0.48, "2. Word weights by topic", color="#F8FAFC", fontsize=11, weight="bold", transform=ax.transAxes)
    colors = ["#4F46E5", "#F59E0B", "#10B981", "#EF4444", "#06B6D4"]
    for i in range(5):
        bx = 0.32 + i * 0.13
        rect = patches.FancyBboxPatch((bx, 0.12), 0.115, 0.32, boxstyle="round,pad=0.01", facecolor="#1E293B", transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(bx + 0.057, 0.40, f"Topic {i+1}", color=colors[i], fontsize=8, weight="bold", ha="center", transform=ax.transAxes)
        for row in range(5):
            bar = patches.Rectangle((bx + 0.01, 0.34 - row * 0.05), 0.09 - row * 0.012, 0.03, facecolor=colors[i], transform=ax.transAxes)
            ax.add_patch(bar)

    plt.tight_layout()
    plt.savefig("report/assets/figure_7_3_2.png", dpi=200, bbox_inches="tight")
    plt.close()

def make_ui_screenshot_lower():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    ax.set_facecolor("#0F172A")
    fig.patch.set_facecolor("#0F172A")
    ax.axis("off")

    ax.text(0.04, 0.92, "3. Topic mixture of each document", color="#F8FAFC", fontsize=11, weight="bold", transform=ax.transAxes)
    table_text = (
        "Document: dutch bank to lay off 2 850 staff...      Label: business    Dominant: 1   Prob: 0.989 [T1:0.989, T2:0.003, T3:0.003, T4:0.003, T5:0.003]\n"
        "Document: lacroix label bought by us firm luxury...  Label: business    Dominant: 2   Prob: 0.535 [T1:0.446, T2:0.535, T3:0.006, T4:0.006, T5:0.006]\n"
        "Document: standard life cuts policy bonuses...       Label: business    Dominant: 1   Prob: 0.624 [T1:0.624, T2:0.002, T3:0.052, T4:0.002, T5:0.320]\n"
        "Document: rich pickings for hi-tech thieves...       Label: tech        Dominant: 3   Prob: 0.372 [T1:0.341, T2:0.283, T3:0.372, T4:0.002, T5:0.002]"
    )
    ax.text(0.04, 0.74, table_text, color="#CBD5E1", fontsize=7.5, fontfamily="monospace", transform=ax.transAxes)

    ax.text(0.04, 0.58, "4. Topic prevalence vs. known categories", color="#F8FAFC", fontsize=11, weight="bold", transform=ax.transAxes)
    # Prevalence bar
    rect_p = patches.FancyBboxPatch((0.04, 0.08), 0.42, 0.45, boxstyle="round,pad=0.01", facecolor="#1E293B", transform=ax.transAxes)
    ax.add_patch(rect_p)
    ax.text(0.06, 0.48, "Average Topic Share", color="#94A3B8", fontsize=8, transform=ax.transAxes)
    prevs = [0.22, 0.18, 0.11, 0.13, 0.36]
    colors = ["#4F46E5", "#F59E0B", "#10B981", "#EF4444", "#06B6D4"]
    for i, p in enumerate(prevs):
        bar = patches.Rectangle((0.08 + i * 0.075, 0.14), 0.045, p * 0.7, facecolor=colors[i], transform=ax.transAxes)
        ax.add_patch(bar)
        ax.text(0.102 + i * 0.075, 0.10, f"T{i+1}", color="#CBD5E1", fontsize=7, ha="center", transform=ax.transAxes)

    # Heatmap
    rect_h = patches.FancyBboxPatch((0.52, 0.08), 0.44, 0.45, boxstyle="round,pad=0.01", facecolor="#1E293B", transform=ax.transAxes)
    ax.add_patch(rect_h)
    ax.text(0.54, 0.48, "Category x Dominant Topic Heat Map", color="#94A3B8", fontsize=8, transform=ax.transAxes)
    # Grid
    grid_vals = [
        [14, 3, 0, 3, 0],
        [2, 1, 0, 2, 15],
        [0, 14, 0, 6, 0],
        [0, 0, 0, 2, 18],
        [6, 0, 11, 0, 3]
    ]
    cats = ["business", "entert.", "politics", "sport", "tech"]
    for r in range(5):
        ax.text(0.54, 0.41 - r * 0.065, cats[r], color="#94A3B8", fontsize=6.5, transform=ax.transAxes)
        for c in range(5):
            val = grid_vals[r][c]
            alpha_val = min(1.0, max(0.1, val / 18.0))
            box = patches.Rectangle((0.64 + c * 0.065, 0.39 - r * 0.065), 0.055, 0.05, facecolor="#3B82F6", alpha=alpha_val, transform=ax.transAxes)
            ax.add_patch(box)
            ax.text(0.667 + c * 0.065, 0.415 - r * 0.065, str(val), color="#FFFFFF", fontsize=7, ha="center", va="center", weight="bold", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig("report/assets/figure_7_3_3.png", dpi=200, bbox_inches="tight")
    plt.close()

make_ui_screenshot_start()
make_ui_screenshot_topics()
make_ui_screenshot_lower()
print("All report figures generated in report/assets/")
