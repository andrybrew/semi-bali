"""
Step 6. Figures 1-4 and Online Resource Figs. S1-S4.

Inputs : ../01_data/bali_adm1_geoboundaries.geojson  (geoBoundaries gbOpen IDN ADM1, CC BY 4.0;
                                                      Runfola et al. 2020), Bali feature only
         ../01_data/destination_coordinates.csv
         ../04_network/edges/edges_<YYYYQn>.csv, ../04_network/node_metrics_quarterly.csv
         ../05_semi/table3_semi.csv
Outputs: fig1_workflow.png, fig2_edgelist.png, fig3a_2019.png, fig3b_2020.png,
         figS1_2021.png ... figS4_2024.png, fig4_semi.png  (300 dpi)
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(HERE, "..", *a)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
DPI = 300
COL = ["#2b6cb0", "#dd6b20", "#2f855a", "#9b2c2c", "#6b46c1"]

# ---------------------------------------------------------------- Fig. 1 workflow
def box(ax, x, y, w, h, text, fc="#ffffff", bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=fc, ec="#333333", lw=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=7.6, weight="bold" if bold else "normal", wrap=True)

def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333"))

fig, ax = plt.subplots(figsize=(7.2, 8.2)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
W, H = 0.46, 0.075
L, R, C = 0.245, 0.755, 0.5
box(ax, C, 0.95, 0.62, 0.06, "Google Maps reviews of 33 Bali destinations (Apify)", "#edf2f7", True)
box(ax, L, 0.84, W, H, "Network extraction\n180,664 unique reviewer–destination records")
box(ax, R, 0.84, W, H, "Text extraction\n82,035 unique text-bearing reviews")
box(ax, L, 0.72, W, H, "Pseudonymised reviewers;\nreviewer co-occurrence per quarter")
box(ax, R, 0.72, W, H, "Human annotation: 582 reviews,\ntwo annotators + adjudicator")
box(ax, L, 0.60, W, H, "24 quarterly co-visitation\nnetworks, 2019–2024")
box(ax, R, 0.60, W, H, "TF-IDF, BERT, RoBERTa\n5-fold CV; RoBERTa selected")
box(ax, L, 0.48, W, H, "Density, clustering, Louvain\nmodularity, ARI  (RQ1)", "#fefcbf")
box(ax, R, 0.48, W, H, "Review-level labels, 5 experience\ncategories + Uncategorized")
box(ax, L, 0.36, W, H, "Structural embeddedness S\n(other destinations per reviewer, 2024)", "#ebf8ff")
box(ax, R, 0.36, W, H, "Experiential distinctiveness HV\n(% Cultural Immersion + Adventure)", "#ebf8ff")
box(ax, C, 0.23, 0.62, H, "SEMI = z(HV) − z(S) and four-quadrant typology\nassociation of S and HV  (RQ2)", "#fefcbf", True)
box(ax, C, 0.10, 0.62, H, "1,000 bootstrap draws and alternative specifications\n(time windows, HV definition, weights, threshold)  (RQ3)", "#fefcbf")
for x in (L, R):
    arrow(ax, x, 0.95 - 0.03 if False else 0.92, x, 0.84 + H / 2)
    for y0, y1 in ((0.84, 0.72), (0.72, 0.60), (0.60, 0.48), (0.48, 0.36)):
        arrow(ax, x, y0 - H / 2, x, y1 + H / 2)
    arrow(ax, x, 0.36 - H / 2, C + (x - C) * 0.5, 0.23 + H / 2)
arrow(ax, C, 0.23 - H / 2, C, 0.10 + H / 2)
fig.savefig(os.path.join(HERE, "fig1_workflow.png"), dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 2 edge-list construction
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.25, 0.8, 1]})
for a in axs: a.axis("off")
rows = [("Reviewer 1", "A, B, C"), ("Reviewer 2", "B, D"), ("Reviewer 3", "A")]
axs[0].set_title("(1) Reviews posted in one quarter", fontsize=8.5, loc="left")
axs[0].table(cellText=rows, colLabels=["Pseudonymised reviewer", "Destinations reviewed"], loc="center",
             cellLoc="center", colWidths=[0.55, 0.45])
t0 = axs[0].tables[0]; t0.auto_set_font_size(False); t0.set_fontsize(7.5); t0.scale(1, 1.6)
axs[1].set_title("(2) Destination pairs", fontsize=8.5, loc="left")
axs[1].table(cellText=[("A", "B"), ("A", "C"), ("B", "C"), ("B", "D")], colLabels=["Source", "Target"],
             loc="center", cellLoc="center")
t1 = axs[1].tables[0]; t1.auto_set_font_size(False); t1.set_fontsize(7.5); t1.scale(1, 1.6)
axs[2].set_title("(3) Unweighted co-visitation network", fontsize=8.5, loc="left")
pos = {"A": (0.2, 0.75), "B": (0.55, 0.45), "C": (0.2, 0.2), "D": (0.9, 0.6)}
for u, v in [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D")]:
    axs[2].plot(*zip(pos[u], pos[v]), color="#666666", lw=1.2, zorder=1)
for n, (x, y) in pos.items():
    axs[2].scatter(x, y, s=420, color="#2b6cb0", zorder=2); axs[2].text(x, y, n, color="white", ha="center", va="center", weight="bold")
axs[2].set_xlim(0, 1.05); axs[2].set_ylim(0, 1)
axs[2].text(0.02, -0.08, "Reviewer 3 reviewed one study destination and adds no edge.", fontsize=7, transform=axs[2].transAxes)
fig.savefig(os.path.join(HERE, "fig2_edgelist.png"), dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 3 and S1-S4 network maps
geo = json.load(open(P("01_data", "bali_adm1_geoboundaries.geojson")))
coords = pd.read_csv(P("01_data", "destination_coordinates.csv"), index_col=0)
nodes = pd.read_csv(P("04_network", "node_metrics_quarterly.csv"))
k = 1 / np.cos(np.radians(8.5))

def draw_bali(ax):
    for poly in geo["features"][0]["geometry"]["coordinates"]:
        ring = np.array(poly[0])
        ax.add_patch(Polygon(np.c_[ring[:, 0] * k, ring[:, 1]], closed=True, fc="#f4f4f2", ec="#9a9a9a", lw=0.5, zorder=0))

def year_figure(year, fname, label=None):
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.6))
    for qi, ax in enumerate(axs.flat, start=1):
        q = f"{year}Q{qi}"
        draw_bali(ax)
        e = pd.read_csv(P("04_network", "edges", f"edges_{q}.csv"))
        nq = nodes[nodes.quarter == q].set_index("dest")
        xy = lambda d: (coords.lng[d] * k, coords.lat[d])
        for s, t in zip(e.Source, e.Target):
            same = nq.community[s] == nq.community[t]
            (x0, y0), (x1, y1) = xy(s), xy(t)
            ax.plot([x0, x1], [y0, y1], lw=0.35, alpha=0.55,
                    color=COL[nq.community[s] % 5] if same else "#bdbdbd", zorder=1)
        for d, r in nq.iterrows():
            x, y = xy(d)
            ax.scatter(x, y, s=8 + 3.2 * r.degree, color=COL[r.community % 5], ec="white", lw=0.4, zorder=3)
        for d in nq.degree.nlargest(3).index:
            x, y = xy(d); ax.text(x + 0.012, y + 0.012, d, fontsize=5.5, zorder=4)
        ax.set_xlim(114.42 * k, 115.73 * k); ax.set_ylim(-8.88, -8.04); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"Q{qi} {year}: {len(nq)} nodes, {len(e)} edges", fontsize=8)
    if label: fig.text(0.01, 0.98, label, fontsize=11, weight="bold", va="top")
    fig.text(0.5, 0.01, "Node size: degree. Colour: Louvain community within the quarter (not comparable across panels); "
             "grey edges link different communities.\nBoundary: geoBoundaries IDN ADM1 (CC BY 4.0).",
             ha="center", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(os.path.join(HERE, fname), dpi=DPI, facecolor="white"); plt.close(fig)

year_figure(2019, "fig3a_2019.png"); year_figure(2020, "fig3b_2020.png")
for i, y in enumerate(range(2021, 2025), start=1): year_figure(y, f"figS{i}_{y}.png")

# ---------------------------------------------------------------- Fig. 4 SEMI scatter
t = pd.read_csv(P("05_semi", "table3_semi.csv"), index_col=0)
short = {"Garuda Wisnu Kencana Cultural Park": "GWK", "Sacred Monkey Forest Sanctuary": "Monkey Forest",
         "Crystal Bay Nusa Penida": "Crystal Bay", "Besakih Great Temple": "Besakih", "Penida Island": "Penida Is."}
mk = {"Hidden gem": ("o", "#2b6cb0"), "Iconic anchor": ("s", "#2f855a"), "Generic hub": ("^", "#dd6b20"), "Marginal": ("D", "#b7791f")}
fig, ax = plt.subplots(figsize=(7.2, 5.2))
for qn, (m, c) in mk.items():
    s = t[t.quadrant == qn]; ax.scatter(s.S, s.HV, marker=m, color=c, s=34, label=qn, zorder=3)
for d, r in t.iterrows():
    nm = short.get(d, d.replace(" Beach", "").replace(" Temple", "").replace(" Waterfall", "").replace(" Rice Terrace", ""))
    ax.annotate(nm, (r.S, r.HV), xytext=(4, 3), textcoords="offset points", fontsize=6.8)
ax.axvline(t.S.median(), ls="--", color="#888888", lw=0.8); ax.axhline(t.HV.median(), ls="--", color="#888888", lw=0.8)
ax.set_xlabel("Structural embeddedness S (mean number of other study destinations reviewed per reviewer, 2024)")
ax.set_ylabel("Experiential distinctiveness HV (% Cultural Immersion + Adventure & Exploration)")
ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(HERE, "fig4_semi.png"), dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("figures written")
