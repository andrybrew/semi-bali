"""
Step 5. Structural-Experiential Mismatch Index (Table 3, Table S1, Table S3, Fig. 4 data).

HV_i  share (%) of destination i's classified reviews (incl. Uncategorized) labelled
      Cultural Immersion (5) or Adventure & Exploration (3)
S_i   mean number of OTHER study destinations reviewed in the same window by the
      reviewers of destination i (main window: calendar year 2024)
SEMI_i = z(HV_i) - z(S_i), z = standardisation across destinations (population SD)
Destinations with >= 100 classified reviews. Quadrants split at the medians of HV and S.
Bootstrap: 1,000 draws (seed 42); reviews resampled within destination for HV,
reviewers resampled within destination for S; medians and z recomputed in every draw.

Inputs : ../01_data/network_records.csv, ../01_data/corpus_for_inference.csv,
         ../03_classification/corpus_predictions_v2.csv
Outputs: table3_semi.csv, tableS1_robustness.csv, tableS3_experience_shares.csv
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D1 = os.path.join(HERE, "..", "01_data")
LAB = {0: "Uncategorized", 1: "Luxury & High-End", 2: "Relaxation & Leisure",
       3: "Adventure & Exploration", 4: "Budget & Backpacking", 5: "Cultural Immersion"}
HV_SET, MIN_N, B, SEED = (3, 5), 100, 1000, 42

net = pd.read_csv(os.path.join(D1, "network_records.csv.gz")).dropna(subset=["reviewer_id"])
net["date"] = pd.to_datetime(net.date, utc=True)
pred = pd.read_csv(os.path.join(HERE, "..", "03_classification", "corpus_predictions_v2.csv"))
txt = pd.read_csv(os.path.join(D1, "corpus_for_inference.csv.gz"), usecols=["doc_id", "publishedAtDate"])
pred = pred.merge(txt, on="doc_id", validate="1:1")
pred["date"] = pd.to_datetime(pred.publishedAtDate, utc=True)
assert len(pred) == 82035

def k_table(r):
    """reviewer-destination pairs with k = number of other study destinations of that reviewer"""
    g = r.drop_duplicates(["dest", "reviewer_id"])
    return g.assign(k=g.reviewer_id.map(g.groupby("reviewer_id").dest.nunique() - 1))

def S_of(r):
    return k_table(r).groupby("dest").k.mean()

def HV_of(p, hv=HV_SET):
    return p.assign(h=p.pred.isin(hv)).groupby("dest").h.mean() * 100

z = lambda s: (s - s.mean()) / s.std(ddof=0)

def semi(HV, S, n, min_n=MIN_N, w=1.0):
    d = pd.DataFrame({"n": n, "HV": HV, "S": S}).dropna()
    d = d[d.n >= min_n].copy()
    d["SEMI"] = w * z(d.HV) - z(d.S)
    hiH, hiS = d.HV >= d.HV.median(), d.S >= d.S.median()
    d["quadrant"] = np.select([hiH & ~hiS, hiH & hiS, ~hiH & hiS], ["Hidden gem", "Iconic anchor", "Generic hub"], "Marginal")
    return d

win = lambda r, a, b: r[(r.date >= pd.Timestamp(a, tz="UTC")) & (r.date < pd.Timestamp(b, tz="UTC"))]
n_all = pred.groupby("dest").size()
net24 = win(net, "2024-01-01", "2025-01-01")
M = semi(HV_of(pred), S_of(net24), n_all)

# ---- bootstrap
rng = np.random.default_rng(SEED)
kt = k_table(net24)
kv = {d: g.k.to_numpy() for d, g in kt.groupby("dest") if d in M.index}
hv = {d: g.pred.isin(HV_SET).to_numpy() for d, g in pred.groupby("dest") if d in M.index}
sims, quads = [], []
for _ in range(B):
    HVb = pd.Series({d: rng.choice(a, a.size).mean() * 100 for d, a in hv.items()})
    Sb = pd.Series({d: rng.choice(a, a.size).mean() for d, a in kv.items()})
    Db = semi(HVb, Sb, M.n, min_n=0)
    sims.append(Db.SEMI); quads.append(Db.quadrant == M.quadrant)
sims, quads = pd.DataFrame(sims), pd.DataFrame(quads)
M["SEMI_lo90"], M["SEMI_hi90"] = sims.quantile(0.05), sims.quantile(0.95)
M["quadrant_stability"] = quads.mean() * 100
order = ["Hidden gem", "Iconic anchor", "Generic hub", "Marginal"]
M["qorder"] = M.quadrant.map(order.index)
M = M.sort_values(["qorder", "SEMI"], ascending=[True, False]).drop(columns="qorder")
M.round({"HV": 1, "S": 2, "SEMI": 2, "SEMI_lo90": 2, "SEMI_hi90": 2, "quadrant_stability": 0}) \
 .to_csv(os.path.join(HERE, "table3_semi.csv"))

rho = lambda a, b: a.rank().corr(b.rank())
print(f"destinations {len(M)}; median HV {M.HV.median():.1f}; median S {M.S.median():.2f}; "
      f"Spearman(HV,S) {rho(M.HV, M.S):.2f}; Spearman(S, 2024 reviewers) {rho(M.S, kt.groupby('dest').size().reindex(M.index)):.2f}")
print(M.round(2).to_string())

# ---- Table S1 robustness
def compare(name, D):
    c = D.index.intersection(M.index)
    return dict(specification=name, destinations=len(D), spearman_vs_main=round(rho(D.SEMI[c], M.SEMI[c]), 2),
                same_quadrant_pct=round((D.quadrant[c] == M.quadrant[c]).mean() * 100),
                spearman_HV_S=round(rho(D.HV, D.S), 2))
S_q24 = pd.concat([S_of(win(net, f"2024-{m:02d}-01", f"{2024 + (m == 10)}-{(m + 3 - 1) % 12 + 1:02d}-01"))
                   for m in (1, 4, 7, 10)], axis=1).mean(axis=1)
p23 = pred[pred.date >= pd.Timestamp("2023-01-01", tz="UTC")]
pal = win(pred, "2024-01-01", "2025-02-01")
rows = [
    compare("Main: S 2024; HV all years", M),
    compare("S 2023-2024", semi(HV_of(pred), S_of(win(net, "2023-01-01", "2025-01-01")), n_all)),
    compare("S per quarter of 2024 (mean)", semi(HV_of(pred), S_q24, n_all)),
    compare("HV from 2023 onwards", semi(HV_of(p23), S_of(net24), p23.groupby("dest").size())),
    compare("S 2019-2024", semi(HV_of(pred), S_of(win(net, "2019-01-01", "2025-01-01")), n_all)),
    compare("S Oct 2024-Jan 2025", semi(HV_of(pred), S_of(win(net, "2024-10-01", "2025-02-01")), n_all)),
    compare("Aligned window: S and HV Jan 2024-Jan 2025",
            semi(HV_of(pal), S_of(win(net, "2024-01-01", "2025-02-01")), pal.groupby("dest").size())),
    compare("HV = Cultural Immersion only", semi(HV_of(pred, (5,)), S_of(net24), n_all)),
    compare("HV = Adventure & Exploration only", semi(HV_of(pred, (3,)), S_of(net24), n_all)),
    compare("HV adds Budget & Backpacking", semi(HV_of(pred, (3, 4, 5)), S_of(net24), n_all)),
    compare("HV excludes Uncategorized from denominator",
            semi(HV_of(pred[pred.pred != 0]), S_of(net24), pred[pred.pred != 0].groupby("dest").size())),
    compare("Weight on z(HV) = 0.5", semi(HV_of(pred), S_of(net24), n_all, w=0.5)),
    compare("Weight on z(HV) = 2", semi(HV_of(pred), S_of(net24), n_all, w=2.0)),
    compare("Minimum 50 classified reviews", semi(HV_of(pred), S_of(net24), n_all, min_n=50)),
]
S1 = pd.DataFrame(rows)
S1.to_csv(os.path.join(HERE, "tableS1_robustness.csv"), index=False)
print(S1.to_string(index=False))

# ---- Table S3 experience shares (all 33 destinations)
S3 = pd.crosstab(pred.dest, pred.pred, normalize="index").mul(100).rename(columns=LAB)
S3.insert(0, "n", n_all)
S3.round(1).to_csv(os.path.join(HERE, "tableS3_experience_shares.csv"))
