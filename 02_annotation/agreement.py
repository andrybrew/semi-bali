"""
Step 2. Inter-annotator agreement and gold standard (Section 3.4).

annotation_rounds.csv: 582 reviews; labels of Annotators A and B in round 1 (independent,
codebook v1) and round 2 (each annotator re-examined the 280 round-1 disagreements under
codebook v2, seeing both round-1 labels); gold = final label (agreed in round 1, agreed in
round 2, or decided by a third human adjudicator for the 161 remaining disagreements).
gold_standard_582.csv: gold labels with review text, used to train and evaluate the classifier.
"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "annotation_rounds.csv"))

def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in np.union1d(a, b))
    return (po - pe) / (1 - pe)

hv = lambda s: s.isin([3, 5])
for r in (1, 2):
    a, b = d[f"A_round{r}"], d[f"B_round{r}"]
    print(f"round {r}: six-class kappa {kappa(a, b):.2f}; HV vs other kappa {kappa(hv(a), hv(b)):.2f}; "
          f"disagreements {(a != b).sum()}")
print(d.gold_source.value_counts().to_string())
print(d.gold.value_counts().sort_index().to_string())
