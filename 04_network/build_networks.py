"""
Step 4. Quarterly co-visitation networks, 2019-2024 (Table 2, Section 4.1).

Node = destination; undirected, unweighted edge = at least one reviewer reviewed both
destinations within the quarter. Louvain (resolution 1.0): partition with the highest
modularity over 100 random initialisations (seeds 0-99). Community persistence: adjusted
Rand index between consecutive quarters on destinations present in both.

Input : ../01_data/network_records.csv
Output: table2_quarterly_metrics.csv, node_metrics_quarterly.csv, edges/edges_<YYYYQn>.csv
"""
import os, itertools
import pandas as pd, networkx as nx
from networkx.algorithms.community import louvain_communities, modularity
from sklearn.metrics import adjusted_rand_score

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "edges"), exist_ok=True)
r = pd.read_csv(os.path.join(HERE, "..", "01_data", "network_records.csv.gz"))
r["date"] = pd.to_datetime(r.date, utc=True)
r = r[r.date.dt.year.between(2019, 2024) & r.reviewer_id.notna()].copy()
print(f"records 2019-2024 {len(r):,}; reviewers {r.reviewer_id.nunique():,}; "
      f"multi-destination reviewers {(r.groupby('reviewer_id').dest.nunique() > 1).sum():,}")
r["q"] = r.date.dt.year.astype(str) + "Q" + r.date.dt.quarter.astype(str)

rows, nodes, parts = [], [], {}
for q, g in sorted(r.groupby("q")):
    G = nx.Graph()
    multi = 0
    for _, h in g.groupby("reviewer_id"):
        d = sorted(h.dest.unique())
        if len(d) > 1:
            multi += 1
            G.add_edges_from(itertools.combinations(d, 2))
    best = max((louvain_communities(G, resolution=1.0, seed=s) for s in range(100)),
               key=lambda c: modularity(G, c))
    parts[q] = {n: i for i, c in enumerate(best) for n in c}
    bet = nx.betweenness_centrality(G)
    rows.append(dict(quarter=q, multi_dest_reviewers=multi, nodes=G.number_of_nodes(),
                     avg_degree=2 * G.number_of_edges() / G.number_of_nodes(),
                     density=nx.density(G), avg_clustering=nx.average_clustering(G),
                     modularity=modularity(G, best), communities=len(best),
                     max_betweenness=max(bet.values())))
    for n in G: nodes.append(dict(quarter=q, dest=n, degree=G.degree(n), betweenness=bet[n], community=parts[q][n]))
    pd.DataFrame(G.edges(), columns=["Source", "Target"]).to_csv(os.path.join(HERE, "edges", f"edges_{q}.csv"), index=False)

T = pd.DataFrame(rows)
qs = list(T.quarter)
ari = [None]
for a, b in zip(qs, qs[1:]):
    common = sorted(set(parts[a]) & set(parts[b]))
    ari.append(adjusted_rand_score([parts[a][n] for n in common], [parts[b][n] for n in common]))
T["ari_prev"] = ari
T.round(3).to_csv(os.path.join(HERE, "table2_quarterly_metrics.csv"), index=False)
pd.DataFrame(nodes).round(3).to_csv(os.path.join(HERE, "node_metrics_quarterly.csv"), index=False)
print(T.round(3).to_string(index=False))
print("mean ARI 2020Q1-2024Q4:", round(T[T.quarter >= "2020Q1"].ari_prev.mean(), 3))
