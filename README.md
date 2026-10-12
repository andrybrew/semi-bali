# SEMI Bali: reproducibility package

Code and derived data for "The Structural–Experiential Mismatch Index: A Computational
Instrument Linking Co-visitation Networks and Transformer-Based Classification of Visitor
Reviews". Every number in Tables 2–3, Figs. 1, 3 and 4 and Online Resource 1 (Tables S1–S4, Figs. S1–S4)
is produced by the scripts below. Run the steps in order; each script writes its outputs next to itself.

| Step | Folder | Script | Produces |
|---|---|---|---|
| 1a | `01_data` | `prepare_data.py` | `reviews_clean.csv.gz` (148,531 unique reviews), `corpus_for_inference.csv.gz` (82,035 text-bearing reviews), `coverage_by_destination.csv` (Table S2) |
| 1b | `01_data` | `prepare_network.py` | `network_records.csv.gz` (180,664 unique reviewer–destination records) |
| 2 | `02_annotation` | `agreement.py` | Cohen's κ for rounds 1 and 2; gold standard of 582 reviews |
| 3 | `03_classification` | `classifier_v2.ipynb` | `cv_metrics_v2.json` (Table S4), `cv_predictions_v2.csv`, `corpus_predictions_v2.csv` |
| 4 | `04_network` | `build_networks.py` | `table2_quarterly_metrics.csv` (Table 2), `node_metrics_quarterly.csv`, `edges/` |
| 5 | `05_semi` | `compute_semi.py` | `table3_semi.csv` (Table 3, Fig. 4), `tableS1_robustness.csv` (Table S1), `tableS3_experience_shares.csv` (Table S3) |
| 6 | `06_figures` | `make_figures.py` | Figs. 1, 3, 4 and Figs. S1–S4 (Fig. 2 is a manually prepared illustration) |

## Data
Two Google Maps extractions of 33 Bali destinations were collected with Apify: a network
extraction (185,585 reviewer–destination records with timestamps) and a text extraction
(173,814 records with review text and rating). Both raw exports contain reviewer display
names and are not distributed; they are available from the corresponding author on
reasonable request. Steps 1a and 1b replace names with a salted SHA-256 pseudonym
(`reviewer_id`); steps 2–5 run from the pseudonymised files in this package. The platform
export contains no stable user ID, so reviewer identity is the display name; records
without a name (3) are excluded from the networks.

Processing: two listings of the same site are merged (Campuhan Ridge Walk into Bukit
Campuhan; Tirta Empul Temple Tour renamed Tirta Empul Temple); records with the same
destination, reviewer and timestamp are removed as duplicates; review text is the English
translation supplied by the platform.

### What is not in this repository
Review text is not redistributed, in line with the platform's terms of service:
`corpus_for_inference.csv.gz` and `gold_standard_582.csv` are provided without the `text`
column. Steps 2, 4, 5 and 6 run fully from the files here; re-running step 3 (classifier
training and inference) requires the review text, available from the corresponding author
on reasonable request. The pseudonymisation salt is not published (set `SEMI_SALT`).
`tableS2_coverage.csv` in `01_data` gives per-destination coverage of both extractions (Table S2).
The Bali boundary (`bali_adm1_geoboundaries.geojson`) is from geoBoundaries gbOpen IDN ADM1
(CC BY 4.0; Runfola et al. 2020, PLOS ONE 15(4): e0231866).

## Annotation (step 2)
Two annotators labelled 582 reviews independently with codebook v1 (round 1). Each then
re-examined the 280 disagreements under codebook v2, seeing both round-1 labels (round 2).
The 161 remaining disagreements were decided by a third human adjudicator.
Label codes: 0 Uncategorized, 1 Luxury & High-End, 2 Relaxation & Leisure,
3 Adventure & Exploration, 4 Budget & Backpacking, 5 Cultural Immersion.

## Classification (step 3)
Stratified 5-fold cross-validation (seed 42) of TF-IDF + logistic regression,
bert-base-uncased and roberta-base; selection rule (highest mean macro-F1) fixed in
advance; 10 epochs, batch 16, learning rate 3e-5, max length 128, class-weighted
cross-entropy. Version 1 (4 epochs, unweighted loss) is included for transparency; it
failed on minority classes and was discarded.

## Networks (step 4)
One undirected, unweighted network per calendar quarter, 2019–2024. Node = destination;
edge = at least one reviewer reviewed both destinations in the quarter. Louvain
(resolution 1.0): partition with the highest modularity over 100 random initialisations.
Persistence: adjusted Rand index between consecutive quarters on shared destinations.

## Index (step 5)
HV = share (%) of a destination's classified reviews labelled Cultural Immersion or
Adventure & Exploration. S = mean number of other study destinations reviewed in 2024 by
the destination's reviewers. SEMI = z(HV) − z(S) (population SD), for destinations with at
least 100 classified reviews; quadrants split at the medians of HV and S. Uncertainty:
1,000 bootstrap draws (seed 42) resampling reviews (HV) and reviewers (S) within each
destination.

## Requirements
See `requirements.txt`. Steps 1, 2, 4, 5 and 6 run on a laptop in a few minutes;
step 3 needs a GPU.
