"""
Step 1b. Reviewer-destination records for the co-visitation networks.

Input : data jurnal.xlsx (network extraction, 185,585 reviewer-destination records with
        timestamps, 33 destinations). Contains reviewer display names and is not distributed.
Output: network_records.csv  180,664 unique records (dest, reviewer_id, date); reviewer
        names replaced by the same salted SHA-256 pseudonym as in prepare_data.py.
Usage : python prepare_network.py "path/to/data jurnal.xlsx"
"""
import sys, os, hashlib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SALT = os.environ["SEMI_SALT"]   # fixed private string; results do not depend on it

raw = pd.read_excel(sys.argv[1])
assert len(raw) == 185585, len(raw)
raw["title"] = raw.title.replace({"Campuhan Ridge Walk": "Bukit Campuhan",
                                  "Tirta Empul Temple Tour": "Tirta Empul Temple"})
assert raw.title.nunique() == 33
x = raw.drop_duplicates(subset=["title", "name", "date"]).copy()
print(f"records without reviewer name (excluded from networks): {x.name.isna().sum()}")
print(f"raw {len(raw):,}  unique {len(x):,}")
x["reviewer_id"] = [hashlib.sha256((SALT + str(n)).encode()).hexdigest()[:16] if pd.notna(n) else "" for n in x.name]
x["name_single_token"] = x.name.astype(str).str.split().str.len() == 1
x = x.rename(columns={"title": "dest"})[["dest", "reviewer_id", "name_single_token", "date"]]
x.to_csv(os.path.join(HERE, "network_records.csv.gz"), index=False)
