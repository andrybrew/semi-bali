"""
Step 1a. Review records and classification corpus.

Input : Full_Data_Combined.xlsx (raw Apify export of Google Maps reviews,
        173,814 rows, 33 destinations). Contains reviewer display names and is
        therefore not distributed; available from the corresponding author.
Output (written to this folder):
  reviews_clean.csv          148,531 unique reviews; reviewer names replaced by
                             a salted SHA-256 pseudonym (reviewer_id)
  corpus_for_inference.csv   82,035 unique text-bearing reviews (classification corpus)
  coverage_by_destination.csv  raw/unique counts, duplicates, first/last date, per-year counts
Usage : python prepare_data.py path/to/Full_Data_Combined.xlsx
"""
import sys, os, hashlib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SALT = os.environ["SEMI_SALT"]   # fixed private string; results do not depend on it

raw = pd.read_excel(sys.argv[1])
assert len(raw) == 173814, len(raw)

# 1. Harmonise listings referring to the same site
raw.loc[raw.title == "Campuhan Ridge Walk", "title"] = "Bukit Campuhan"
raw.loc[raw.title == "Tirta Empul Temple Tour", "title"] = "Tirta Empul Temple"
assert raw.title.nunique() == 33

# 2. Remove duplicates: same destination, reviewer and timestamp
x = raw.drop_duplicates(subset=["title", "name", "publishedAtDate"]).copy()
print(f"raw {len(raw):,}  duplicates {len(raw)-len(x):,}  unique {len(x):,}")

# 3. Pseudonymise reviewers
x["reviewer_id"] = [hashlib.sha256((SALT + str(n)).encode()).hexdigest()[:16] if pd.notna(n) else "" for n in x.name]
x["name_single_token"] = x.name.astype(str).str.split().str.len() == 1   # higher risk of name collision
x["has_text"] = x.textTranslated.notna() & (x.textTranslated.astype(str).str.strip() != "")
x = x.rename(columns={"title": "dest"})
x[["dest", "reviewer_id", "name_single_token", "publishedAtDate", "stars", "has_text"]].to_csv(os.path.join(HERE, "reviews_clean.csv.gz"), index=False)

# 4. Classification corpus (text supplied by the platform, English translation)
c = x[x.has_text].copy()
c.insert(0, "doc_id", [f"D{i:06d}" for i in range(len(c))])
c = c.rename(columns={"textTranslated": "text"})
c["text"] = c.text.astype(str).str.strip()
c[["doc_id", "dest", "publishedAtDate", "stars", "text"]].to_csv(os.path.join(HERE, "corpus_for_inference.csv.gz"), index=False)
print(f"text-bearing unique reviews {len(c):,}")

# 5. Coverage audit
raw["y"] = pd.to_datetime(raw.publishedAtDate, utc=True).dt.year
x["y"] = pd.to_datetime(x.publishedAtDate, utc=True).dt.year
cov = pd.DataFrame({
    "reviews_raw": raw.groupby("title").size(),
    "reviews_unique": x.groupby("dest").size(),
    "text_unique": c.groupby("dest").size(),
    "first_review": x.groupby("dest").publishedAtDate.min().str[:10],
    "last_review": x.groupby("dest").publishedAtDate.max().str[:10],
})
cov["duplicates"] = cov.reviews_raw - cov.reviews_unique
yr = pd.crosstab(x.dest, x.y).reindex(columns=range(2019, 2025), fill_value=0).add_prefix("unique_")
cov = cov.join(yr).sort_values("first_review")
cov.index.name = "dest"
cov.to_csv(os.path.join(HERE, "coverage_by_destination.csv"))
print(cov.to_string())
