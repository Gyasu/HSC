import pandas as pd
import os
from tqdm import tqdm

# -----------------------------
# Paths
# -----------------------------
CLINVAR_PATH = "context_clinvar_with_HuSC.csv"
HUSC_DIR = "../outputs/HuSC_14"
OUTPUT_PATH = "context_clinvar_with_HuSC.csv"

# -----------------------------
# Load ClinVar data
# -----------------------------
clinvar_df = pd.read_csv(CLINVAR_PATH)

# Remove old columns if re-running
cols_to_drop = [c for c in clinvar_df.columns if c.lower().startswith("hsc")]
clinvar_df = clinvar_df.drop(columns=cols_to_drop, errors="ignore")

# Initialize HuSC column
clinvar_df["HuSC_14"] = pd.NA

# Ensure consistent dtypes
clinvar_df["uniprot_pos"] = clinvar_df["uniprot_pos"].astype(str)

# -----------------------------
# Preload all HuSC files
# -----------------------------
husc_lookup = {}

for fname in tqdm(os.listdir(HUSC_DIR), desc="Loading HuSC files"):
    if not fname.endswith("_husc.tsv"):
        continue

    uniprot_id = fname.replace("_husc.tsv", "")
    fpath = os.path.join(HUSC_DIR, fname)

    try:
        df = pd.read_csv(fpath, sep="\t")
    except Exception as e:
        print(f"⚠️ Skipping {fname}: {e}")
        continue

    if "HuSC" not in df.columns:
        continue

    df["uniprot_pos"] = df["uniprot_pos"].astype(str)

    # Build lookup: (AA, pos) → HuSC
    husc_lookup[uniprot_id] = {
        (row.uniprot_aa, row.uniprot_pos): row.HuSC
        for row in df.itertuples(index=False)
    }

# -----------------------------
# Annotate ClinVar
# -----------------------------
for idx, row in tqdm(
    clinvar_df.iterrows(),
    total=len(clinvar_df),
    desc="Annotating HuSC scores"
):
    uniprot_id = row.uniprot_id
    key = (row.uniprot_aa, row.uniprot_pos)

    if uniprot_id in husc_lookup:
        clinvar_df.at[idx, "HuSC_14"] = husc_lookup[uniprot_id].get(key)

# -----------------------------
# Save result
# -----------------------------
clinvar_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Annotated ClinVar table saved to: {OUTPUT_PATH}")
