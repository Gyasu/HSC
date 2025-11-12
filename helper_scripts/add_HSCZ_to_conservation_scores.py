import os
import pandas as pd
from tqdm import tqdm

# ===== USER INPUTS =====
input_csv = "all_conservation_scores_with_CS_8.csv"              # main CSV file with uniprot_id, uniprot_pos
hsc_dir = "../outputs/HSC_8"     # directory containing {uniprot_id}_hsc.tsv
output_csv = "all_conservation_scores_with_HSCZ_8.csv"   # output file

# ===== READ MAIN CSV =====
df = pd.read_csv(input_csv)

# Check required columns
required_cols = {"uniprot_id", "uniprot_pos"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Input CSV must contain columns: {required_cols}")

# Prepare column for scores
df["HSCZ"] = pd.NA

# ===== LOAD AND MERGE HSC SCORES =====
# Group by uniprot_id to avoid reopening the same file repeatedly
for uniprot_id, sub_df in tqdm(df.groupby("uniprot_id"), desc="Processing proteins"):
    hsc_file = os.path.join(hsc_dir, f"{uniprot_id}_hsc.tsv")
    if not os.path.exists(hsc_file):
        print(f"⚠️  Missing file: {hsc_file}")
        continue

    # Load HSC file
    try:
        hsc_df = pd.read_csv(hsc_file, sep="\t", usecols=["uniprot_pos", "HSCZ"])
    except Exception as e:
        print(f"Error reading {hsc_file}: {e}")
        continue

    # Merge on uniprot_pos
    merged = sub_df.merge(hsc_df, on="uniprot_pos", how="left", suffixes=("", "_new"))

    # Assign matched scores back to main df
    df.loc[merged.index, "HSCZ"] = merged["HSCZ_new"].values

# ===== SAVE OUTPUT =====
df.to_csv(output_csv, index=False)
print(f"\n✅ Saved merged data with HSCZ scores to: {output_csv}")
