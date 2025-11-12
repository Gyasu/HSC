import pandas as pd
import os
from tqdm import tqdm

# -----------------------------
# File paths
# -----------------------------
clinvar_path = 'context_clinvar_with_HSCZ.csv'
hsc_dir = '../outputs/HSC_2'

# -----------------------------
# Load ClinVar + Context Data
# -----------------------------
clinvar_df = pd.read_csv(clinvar_path)

# Drop previous HSC or COSMIS-related columns if present
# cols_to_drop = ['cosmis-af', 'cosmis-af-z', 'cosmis-afz', 
#                 'HSCZ_2', 'HSCZ_5', 'HSCZ_8', 'HSCZ_11', 'HSCZ_14']
# clinvar_df = clinvar_df.drop(columns=[c for c in cols_to_drop if c in clinvar_df.columns], errors='ignore')

# Add an empty column to hold HSCZ_2 scores
clinvar_df['total_obs_af'] = pd.NA

# Ensure uniprot_pos is string for matching
clinvar_df['uniprot_pos'] = clinvar_df['uniprot_pos'].astype(str)

# -----------------------------
# Map HSC scores into ClinVar table
# -----------------------------
for idx, row in tqdm(clinvar_df.iterrows(), total=clinvar_df.shape[0], desc="Annotating HSC scores"):
    uniprot_id = row['uniprot_id']
    ref_aa = row['uniprot_aa']
    position = row['uniprot_pos']

    # Expected file name pattern
    hsc_file = f"{uniprot_id}_hsc.tsv"
    hsc_path = os.path.join(hsc_dir, hsc_file)

    # Load corresponding file
    try:
        hsc_df = pd.read_csv(hsc_path, sep='\t')
    except FileNotFoundError:
        # No file for this protein
        continue
    except pd.errors.EmptyDataError:
        print(f"⚠️ Empty file: {hsc_path}")
        continue
    except Exception as e:
        print(f"⚠️ Error reading {hsc_path}: {e}")
        continue

    # Make sure position is string for matching
    hsc_df['uniprot_pos'] = hsc_df['uniprot_pos'].astype(str)

    # Find matching row (same AA + same position)
    match = hsc_df[
        (hsc_df['uniprot_aa'] == ref_aa) &
        (hsc_df['uniprot_pos'] == position)
    ]

    # Assign score if found
    if not match.empty and 'log_cs_mis_obs' in match.columns:
        clinvar_df.at[idx, 'total_obs_af'] = match['log_cs_mis_obs'].values[0]
    # Optionally log missing matches
    # else:
    #     print(f"No match for {uniprot_id}:{ref_aa}{position}")

# -----------------------------
# Save updated file
# -----------------------------
output_path = 'context_clinvar_with_HSCZ.csv'
clinvar_df.to_csv(output_path, index=False)
print(f"\n✅ Annotated ClinVar table saved to: {output_path}")
