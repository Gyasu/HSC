import os
from tqdm import tqdm
import pandas as pd
from statsmodels.stats.multitest import multipletests


scores_dir = '../outputs/HSC_8'
number_of_sites = 0
all_pvals = []

for file in tqdm(os.listdir(scores_dir)):
    file_path = os.path.join(scores_dir, file)
    if os.path.isfile(file_path) and file.endswith('.tsv'):
        df = pd.read_csv(file_path, sep='\t', header=0)
        number_of_sites += len(df)
        pvals = df['mis_p_value'].values
        all_pvals.extend(pvals)

print(f"Total number of residues is: {number_of_sites}")
all_pvals = pd.Series(all_pvals)
print(f"Total number of sites (with p-values): {len(all_pvals)}")


# --- Apply Benjamini–Hochberg FDR correction --- #
_, qvals, _, _ = multipletests(all_pvals, method='fdr_bh')

# --- Count significant sites --- #
significant = qvals < 0.05
num_significant = significant.sum()

print(f"Number of significantly constrained sites (FDR < 0.05): {num_significant}")
print(f"Proportion significant: {num_significant / len(all_pvals):.3f}")

