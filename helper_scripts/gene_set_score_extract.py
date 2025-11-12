import os
import pandas as pd
from tqdm import tqdm

# Input files
dominant_genes = '/wynton/home/capra/gyasu/capra_lab/cosmis/figure-code-data/fig_5c/data/all_dominant_uniprot.tsv'
recessive_genes = '/wynton/home/capra/gyasu/capra_lab/cosmis/figure-code-data/fig_5c/data/all_recessive_uniprot.tsv'
haploinsufficient_genes = '/wynton/home/capra/gyasu/capra_lab/cosmis/figure-code-data/fig_5c/data/clingen_level3_haploinsufficient_uniprot.tsv'
essential_genes = '/wynton/home/capra/gyasu/capra_lab/cosmis/figure-code-data/fig_5c/data/essential_genes_uniprot.tsv'
non_essential_genes = '/wynton/home/capra/gyasu/capra_lab/cosmis/figure-code-data/fig_5c/data/non_essential_genes_uniprot.tsv'
olfactory_genes = '/wynton/home/capra/gyasu/capra_lab/cosmis/figure-code-data/fig_5c/data/olfactory_receptor_uniprot.tsv'

# Class dictionary
class_dict = {
    'Olfactory': olfactory_genes, 
    'Non essential': non_essential_genes, 
    'Recessive': recessive_genes, 
    'Dominant': dominant_genes, 
    'Essential': essential_genes, 
    'Haploinsufficient': haploinsufficient_genes
}

# Directory containing HSC files
hsc_8_dir = '../outputs/HSC_8'

# Collect results here
dfs = []

# Iterate through each gene class
for gene_class, path in class_dict.items():
    gene_set_df = pd.read_csv(path, sep='\t', header=0)
    
    for uniprot_id in tqdm(gene_set_df['uniprot_id'], desc=f"Processing {gene_class}"):
        hsc_file = os.path.join(hsc_8_dir, f"{uniprot_id}_hsc.tsv")
        
        if os.path.exists(hsc_file):
            hsc_df = pd.read_csv(hsc_file, sep='\t', header=0)
            
            # Skip if HSCZ column missing or file empty
            if 'HSCZ' not in hsc_df.columns or hsc_df.empty:
                continue

            # Keep only the desired column and add identifiers
            subset_df = hsc_df[['HSCZ']].copy()
            subset_df['uniprot_id'] = uniprot_id
            subset_df['class'] = gene_class

            dfs.append(subset_df)

# Combine all into a single DataFrame
final_df = pd.concat(dfs, ignore_index=True)

# Reorder columns
final_df = final_df[['uniprot_id', 'HSCZ', 'class']]

# Save combined results
final_df.to_csv('gene_set_scores_HSC_8.tsv', sep='\t', index=False)

print(f"Combined DataFrame shape: {final_df.shape}")
