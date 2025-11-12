import os
import pandas as pd
from tqdm import tqdm

output_dir = '../outputs/HSC_11'
output_file = 'hsc_11_all_scores.txt'

hsc_scores = []

for file in tqdm(os.listdir(output_dir)):
    if file.endswith('_hsc.tsv'):
        filepath = os.path.join(output_dir, file)
        hsc_df = pd.read_csv(filepath, sep='\t')
        if 'HSCZ' in hsc_df.columns:
            hsc_scores.extend(hsc_df['HSCZ'].tolist())

with open(output_file, 'w') as f:
    for score in hsc_scores:
        f.write(f"{score}\n")

print(f"Wrote {len(hsc_scores)} scores to {output_file}")
