import os
import pandas as pd 

dms_summary_path = '/wynton/home/capra/gyasu/capra_lab/cosmis_eval/DMS_substitutions.csv'
llr_dir = '/wynton/home/capra/gyasu/capra_lab/esm-cosmis/LLRs/ESM_650M_LLRs'

dms_summary = pd.read_csv(dms_summary_path, header=0)

for idx, row in dms_summary.iterrows():
    taxon = row['taxon']

    if taxon == 'Human':
        real_uniprot = row['real_uniprot']
        llr_df = pd.read_csv(f"{llr_dir}/{real_uniprot}_LLR.csv", header=0, index=0)




