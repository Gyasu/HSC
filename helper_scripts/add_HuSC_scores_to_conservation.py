import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Paths
husc_dir = Path("../outputs/HuSC_8")

# Read main dataframe
df = pd.read_csv("all_conservation_scores_with_CS_8.csv", header=0)

df = df.drop(columns = ['CS_8'])

# Initialize HuSC column
df["HuSC"] = pd.NA

# Process per UniProt ID
for uniprot_id, idx in tqdm(df.groupby("uniprot_id").groups.items()):
    husc_file = husc_dir / f"{uniprot_id}_husc.tsv"

    if not husc_file.exists():
        print(f"Missing HuSC file for {uniprot_id}")
        continue

    # Read HuSC file
    husc_df = pd.read_csv(husc_file, sep="\t")

    # Map uniprot_pos → HuSC
    pos_to_husc = husc_df.set_index("uniprot_pos")["HuSC"]

    # Assign values
    df.loc[idx, "HuSC"] = df.loc[idx, "uniprot_pos"].map(pos_to_husc)


df.to_csv("all_conservation_scores_with_HuSC.csv")