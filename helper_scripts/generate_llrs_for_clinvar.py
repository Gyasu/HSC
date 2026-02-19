import torch
import os
from peft import LoraConfig, get_peft_model
import pandas as pd
import numpy as np
from Bio import SeqIO
import gzip
from tqdm import tqdm
from transformers import EsmForMaskedLM, EsmTokenizer

# ========== FASTA PARSING HELPERS ==========

def get_uniprot_accession(record):
    """
    Extract 'P12345' from FASTA header like:
    sp|P12345|GENE_HUMAN ...
    """
    parts = record.id.split('|')
    return parts[1]

# Load UniProt FASTA → dict: {accession: SeqRecord}
fasta_path = "../datafiles/UP000005640_9606.fasta.gz"
with gzip.open(fasta_path, "rt") as handle:
    pep_dict = SeqIO.to_dict(
        SeqIO.parse(handle, "fasta"),
        key_function=get_uniprot_accession
    )

# ========== LLM SETUP ==========

AAorder = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']

model_name = "facebook/esm2_t30_150M_UR50D"
model = EsmForMaskedLM.from_pretrained(model_name)
tokenizer = EsmTokenizer.from_pretrained(model_name)

target_modules = ["query", "key", "value", "output.dense"]

lora_config = LoraConfig(
    task_type="MASKED_LM",
    r=8,
    lora_alpha=64,
    target_modules=target_modules,
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)

device = torch.device("cuda")
model_path = "../../capra_lab/esm-cosmis/ESM_150M_test/esm_finetuned_150M.pth"
state = torch.load(model_path, map_location=device)

model.load_state_dict(state, strict=True)
model.eval()
model.to(device)

# ========== LLR COMPUTATION ==========

def get_LLR_scores(input_ids, model, device, seq_list):
    with torch.no_grad():
        results = model(input_ids.to(device))
        logits = results.logits
        log_softmax_results = torch.log_softmax(logits, dim=-1)

        WTlogits = log_softmax_results[:, 1:-1, :].squeeze(0)

        WTlogits_df = pd.DataFrame(
            WTlogits[:, 4:24].cpu().numpy(),
            columns=AAorder,
            index=[f"{aa} {i+1}" for i, aa in enumerate(seq_list)]
        ).T

        wt_norm = np.diag(WTlogits_df.loc[[i.split(' ')[0] for i in WTlogits_df.columns]])
        LLR = WTlogits_df - wt_norm
        return LLR

# ========== LOAD CLINVAR FILE ==========

clinvar_df = pd.read_csv("../scripts/clinvar_esm_am.csv")
uniprot_ids = clinvar_df["main-isoform"].dropna().unique()

# ========== OUTPUT DIRECTORY ==========

output_dir = "LLR_output_clinvar_150M"
os.makedirs(output_dir, exist_ok=True)

# ========== MAIN LOOP ==========

for id in tqdm(uniprot_ids, desc="Generating LLRs for UniProt ids"):

    if os.path.exists(f"{output_dir}/{id}_LLR.csv"):
        continue

    if id not in pep_dict:
        print(f"[Warning] Sequence for {id} not found in FASTA. Skipping.")
        continue

    sequence = str(pep_dict[id].seq)
    seq_list = list(sequence)

    # Too long? skip (ESM2 limit)
    if len(sequence) > 1022:
        print(f"[Skip] {id} length {len(sequence)} > 1022.")
        continue

    tokenized = tokenizer(
        sequence,
        padding=False,
        truncation=True,
        max_length=1022,
        return_tensors="pt"
    )

    input_ids = tokenized["input_ids"]

    if input_ids.shape[1] - 2 != len(sequence):
        print("tokenization mismatch!")
        continue

    LLR = get_LLR_scores(input_ids, model, device, seq_list)

    out_path = os.path.join(output_dir, f"{id}_LLR.csv")
    LLR.to_csv(out_path)

    print(f"Saved LLR for {id} → {out_path}")
