#!/bin/bash
#
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l mem_free=16G
#$ -l scratch=48G
#$ -l h_rt=10:00:00


date
hostname

module load CBI miniconda3/23.5.2-0-py311
conda activate constraintometer

python scripts/enst_to_mp.py -c config.json -t datafiles/enst_ids.txt -o mutation_counts.tsv -s syn_probs_vs_maf.tsv

