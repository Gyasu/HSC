#!/bin/bash
#
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l mem_free=16G
#$ -l scratch=48G
#$ -l h_rt=2:00:00
#$ -t 1-411

date
hostname

# Load your environment
module load CBI miniconda3/23.5.2-0-py311
conda activate constraintometer

# Determine the array task ID
i=$SGE_TASK_ID

input_file="all_uniprots/split_${i}.txt"
log_file="hsc8_${i}.log"

python hsc/main_final.py -c config.json -i $input_file -l $log_file -r 8


[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"