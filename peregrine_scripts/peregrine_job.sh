#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --job-name=train_nli_model
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.m.sickert@student.rug.nl

module purge

module load Python/3.7.4-GCCcore-8.3.0

module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
# unfortunately we cannot use the optimized version of scikit-learn due to version conflicts with transformers package

source /data/$USER/.envs/ik_nlp/bin/activate

export HF_DATASETS_CACHE="/scratch/$USER/.cache/huggingface/datasets"

python -u main.py
