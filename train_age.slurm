#!/bin/bash
#SBATCH --job-name=age_train
#SBATCH --output=outputs/slurm_%j.log
#SBATCH --error=outputs/slurm_%j.err
#SBATCH --time=1-10:00:00
#SBATCH --partition=debug
#SBATCH --qos=hpc_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@domain.com

# ── Environment ────────────────────────────────────────────────
source ~/projects/gender_recognition/env/bin/activate
cd ~/projects/gender_recognition

# Cache directories (keep pre-downloaded weights inside the project)
export HF_HOME=$PWD/cache/huggingface
export TIMM_HOME=$PWD/cache/timm

# Prevent CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p outputs

# ── Run ────────────────────────────────────────────────────────
# To resume a crashed/timed-out job, uncomment --resume below and resubmit.
python gender_recognition.py \
    --data        age-prediction \
    --epochs      20 \
    --batch_size  48 \
    --num_workers 2 \
    --seed        42 \
    --output_dir  outputs
    # --resume    outputs/checkpoint_last.pth
