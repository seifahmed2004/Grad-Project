#!/bin/bash
#SBATCH --job-name=tts_train
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/slurm/zewgp003/projects/tts_project/logs/output_%j.log

source ~/.bashrc
conda activate tts_env

cd /nfs/slurm/zewgp003/projects/tts_project/code

python train_tts_hpc.py \
  --work-dir /nfs/slurm/zewgp003/projects/tts_project \
  --data-dir /nfs/slurm/zewgp003/projects/tts_project/data/LJSpeech-1.1 \
  --epochs 1 \
  --batch-size 16 \
  --num-workers 2
