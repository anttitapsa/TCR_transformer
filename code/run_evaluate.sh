#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --out=logs/evaluate.out
#SBATCH --error=logs/evaluate.error

srun python3 -u evaluate.py 