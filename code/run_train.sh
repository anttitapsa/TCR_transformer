#!/bin/bash
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=50G
#SBATCH --out=logs/train.out
#SBATCH --error=logs/train.error
#SBATCH --mail-user=<Your email>
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL

srun python3 -u train.py 
