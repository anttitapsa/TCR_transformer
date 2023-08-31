#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --out=logs/generate.out
#SBATCH --error=logs/generate.error


srun python3 -u generate.py 
