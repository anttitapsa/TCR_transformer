#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --mem=13G
#SBATCH --output=preprocess.out
#SBATCH --error=preprocess.error
#SBATCH --mail-user=antti.t.huttunen@aalto.fi
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

srun python3 -u process_data.py
