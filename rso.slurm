#!/bin/bash 
#SBATCH --ntasks=20
#SBATCH --time=24:00:00
#SBATCH --output=rso.txt
#SBATCH --error=erso.txt
#SBATCH --job-name=rso
cd /home/rcf-proj2/ma3/azizim/RSO/
source /usr/usc/python/3.6.0/setup.sh
python3 all_models.py 