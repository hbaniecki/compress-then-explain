#!/bin/bash

#SBATCH --job-name=cte
#SBATCH --gpus=0
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20G
#SBATCH --output=logs/log_%j.log 
#SBATCH --time=12-00:00:00

set -e
hostname; pwd; date

conda activate cte

date

python figure_5_7_8_13_15-24.py --dataset $1 --method $2 --start $3 --stop $4 --output $5

date