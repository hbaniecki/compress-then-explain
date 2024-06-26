## setup environment

1. `conda env create -f env.yml -n cte`
2. `conda activate cte`
3. `pip install git+https://github.com/AI4LIFE-GROUP/OpenXAI --no-dependencies`
4. `pip install keras -U` // `keras>=3.0` needs to be installed after `conda install tensorflow==2.15.0` in `env.yml`

## run experiments

- `figure_2_and_4.py`, `figure_6.py`, `figure_3_and_9-12_cte.py` & `figure_3_and_9-12_gt.py` (minus `gmsc`) can be run on a personal computer
- results for the `gmsc` dataset in `figure_3_and_9-12_cte.py` & `figure_3_and_9-12_gt.py` may require a cluster / parallel computation
- cluster / slurm: `s_batch.py` runs `sbatch s_batch.sh` that runs `figure_5_7_8_13_15-24.py` for each dataset-method pair