import subprocess
import pandas as pd

DATASETS = pd.read_csv("table2.csv")

METHODS = [
    "compress", 
    "sample1", 
    "sample20",
    "cluster",
    "sample2", 
    "sample3", 
    "sample4",
]

PATH_OUTPUT = "/metadata"

START = 0

for iter, row in DATASETS.iterrows():
    for method in METHODS:
        if method == "sample20":
            STOP = 3
            subprocess.run(["sbatch", "s_batch.sh", str(row['task_id']), method, str(START), str(STOP), PATH_OUTPUT])
        else:
            STOP = 33
            subprocess.run(["sbatch", "s_batch.sh", str(row['task_id']), method, str(START), str(STOP), PATH_OUTPUT])