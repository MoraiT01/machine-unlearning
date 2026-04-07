import subprocess
import sys
import os
import glob
from typing import Literal

import aggregation, timestemps

def main(
    shards: int,
    strategy: Literal["uniform"] = "uniform",
    dataset: str = "datasets/purchase/datasetfile",
    ):
    # Create report file with header if it doesn't exist
    if not os.path.exists("general-report.csv"):
        with open("general-report.csv", "w") as f:
            f.write("nb_shards,nb_requests,accuracy,retraining_time\n")

    for j in range(16):
        r = j * shards // 5

        # Get accuracy from aggregation.py
        acc_result = aggregation.main(
            strategy="uniform",
            container=str(shards),
            dataset=dataset,
            label=str(r),
        )
        # acc = acc_result.stdout.strip()
        acc = acc_result

        # Concatenate shard time files into a single times file
        time_files = glob.glob(f"containers/{shards}/times/shard*-{r}.time")
        with open(f"containers/{shards}/times/times", "w") as out_f:
            for time_file in time_files:
                with open(time_file, "r") as in_f:
                    out_f.write(in_f.read())

        # Get retraining time from time.py
        time_result = timestemps.main(
            container=str(shards),
        )
        # time_val = time_result.stdout.strip().split(",")[0]
        time_val = time_result

        # Append results to report
        with open("general-report.csv", "a") as f:
            f.write(f"{shards},{r},{acc},{time_val}\n")

        print(f"j={j}, r={r}, acc={acc}, time={time_val}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards",
        default=None,
        type=int,
        help="Split the dataset in the given number of shards in an optimized manner (PLS-GAP partitionning) according to the given distribution, create the corresponding splitfile",
    )
    args = parser.parse_args()
    main(shards=args.shards)