import subprocess
import os
from typing import Literal

import distribution

def main(
    shards: int,
    dataset: Literal["purchase", "mnist"],
):
    script_path = os.path.join("datasets", dataset, "prepare_data.py")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)

    dataset_path = f"datasets/{dataset}/datasetfile"

    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

    ### INIT Script
    if not os.path.exists(f"containers/{shards}"):
        os.mkdir("containers") if not os.path.exists("containers") else None
        os.mkdir(f"containers/{shards}") if not os.path.exists(f"containers/{shards}") else None
        os.mkdir(f"containers/{shards}/cache") if not os.path.exists(f"containers/{shards}/cache") else None
        os.mkdir(f"containers/{shards}/times") if not os.path.exists(f"containers/{shards}/times") else None
        os.mkdir(f"containers/{shards}/outputs") if not os.path.exists(f"containers/{shards}/outputs") else None 
        with open(f"containers/{shards}/times/null.time", "w") as f:
            f.write(str(0))

    distribution.main(
        args_shards=int(shards),
        args_requests=None,
        distribution="uniform",
        container=f"{shards}",
        dataset=dataset_path,
        label="0",
    )

    for j in range(1, 16):
        r = j * shards // 5
        distribution.main(
            args_requests=r,
            distribution="uniform",
            container=f"{shards}",
            dataset=dataset_path,
            label=str(r),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards",
        default=None,
        type=int,
        help="Split the dataset in the given number of shards in an optimized manner (PLS-GAP partitionning) according to the given distribution, create the corresponding splitfile",
    )
    parser.add_argument(
        "--dataset",
        default="purchase",
        help="Dataset to prepare, out of: [purchase, mnist]",
    )

    args = parser.parse_args()
    main(
        shards=args.shards,
        dataset=args.dataset,
        )
