import subprocess
import sys
from typing import Literal

import sisa

def main(
    shards: int,
    model: Literal["purchase", "mnist"] = "purchase",
    dataset: str = "datasets/purchase/datasetfile",
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    optimizer: Literal["sgd"] = "sgd",
    chkpt_interval: int = 1,
):

    for i in range(shards):
        for j in range(16):
            r = j * shards // 5
            print(f"shard: {i+1}/{shards}, requests: {j+1}/16")
            
            sisa.main(
                model=model,
                train=True,
                slices=1,
                dataset=dataset,
                label=str(r),
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                optimizer=optimizer,
                chkpt_interval=chkpt_interval,
                container=str(shards),
                shard=i,
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
    args = parser.parse_args()
    main(shards=args.shards)