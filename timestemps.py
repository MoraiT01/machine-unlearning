import pandas as pd

# Compute stats based on the execution time (cumulated feed-forward + backprop.) of the shards

def main(
    container: str,
):
    t = pd.read_csv('containers/{}/times/times.tmp'.format(container), names=['time'])
    print('{},{}'.format(t['time'].sum(),t['time'].mean()))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--container', help="Name of the container")
    args = parser.parse_args()

    main(
        container=args.container,
    )